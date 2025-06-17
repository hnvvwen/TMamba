# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from collections import namedtuple

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from lib.models.trackingmamba.rope import *
import random
from lib.models.layers.head import build_box_head
import importlib
import lib.train.admin.settings as ws_settings
from .utils import combine_tokens, recover_tokens

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
#new
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

from lib.models.trackingmamba.selective_scan_interface import selective_scan_fn, mamba_inner_fn   

import torch.nn.functional as F
#from lib.models.trackingmamba.backbone import build_backbone,Backbone
#from lib.utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                       accuracy, get_world_size, interpolate,
#                       is_dist_avail_and_initialized, inverse_sigmoid)
#from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d



__all__ = [
    'vim_tiny_patch16_224', 'vim_small_patch16_224', 'vim_base_patch16_224',
    'vim_tiny_patch16_384', 'vim_small_patch16_384', 'vim_base_patch16_384',
]


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

#####new
class MultiScaleCausalConv(nn.Module):
    def __init__(self, dim, kernel_sizes=[3,5,7], activation=nn.SiLU()):
        super().__init__()
        self.dim = dim
        self.kernel_sizes = kernel_sizes
        self.activation = activation

        # 第一层多尺度因果卷积
        self.convs1 = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=k, groups=dim, padding=0) for k in kernel_sizes
        ])
        self.weights1 = nn.Parameter(torch.ones(len(kernel_sizes)))

        # 第二层多尺度因果卷积
        self.convs2 = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=k, groups=dim, padding=0) for k in kernel_sizes
        ])
        self.weights2 = nn.Parameter(torch.ones(len(kernel_sizes)))

    def forward(self, x):
        # x: (batch, dim, length)

        # 第一层多尺度卷积
        outs1 = []
        for conv, k in zip(self.convs1, self.kernel_sizes):
            pad = (k - 1, 0)
            x_padded = F.pad(x, pad)
            out = conv(x_padded)
            outs1.append(out)
        stacked1 = torch.stack(outs1, dim=0)
        weight1 = F.softmax(self.weights1, dim=0).view(-1,1,1,1)
        out1 = (stacked1 * weight1).sum(dim=0)
        out1 = self.activation(out1)

        # 第二层多尺度卷积（对第一层结果做同样处理）
        outs2 = []
        for conv, k in zip(self.convs2, self.kernel_sizes):
            pad = (k - 1, 0)
            out1_padded = F.pad(out1, pad)
            out = conv(out1_padded)
            outs2.append(out)
        stacked2 = torch.stack(outs2, dim=0)
        weight2 = F.softmax(self.weights2, dim=0).view(-1,1,1,1)
        out2 = (stacked2 * weight2).sum(dim=0)
        out2 = self.activation(out2)

        return out2
#class MultiScaleCausalConv(nn.Module):
#    def __init__(self, dim, kernel_sizes=[3, 5, 7], activation=nn.SiLU()):
#        super().__init__()
#        self.dim = dim
#        self.kernel_sizes = kernel_sizes
#        self.activation = activation
#
#        # 第一层多尺度普通卷积，padding保证输出长度不变 (padding = (k-1)//2)
#        self.convs1 = nn.ModuleList([
#            nn.Conv1d(dim, dim, kernel_size=k, padding=(k - 1) // 2, groups=dim) for k in kernel_sizes
#        ])
#        self.weights1 = nn.Parameter(torch.ones(len(kernel_sizes)))
#
#        # 第二层多尺度普通卷积
#        self.convs2 = nn.ModuleList([
#            nn.Conv1d(dim, dim, kernel_size=k, padding=(k - 1) // 2, groups=dim) for k in kernel_sizes
#        ])
#        self.weights2 = nn.Parameter(torch.ones(len(kernel_sizes)))
#
#    def forward(self, x):
#        # x: (batch, dim, length)
#
#        # 第一层多尺度卷积
#        outs1 = []
#        for conv in self.convs1:
#            out = conv(x)
#            outs1.append(out)
#        stacked1 = torch.stack(outs1, dim=0)  # (n_kernels, batch, dim, length)
#        weight1 = F.softmax(self.weights1, dim=0).view(-1, 1, 1, 1)
#        out1 = (stacked1 * weight1).sum(dim=0)
#        out1 = self.activation(out1)
#
#        # 第二层多尺度卷积
#        outs2 = []
#        for conv in self.convs2:
#            out = conv(out1)
#            outs2.append(out)
#        stacked2 = torch.stack(outs2, dim=0)
#        weight2 = F.softmax(self.weights2, dim=0).view(-1, 1, 1, 1)
#        out2 = (stacked2 * weight2).sum(dim=0)
#        out2 = self.activation(out2)
#
#        return out2
class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



#####new


class VisionMamba(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16,
                 stride=16,
                 depth=24, 
                 embed_dim=192, 
                 channels=3, 
                 num_classes=1000,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=False,
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="none",
                 if_cls_token=False,
                 if_devide_out=False,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=False,
                 #new

                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        # self.if_cls_token = if_cls_token
        self.if_cls_token = False
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models  192

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches


##new
        
        self.d_inner = int(1 * self.d_model)  #384
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=4,
            groups=self.d_inner,
            padding=4 - 1,
            **factory_kwargs,
        )
        bias=False,
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.activation = "silu"
        dt_rank="auto"
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        d_state=16
        self.d_state = d_state
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32  状态空间参数化

        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.multi_scale_conv =  MultiScaleCausalConv(384)
#        num_queries=100
#        self.query_embed = nn.Embedding(num_queries, embed_dim*2) #num_queries在coco中常设为100
#        self.num_feature_levels = 4 # 决定了模型会使用几个不同分辨率的特征图（如 P3、P4、P5、P6...）来进行多尺度可变形注意力。
#        self.input_proj = nn.ModuleList([
#                nn.Sequential(
#                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
#                    nn.GroupNorm(32, hidden_dim),
#                )])
#        self.backbone = backbone

#普通attention
#        self.mixer = Attention(
#            embed_dim,
#            num_heads=8,
#            qkv_bias=False,
#            qk_norm=False,
#            attn_drop=0.,
#            proj_drop=0.,
#            norm_layer=nn.LayerNorm,
#            )
#        layer_scale=None
#        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
#        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(embed_dim))  if use_layer_scale else 1
#        drop_path=0.
#        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#        norm_layer=nn.LayerNorm
#        self.norm1 = norm_layer(embed_dim)
##new 

        
        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                # self.num_tokens = 1
            
        if if_abs_pos_embed:
            # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_embed_x = nn.Parameter(torch.zeros(1, 256, self.embed_dim))
            self.pos_embed_z = nn.Parameter(torch.zeros(1, 64, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # settings = ws_settings.Settings()
        # config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
        # cfg = config_module.cfg
        # self.head = build_box_head(cfg, embed_dim)


        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
                # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # original init
        self.patch_embed.apply(segm_init_weights)
        # self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed_x, std=.02)
            trunc_normal_(self.pos_embed_z, std=.02)
            
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, z, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        x = self.patch_embed(x)                  #x.shape = torch.Size([B, 3, 256, 256])  -> torch.Size([2, 256, 384])
        z = self.patch_embed(z)                  #z.shape = torch.Size([B, 3, 128, 128])  -> torch.Size([2, 64, 384])
        B, M, _ = x.shape
       
        if self.if_cls_token:                 # False
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                M = x.shape[1]
            else:
                if self.use_middle_cls_token:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = M // 2
                    # add cls token in the middle
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)       
                elif if_random_cls_token_position:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = random.randint(0, M)
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                    print("token_position: ", token_position)
                else:
                    cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
                M = x.shape[1]                 
       
        if self.if_abs_pos_embed:                  # True 
            x = x + self.pos_embed_x               # x = x + positon_embemding =torch.Size([B, 256, 384]) + torch.Size([1, 256, 384]) = torch.Size([B, 256, 384])
            z = z + self.pos_embed_z               # z = z + positon_embemding =torch.Size([B, 64, 384]) + torch.Size([1, 64, 384]) = torch.Size([B, 64, 384])
            x = torch.cat((z, x), dim=1)           # torch.Size([B, 320, 384])
            x = self.pos_drop(x)                   # x.shape = torch.Size([B, 320, 384])
            
        if if_random_token_rank:                   #False
           
            shuffle_indices = torch.randperm(M)

            if isinstance(token_position, list):
                print("original value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("original value: ", x[0, token_position, 0])
            print("original token_position: ", token_position)

            x = x[:, shuffle_indices, :]

            if isinstance(token_position, list):
                new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in range(len(token_position))]
                token_position = new_token_position
            else:
                token_position = torch.where(shuffle_indices == token_position)[0].item()

            if isinstance(token_position, list):
                print("new value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("new value: ", x[0, token_position, 0])
            print("new token_position: ", token_position)


        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:        # False
            x = x.flip([1])
            if_flip_img_sequences = True
#new   
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        batch, seqlen, dim = x.shape
        conv_state, ssm_state = None, None
        if inference_params is not None:  #在推理（如增量生成）阶段，如果当前 token 不是序列的第一个 token（seqlen_offset > 0），则直接调用 step() 处理当前 token，而不是走完整个 forward 路径，提高效率。
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(x, conv_state, ssm_state)
                return out
        assert self.activation in ["silu", "swish"]
        #print("x shape:", x.shape)#x shape: torch.Size([100, 320, 384])

        #print("original weight shape:", self.conv1d.weight.shape) #original weight shape: torch.Size([768, 1, 4])
        #print("rearranged weight shape:", rearrange(self.conv1d.weight, "d 1 w -> d w").shape)#rearranged weight shape: torch.Size([768, 4])
        x = x.permute(0, 2, 1)  # 从 (B, L, D) -> (B, D, L)
        z1=x
        
#        x = causal_conv1d_fn(
#                x=x,
#                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
#                bias=self.conv1d.bias,
#                activation=self.activation,
#            )
        x = self.multi_scale_conv(x)
        #print("x_con",x.shape)
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d) 状态空间参数构造（输入相关）
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        
        #print("z before permute:", z.shape)  # torch.Size([B, L, D]) ?
        
        
        #print("z1 after permute:", z1.shape)
        #print("x after shape",x.shape)
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z1,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        y = rearrange(y, "b d l -> b l d")  
        x = self.out_proj(y)                

        
#new
    
        # mamba impl
        residual = None
        hidden_states = x
        
        
        if not self.if_bidirectional:                                 # True
            for layer in self.layers:
                if if_flip_img_sequences and self.if_rope:            # False
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # rope about
                if self.if_rope:                                       # False
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:             # False
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
        
        else:             # False
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])
            
        
          
        if not self.fused_add_norm:         #False
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:       #True
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(                                         # hidden_states.shape = torch.Size([B, 320, 384])
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
###new
         
#        backbone =Backbone('resnet50', train_backbone=False, return_interm_layers=True, dilation=True)
#        samples = nested_tensor_from_tensor_list(samples)
#        features, pos = self.backbone(samples)
#        srcs = []
#        masks = []
#        for l, feat in enumerate(features):
#            src, mask = feat.decompose()
#            srcs.append(self.input_proj[l](src))
#            masks.append(mask)
#            assert mask is not None
#            
#        if self.num_feature_levels > len(srcs):
#            _len_srcs = len(srcs)
#            for l in range(_len_srcs, self.num_feature_levels):
#                if l == _len_srcs:
#                    src = self.input_proj[l](features[-1].tensors)
#                else:
#                    src = self.input_proj[l](srcs[-1])
#                m = samples.mask
#                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
#                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
#                srcs.append(src)
#                masks.append(mask)
#                pos.append(pos_l)
#         query_embeds = None
#         query_embeds = self.query_embed.weight
#         hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)
#
#        outputs_classes = []
#        outputs_coords = []
#        for lvl in range(hs.shape[0]):
#            if lvl == 0:
#                reference = init_reference
#            else:
#                reference = inter_references[lvl - 1]
#            reference = inverse_sigmoid(reference)
#            outputs_class = self.class_embed[lvl](hs[lvl])
#            tmp = self.bbox_embed[lvl](hs[lvl])
#            if reference.shape[-1] == 4:
#                tmp += reference
#            else:
#                assert reference.shape[-1] == 2
#                tmp[..., :2] += reference
#            outputs_coord = tmp.sigmoid()
#            outputs_classes.append(outputs_class)
#            outputs_coords.append(outputs_coord)
#        outputs_class = torch.stack(outputs_classes)
#        outputs_coord = torch.stack(outputs_coords)
#
#        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
#        
#        out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
#
#        
#        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
#        out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}



###new
        #hidden_states = hidden_states + self.drop_path(self.gamma_1 * self.mixer(self.norm1(hidden_states))) 
        # return only cls token if it exists
        if self.if_cls_token:          #False
            if self.use_double_cls_token:
                return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
            else:
                if self.use_middle_cls_token:
                    return hidden_states[:, token_position, :]
                elif if_random_cls_token_position:
                    return hidden_states[:, token_position, :]
                else:
                    return hidden_states[:, token_position, :]

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':         #True
            return hidden_states.mean(dim=1)         #hidden_states.shape = torch.Size([2, 384])
        elif self.final_pool_type == 'max':
            return hidden_states
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        x = self.forward_features(x, inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank)
        if return_features:
            return x
        x = self.head(x)
        if self.final_pool_type == 'max':
            x = x.max(dim=1)[0]
        return x

#new
    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state    
#new

@register_model
def vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, stride=8, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='all', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    
    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="to.do",
        #     map_location="cpu", check_hash=True
        # )
        # model.load_state_dict(checkpoint["model"])
        checkpoint = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
        print('Load pretrained model from: ' + pretrained)
    return model

@register_model
def vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, stride=8, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


