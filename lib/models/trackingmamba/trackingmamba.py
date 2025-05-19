"""
Basic vim model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.trackingmamba.models_mamba import create_block
from timm.models import create_model
import torch.nn.functional as F
#from lib.models.osmtrack.mamba_cross import CrossMamba
from thop import profile
from lib.config.trackingmamba.config import cfg


class TrackingMamba(nn.Module):
    """ This is the base class for vim """

    def __init__(self, visionmamba, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = visionmamba
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
      
        x = self.backbone.forward_features( z=template, x=search,
                                                inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False)

        # Forward head
        search_feature = x[:, -self.feat_len_s:]
        x = search_feature
        feat_last = search_feature
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)
       
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        search_feature = cat_feature
        opt = (search_feature.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
           
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_trackingmamba(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('TrackingMamba' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    backbone = create_model( model_name= cfg.MODEL.BACKBONE.TYPE, pretrained= pretrained, num_classes=1000,
            drop_rate=0.0, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, drop_block_rate=None, img_size=256
            )
    hidden_dim = 384
    box_head = build_box_head(cfg, hidden_dim)
    model = TrackingMamba(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )
   
    if 'OSMTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
    #print(1/0)
    return model

if __name__ == '__main__':
    net = build_trackingmamba(cfg)
    net = net.cuda()
    var1 = torch.Tensor(1, 3, 128, 128).cuda()
    var2 = torch.Tensor(1, 3, 256, 256).cuda()

    out = net(var1, var2)
    print("over")

   
