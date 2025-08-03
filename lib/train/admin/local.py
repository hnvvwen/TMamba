class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/root/autodl-tmp/'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/root/autodl-tmp/tensorboard'  # Directory for tensorboard files.
        self.pretrained_networks = '/root/autodl-tmp/pretrained_networks'
        self.lasot_dir = '/root/autodl-tmp/lasot/'
        self.got10k_dir = '/root/autodl-tmp/GOT-10k/train/'
        self.got10k_val_dir = '/root/autodl-tmp/GOT-10k/val'
        self.lasot_lmdb_dir = '/home/zhouliyao/projects/pythonprojects/OSMTrack-main/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/zhouliyao/projects/pythonprojects/OSMTrack-main/data/got10k_lmdb'
        self.trackingnet_dir = '/root/autodl-tmp/trackingnet/'
        self.trackingnet_lmdb_dir = '/home/zhouliyao/projects/pythonprojects/OSMTrack-main/data/trackingnet_lmdb'
        self.coco_dir = '/root/autodl-tmp/coco/'
        self.coco_lmdb_dir = '/home/zhouliyao/projects/pythonprojects/OSMTrack-main/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/zhouliyao/projects/datasets/VID/'
        self.imagenet_lmdb_dir = '/home/zhouliyao/projects/pythonprojects/OSMTrack-main/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
