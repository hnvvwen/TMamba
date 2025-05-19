class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/zhouliyao/projects/pythonprojects/OSMTrack-main'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/zhouliyao/projects/pythonprojects/OSMTrack-main/tensorboard'  # Directory for tensorboard files.
        self.pretrained_networks = '/home/zhouliyao/projects/pythonprojects/OSMTrack-main/pretrained_networks'
        self.lasot_dir = '/home/zhouliyao/projects/datasets/LaSOT/'
        self.got10k_dir = '/home/zhouliyao/projects/datasets/GOT-10k/train_data/'
        self.got10k_val_dir = '/home/zhouliyao/projects/pythonprojects/OSMTrack-main/data/got10k/val'
        self.lasot_lmdb_dir = '/home/zhouliyao/projects/pythonprojects/OSMTrack-main/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/zhouliyao/projects/pythonprojects/OSMTrack-main/data/got10k_lmdb'
        self.trackingnet_dir = '/home/zhouliyao/projects/datasets/trackingnet/'
        self.trackingnet_lmdb_dir = '/home/zhouliyao/projects/pythonprojects/OSMTrack-main/data/trackingnet_lmdb'
        self.coco_dir = '/home/zhouliyao/projects/datasets/coco/'
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
