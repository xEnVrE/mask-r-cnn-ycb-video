import copy
import detectron2
import os
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data.transforms.augmentation import Augmentation
from fvcore.transforms.transform import Transform, NoOpTransform
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import DefaultTrainer
import detectron2.data.detection_utils as utils
from dataset_mapper import Mapper
from dataset_full import BOPDatasetFull
# from typing import List, Optional, Union

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                         
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   

dataset = BOPDatasetFull('./dataset_full.pickle', False)
DatasetCatalog.register('bop_ycbv_pbr', lambda : dataset.dataset())
MetadataCatalog.get('bop_ycbv_pbr').set(thing_classes = dataset.class_list())
metadata = MetadataCatalog.get('bop_ycbv_pbr')

class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=Mapper(cfg, True))

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.DATASETS.TRAIN = ('bop_ycbv_pbr',)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.INPUT.MASK_FORMAT = 'bitmask'
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')  # Let training initialize from model zoo
cfg.OUTPUT_DIR = './coco_mask_rcnn_R_50_FPN_3x_120k_unfrozen_augm'
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER= 120000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(dataset.class_list())
cfg.MODEL.BACKBONE.FREEZE_AT = 0

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

tr = Trainer(cfg)
tr.resume_or_load(resume=False)
tr.train()

# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()
