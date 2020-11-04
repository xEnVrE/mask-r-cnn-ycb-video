import cv2
import detectron2
import numpy
import os
import random
from tqdm import tqdm
from glob import glob
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from dataset_full import BOPDatasetFull
from detectron2.utils.visualizer import ColorMode

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

dataset = BOPDatasetFull('./dataset_full.pickle', False)
DatasetCatalog.register('bop_ycbv_pbr', lambda : dataset.dataset())
MetadataCatalog.get('bop_ycbv_pbr').set(thing_classes = dataset.class_list())
metadata = MetadataCatalog.get('bop_ycbv_pbr')
metadata.thing_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for item in dataset.class_list()]

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.DATASETS.TRAIN = ('bop_ycbv_pbr',)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.INPUT.MASK_FORMAT = 'bitmask'
cfg.MODEL.WEIGHTS = './coco_mask_rcnn_R_50_FPN_3x_60k_unfrozen/model_final.pth'
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER= 300
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(dataset.class_list())

predictor = DefaultPredictor(cfg)

# for dir in sorted(glob('/home/IIT.LOCAL/npiga/ycb_real_test/*/')):
for i in range(1):
    dir = '/home/IIT.LOCAL/npiga/robot-code/synthetic-ycb-video-dataset/object_motion/003_cracker_box_real/rgb/'
    for image in tqdm(sorted(glob(dir + '/*.png'))):
        index = image.split('/')[-1].split('.')[0]
        im = cv2.imread(image)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
        out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        out = out.get_image()[:, :, ::-1]
        # cv2.imwrite(dir + '/' + str(index) + '_masks.png', out)
        cv2.imwrite('./test/' + str(index) + '_masks.png', out)
