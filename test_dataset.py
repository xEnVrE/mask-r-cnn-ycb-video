import cv2
import detectron2
import os
import random
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from dataset_full import BOPDatasetFull
from detectron2.utils.visualizer import ColorMode

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                     
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   

dataset = BOPDatasetFull('./dataset_full.pickle', False)
DatasetCatalog.register('bop_ycbv_pbr', lambda : dataset.dataset())
MetadataCatalog.get('bop_ycbv_pbr').set(thing_classes = dataset.class_list())
metadata = MetadataCatalog.get('bop_ycbv_pbr')

counter = 0
for d in random.sample(dataset.dataset(), 3):    
    im = cv2.imread(d["file_name"])
    visualizer = Visualizer(im[:, :, ::-1], metadata=metadata, scale = 1.0)
    out = visualizer.draw_dataset_dict(d)
    cv2.imwrite('./dataset_test_' + str(counter) + '.png', out.get_image()[:, :, ::-1])
    counter += 1

