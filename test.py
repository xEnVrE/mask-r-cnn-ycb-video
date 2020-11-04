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
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

dataset = BOPDatasetFull('./dataset_full.pickle', False)
DatasetCatalog.register('bop_ycbv_pbr', lambda : dataset.dataset())
MetadataCatalog.get('bop_ycbv_pbr').set(thing_classes = dataset.class_list())
metadata = MetadataCatalog.get('bop_ycbv_pbr')

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.DATASETS.TRAIN = ('bop_ycbv_pbr',)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.INPUT.MASK_FORMAT = 'bitmask'
#cfg.MODEL.WEIGHTS = './coco_mask_rcnn_R_50_FPN_3x_60k_unfrozen/model_final.pth'
cfg.MODEL.WEIGHTS = './coco_mask_rcnn_R_50_FPN_3x_120k_unfrozen_augm/model_0059999.pth'
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER= 300
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(dataset.class_list())

predictor = DefaultPredictor(cfg)

for dir in sorted(glob('/home/IIT.LOCAL/npiga/robot-code/synthetic-ycb-video-dataset/object_motion/used/0*/')):
# for i in range(1):
    print('Processing dir ' + dir)
    # object_name = '_'.join(dir.split('/')[-2].split('_')[:-1])
    object_name = dir.split('/')[-2]
    output_path_720 = './masks/' + object_name + '_720p/masks/mrcnn/'
    output_path_640 = './masks/' + object_name + '_640p/masks/mrcnn/'
    os.makedirs(output_path_720, exist_ok = True)
    os.makedirs(output_path_640, exist_ok = True)
    print(object_name)

    for image in tqdm(sorted(glob(dir + '/rgb/*.png'))):
        index = image.split('/')[-1].split('.')[0]
        im = cv2.imread(image)
        im_extended = numpy.zeros((960, 1280, 3))
        im_extended[120:720 + 120, :, :] = im
        im = cv2.resize(im_extended, (640, 480))
        outputs = predictor(im)
        instances = outputs['instances'].to('cpu')
        classes_id = instances.pred_classes.numpy()
        list_detections = []
        masks = instances.pred_masks.numpy()
        mask = numpy.zeros((480, 640), dtype = numpy.uint8)
        for i, id in enumerate(classes_id):
            if dataset.class_list()[id] == object_name.split('_')[0]:
                mask[masks[i, :, :]] = 255
        # v = Visualizer(im[:, :, ::-1],
        #                metadata=metadata,
        #                scale=1.0,
        #                instance_mode=ColorMode.IMAGE_BW
        # )
        # out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        # out = out.get_image()[:, :, ::-1]
        # out = cv2.resize(out, (1280, 960))
        # out_reduced = numpy.zeros((720, 1280, 3))
        # out_reduced = out[120:720 + 120, :, :]
        # cv2.imwrite(output_path + object_name + '_' + str(index) + '.png', out_reduced)

        if object_name == '003_cracker_box_real':
            object_name = '003_cracker_box'
        if object_name == '004_sugar_box_real':
            object_name = '004_sugar_box'
        elif object_name == '006_mustard_bottle_real':
            object_name = '006_mustard_bottle'
        elif object_name == '010_potted_meat_can_real':
            object_name = '010_potted_meat_can'
        
        cv2.imwrite(output_path_640 + object_name + '_' + str(index) + '.png', mask)

        out = cv2.resize(mask, (1280, 960))
        out_reduced = numpy.zeros((720, 1280))
        out_reduced = out[120:720 + 120, :]
        cv2.imwrite(output_path_720 + object_name + '_' + str(index) + '.png', out_reduced)
