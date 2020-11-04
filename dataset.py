import argparse
import cv2
import json
import numpy
import os
import pycocotools
import pickle
from glob import glob
from detectron2.structures import BoxMode
from tqdm import tqdm

class BOPDataset:
    def __init__(self, path, do_generate):
        self.path = path
        self.do_generate = do_generate
        
        self.dataset_dicts = []

        if self.do_generate:
            self.generate()
            self.save()
        else:
            self.load(self.path)

        # self.classes = ['002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '019', '021', '024', '025', '035', '036', '037', '040', '051', '052', '061']
        self.classes = ['002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '019', '021', '024', '025', '035', '037', '040', '051', '052', '061']

    def class_list(self):
        return self.classes
        
    def dataset(self):
        return self.dataset_dicts

    def generate(self):
        subs = sorted(glob(self.path + '/*/'))
        # subs = sorted(glob(self.path + '/000000/'))

        idx = 0
        for sub in subs:
            print('Processing subfolder ' + sub)

            # print('Loading scene_gt.json')
            f = open(sub + 'scene_gt.json')
            scene_gt = json.load(f)
            f.close()

            # print('Loading scene_gt_info.json')
            f = open(sub + 'scene_gt_info.json')
            scene_gt_info = json.load(f)
            f.close()

            for i in tqdm(range(len(scene_gt))):
                gt_container = scene_gt[str(i)]
                gt_info_container = scene_gt_info[str(i)]

                rgb_path = os.path.abspath(sub + '/rgb/' + str(i).zfill(6) + '.jpg')

                record = {}

                record['file_name'] = rgb_path
                record['image_id'] = idx
                record['height'] = 480
                record['width'] = 640

                annotations = []
                for j in range(len(gt_container)):
                    if gt_info_container[j]['px_count_visib'] == 0:
                        continue
                    bbox = gt_info_container[j]['bbox_visib']

                    object_id = gt_container[j]['obj_id'] - 1
                    if object_id == 16:
                        continue
                    mask_path = os.path.abspath(sub + '/mask_visib/' + str(i).zfill(6) + '_' + str(j).zfill(6) +  '.png')

                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
                    mask = numpy.uint8(mask)

                    obj = {
                        'bbox' : [bbox[0], bbox[1], bbox[2], bbox[3]],
                        'bbox_mode': BoxMode.XYWH_ABS,
                        'segmentation' : pycocotools.mask.encode(numpy.asarray(mask, order='F')),
                        'category_id' : object_id
                        }
                    annotations.append(obj)
                record['annotations'] = annotations
                self.dataset_dicts.append(record)

                idx += 1

    def save(self):
        with open('./dataset.pickle', 'wb') as handle:
            pickle.dump(self.dataset_dicts, handle)

    def load(self, path):
        with open(path, 'rb') as handle:
            self.dataset_dicts = pickle.load(handle)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type = str, required = True)
    parser.add_argument('--generate', type = bool, required = True)

    options = parser.parse_args()

    bop_dataset = BOPDataset(options.dataset_path, options.generate)
