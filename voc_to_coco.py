"""Convert PASCAL VOC annotations to MSCOCO format and save to a json file.
The MSCOCO annotation has following structure:
{
    "images": [
        {
            "file_name": ,
            "height": ,
            "width": ,
            "id":
        },
        ...
    ],
    "type": "instances",
    "annotations": [
        {
            "segmentation": [],
            "area": ,
            "iscrowd": ,
            "image_id": ,
            "bbox": [],
            "category_id": ,
            "id": ,
            "ignore":
        },
        ...
    ],
    "categories": [
        {
            "supercategory": ,
            "id": ,
            "name":
        },
        ...
    ]
}
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os.path as osp
from collections import OrderedDict
import json

import xmltodict
import mmcv

logger = logging.getLogger(__name__)


class PASCALVOC2COCO(object):
    """Converters that convert PASCAL VOC annotations to MSCOCO format."""

    def __init__(self):
        self.cat2id = {
            'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4,
            'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8,
            'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12,
            'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16,
            'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20
        }

    def get_img_item(self, file_name, image_id, size):
        """Gets a image item."""
        image = OrderedDict()
        image['file_name'] = file_name
        image['height'] = int(size['height'])
        image['width'] = int(size['width'])
        image['id'] = image_id
        return image

    def get_ann_item(self, obj, image_id, ann_id):
        """Gets an annotation item."""
        x1 = int(obj['bndbox']['xmin']) - 1
        y1 = int(obj['bndbox']['ymin']) - 1
        w = int(obj['bndbox']['xmax']) - x1
        h = int(obj['bndbox']['ymax']) - y1

        annotation = OrderedDict()
        annotation['segmentation'] = [[x1, y1, x1, (y1 + h), (x1 + w), (y1 + h), (x1 + w), y1]]
        annotation['area'] = w * h
        annotation['iscrowd'] = 0
        annotation['image_id'] = image_id
        annotation['bbox'] = [x1, y1, w, h]
        annotation['category_id'] = self.cat2id[obj['name']]
        annotation['id'] = ann_id
        annotation['ignore'] = int(obj['difficult'])
        return annotation

    def get_cat_item(self, name, id):
        """Gets an category item."""
        category = OrderedDict()
        category['supercategory'] = 'none'
        category['id'] = id
        category['name'] = name
        return category

    def convert(self, devkit_path, year, split, save_file):
        """Converts PASCAL VOC annotations to MSCOCO format. """
        split_file = osp.join(devkit_path, 'VOC{}/ImageSets/Main/{}.txt'.format(year, split))
        ann_dir = osp.join(devkit_path, 'VOC{}/Annotations'.format(year))

        name_list = mmcv.list_from_file(split_file)

        images, annotations = [], []
        ann_id = 1
        for name in name_list:
            image_id = int(''.join(name.split('_'))) if '_' in name else int(name)

            xml_file = osp.join(ann_dir, name + '.xml')

            with open(xml_file, 'r') as f:
                ann_dict = xmltodict.parse(f.read(), force_list=('object',))

            # Add image item.
            image = self.get_img_item(name + '.jpg', image_id, ann_dict['annotation']['size'])
            images.append(image)

            if 'object' in ann_dict['annotation']:
                for obj in ann_dict['annotation']['object']:
                    # Add annotation item.
                    annotation = self.get_ann_item(obj, image_id, ann_id)
                    annotations.append(annotation)
                    ann_id += 1
            else:
                logger.warning('{} does not have any object'.format(name))

        categories = []
        for name, id in self.cat2id.items():
            # Add category item.
            category = self.get_cat_item(name, id)
            categories.append(category)

        ann = OrderedDict()
        ann['images'] = images
        ann['type'] = 'instances'
        ann['annotations'] = annotations
        ann['categories'] = categories

        logger.info('Saving annotations to {}'.format(save_file))
        with open(save_file, 'w') as f:
            json.dump(ann, f)


if __name__ == '__main__':
    converter = PASCALVOC2COCO()
    devkit_path = './data/VOCdevkit'
    year = '2012'
    split = 'trainval'
    save_file = './data/VOCdevkit/trainval2012.json'
    converter.convert(devkit_path, year, split, save_file)