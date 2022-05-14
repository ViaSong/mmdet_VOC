import os
import cv2
import json
json_dir='/root/spw/mmdetection/data/COCO/annotations'
train_path=os.path.join(json_dir,'train.json')
val_path=os.path.join(json_dir,'val.json')
# test_path=os.path.join(json_dir,'xml_test/')
img_path='/root/spw/mmdetection/data/VOCdevkit/VOC2007/JPEGImages/' #所有img的目录
train_dir = '/root/spw/mmdetection/data/COCO/train2017'
val_dir = '/root/spw/mmdetection/data/COCO/val2017'
#目标目录，包括三类img的文件夹
splitimg_path='/root/spw/mmdetection/data/COCO'

trainimg = json.load(open(train_path))['images']
# print(len(trainimg))
valimg = json.load(open(val_path))['images']
# for img in trainimg:
#     img_name = img['file_name']
#     load_img = cv2.imread(os.path.join(img_path,img_name))
#     cv2.imwrite(os.path.join(train_dir,img_name),load_img)
print(len(valimg))
for img in valimg:
    img_name = img['file_name']
    load_img = cv2.imread(os.path.join(img_path,img_name))
    cv2.imwrite(os.path.join(val_dir,img_name),load_img)