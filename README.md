# mmdet_VOC
this repository is designed for training Faster RCNN and YOLOv3 models on PASCAL-VOC dataset using mmdetection codebase.
## 1. prepare environment
please refer to https://github.com/open-mmlab/mmdetection

## 2. prepare dataset
download pascal-voc dataset and place it to mmdet_VOC/data
run the given scripts to switch the VOC data format to COCO data format
```
cd mmdet_VOC
# remember to modify path variables in the following python file
python voc_to_coco.py # this will create json files in coco format
python split.py # this will reorganize images to coco format
```
now we have a coco format voc dataset for further training and evaluation

## 3. train and evaluate faster rcnn
```
# logfiles will be saved to mmdet_VOC/faster_rcnn
# if needed, modify hyper parameters in the config file
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py --gpus 1 --work-dir faster_rcnn # this command only supports single GPU training
```
losses and evaluation results will be strored at mmdet_VOC/faster_rcnn/\*.log.json, the model will save latest checkpoint after every epoch and evaluate by then
you can use tensorboard to check the loss curves during the training process
```
# create logfile
python create_tensorboard.py # remenber to modify base_dir, train/eval/test, step
tensorboard --logdir=faster_rcnn/log_file/train
```
test the performance of the model
```
python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py faster_rcnn/latest.pth --work-dir faster_rcnn --eval bbox`
```
## 4. train and evaluate yolov3
commands are similar to the previous section, remenber to use configs/yolo/yolov3_d53_mstrain-608_273e_coco.py as your config file, and modify offered scripts for visualization and einference if u need.
