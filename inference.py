from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os

# config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
config_file = 'configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = 'faster_rcnn/latest.pth'
checkpoint_file = 'yolov3/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:1')
imgs = ['data/VOCdevkit/VOC2012/JPEGImages/2012_000184.jpg','data/VOCdevkit/VOC2012/JPEGImages/2012_003502.jpg','data/VOCdevkit/VOC2012/JPEGImages/2012_003965.jpg']
# imgs = ['data/VOCdevkit/VOC2012/JPEGImages/2012_000870.jpg','data/VOCdevkit/VOC2012/JPEGImages/2012_000871.jpg','data/VOCdevkit/VOC2012/JPEGImages/2012_000880.jpg', 'data/VOCdevkit/VOC2012/JPEGImages/2012_000894.jpg']
for img in imgs:
    result = inference_detector(model, img)
    # show result
    # show_result_pyplot(model, img, result, out_file=os.path.join('faster_rcnn/inference', os.path.basename(img)))
    show_result_pyplot(model, img, result, out_file=os.path.join('yolov3/inference', os.path.basename(img)))