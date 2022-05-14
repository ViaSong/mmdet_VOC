# mmdet_VOC
this repository is designed for training Faster RCNN and YOLOv3 models on PASCAL-VOC dataset using mmdetection codebase.
## 1. prepare environment
please refer to https://github.com/open-mmlab/mmdetection

## 2. prepare dataset
download this repository

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
commands are similar to the previous section, remenber to use configs/yolo/yolov3_d53_mstrain-608_273e_coco.py as your config file, and modify offered scripts for visualization and inference if u need.

## 5. my model
you can download my pretrained model(PASCAL2007 dataset) from Baidu NetDisk https://pan.baidu.com/s/15K6iNJncWQVtIncFi7HMlg, access code is s7c7.

## 6. tips
given codes only support training and inference on PASCAL-VOC dataset, if u want to use your own dataset, remember to organize them in COCO format or just organize them in PASCAL-VOC format and switch to COCO format using given voc_to_coco.py script. and then please make sure that u change the dataset classes written in the config file to avoid error.

if you want to visualize the proposals given by the RPN module in faster rcnn, u can check ~/anaconda/envs/openmmlab/python3.7/site-packages/mmdet/detectors/two_stage.py and add the following code before the return line of function simple_test()
```
#######################################
        # for visualizing the proposals given by RPN
        import numpy as np
        import matplotlib.pyplot as plt
        bboxes = proposal_list[0].cpu().numpy()
        bboxes = bboxes[bboxes[:, 4] > 0.7][:50, :]
        img_meta = img_metas[0]
        bboxes[:, :4] = bboxes[:, :4] / img_meta['scale_factor'][0]
        img = mmcv.imread(img_meta['filename']).astype(np.uint8)
        img = img.copy()
        
        img = mmcv.bgr2rgb(img)
        width, height = img.shape[1], img.shape[0]
        img = np.ascontiguousarray(img)
        win_name = 'RPN_proposals'
        fig = plt.figure(win_name, frameon=False)
        plt.title(win_name)
        canvas = fig.canvas
        dpi = fig.get_dpi()
        # add a small EPS to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        EPS = 1e-2
        fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

        # remove white edges by set subplot margin
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = plt.gca()
        ax.axis('off')

        # max_label = int(max(labels) if len(labels) > 0 else 0)
        # text_palette = palette_val(get_palette(text_color, max_label + 1))
        # text_colors = [text_palette[label] for label in labels]
        from mmdet.core.visualization.palette import get_palette, palette_val
        from mmdet.core.visualization.image import draw_bboxes
        bbox_color = (72, 101, 241)
        thickness = 0.8
        num_bboxes = 0
        if bboxes is not None:
            num_bboxes = bboxes.shape[0]
            bbox_palette = palette_val(get_palette(bbox_color, 1))
            colors = bbox_palette*num_bboxes
            draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)

            horizontal_alignment = 'left'
            positions = bboxes[:, :2].astype(np.int32) + thickness
        plt.imshow(img)

        stream, _ = canvas.print_to_buffer()
        buffer = np.frombuffer(stream, dtype='uint8')
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        img = rgb.astype('uint8')
        img = mmcv.rgb2bgr(img)
        import os
        mmcv.imwrite(img, os.path.join('faster_rcnn/rpn_proposals', os.path.basename(img_meta['filename'])))
        ################################################
```
and then run inference.py to save the proposals of given image
