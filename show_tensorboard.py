from torch.utils.tensorboard import SummaryWriter
import json
import glob
import os

def main(base_dir):
    train_writer = SummaryWriter(os.path.join(base_dir,'log_file/train'))
    test_writer = SummaryWriter(os.path.join(base_dir,'log_file/test'))
    eval_writer = SummaryWriter(os.path.join(base_dir,'log_file/eval'))
    json_file = glob.glob(os.path.join(base_dir, '*.json'))[0]
    # faster RCNN
    # train = ['lr', 'loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'acc', 'loss_bbox', 'loss']
    # test = ['lr', 'loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'acc', 'loss_bbox', 'loss']
    # eval = ['bbox_mAP', 'bbox_mAP_50']
    # YOLOv3
    train = ["lr", "loss_cls", "loss_conf", "loss_xy", "loss_wh", "loss", "grad_norm"]
    test = ["lr", "loss_cls", "loss_conf", "loss_xy", "loss_wh", "loss"]
    eval = ["bbox_mAP", "bbox_mAP_50"]
    with open(json_file, 'r') as f:
        for j in f.readlines()[1:]:
            # 将josn字符串转化为dict字典
            j = json.loads(j)
            epoch = j['epoch']
            iter = j['iter']
            if j['mode'] == 'train':
                # step = (epoch - 1) * 2500 + iter # faster_rcnn
                step = (epoch - 1) * 600 + iter # yolov3
                for key in train:  
                    train_writer.add_scalar(key, j[key], global_step=step)
            else:
                if 'bbox_mAP' in j.keys():
                    step = (epoch - 1) * 4952 + iter
                    for key in eval:
                        eval_writer.add_scalar(key, j[key], global_step=step)
                else:
                    # step = (epoch - 1) * 2477 + iter # faster_rcnn
                    step = (epoch - 1) * 620 + iter # yolov3
                    for key in test:
                        test_writer.add_scalar(key, j[key], global_step=step)
    train_writer.close()
    test_writer.close()
    eval_writer.close()
                    
    # json_file = json.load(open(json_file))
    # import pdb;pdb.set_trace()

if __name__ == '__main__':
    # base_dir = './faster_rcnn'
    base_dir = './yolov3'
    main(base_dir)