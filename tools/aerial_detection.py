# -*- encoding:utf-8 -*-
# @Time    : 2021/1/3 15:15
# @Author  : gfjiang
import os.path as osp
import mmcv
import numpy as np
import cvtools
import matplotlib.pyplot as plt
import cv2.cv2 as cv
from functools import partial
import torch
import math
from cvtools.utils.path import add_prefix_filename_suffix

from mmdet.ops import nms
from mmdet.apis import init_detector, inference_detector


def draw_features(module, input, output, work_dir='./'):
    x = output.cpu().numpy()
    out_channels = list(output.shape)[1]
    height = int(math.sqrt(out_channels))
    width = height
    if list(output.shape)[2] < 128:
        return
    fig = plt.figure(figsize=(32, 32))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(height * width):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv.applyColorMap(img, cv.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        # print("{}/{}".format(i,width*height))
    savename = get_image_name_for_hook(module, work_dir)
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()


def get_image_name_for_hook(module, work_dir='./'):
    """
    Generate image filename for hook function

    Parameters:
    -----------
    module: module of neural network
    """
    # os.makedirs(work_dir, exist_ok=True)
    module_name = str(module)
    base_name = module_name.split('(')[0]
    index = 0
    image_name = '.'  # '.' is surely exist, to make first loop condition True
    while osp.exists(image_name):
        index += 1
        image_name = osp.join(
            work_dir, 'feats', '%s_%d.png' % (base_name, index))
    return image_name


class AerialDetectionOBB(object):

    def __init__(self, config, pth):
        self.imgs = []
        self.cfg = mmcv.Config.fromfile(config)
        self.pth = pth
        print('loading model {} ...'.format(pth))
        self.model = init_detector(self.cfg, self.pth, device='cuda:0')
        self.results = []
        self.img_detected = []
        # self.vis_feats((torch.nn.Conv2d, torch.nn.MaxPool2d))

    def __call__(self,
                 imgs_or_path,
                 det_thrs=0.5,
                 vis=False,
                 vis_thr=0.5,
                 save_root=''):
        if isinstance(imgs_or_path, str):
            self.imgs += cvtools.get_files_list(imgs_or_path)
        else:
            self.imgs += imgs_or_path
        prog_bar = mmcv.ProgressBar(len(self.imgs))
        for _, img in enumerate(self.imgs):
            self.detect(img, det_thrs=det_thrs, vis=vis,
                        vis_thr=vis_thr, save_root=save_root)
            prog_bar.update()

    def detect(self,
               img,
               det_thrs=0.5,
               vis=False,
               vis_thr=0.5,
               save_root=''):
        result = inference_detector(self.model, img)
        # result = self.nms(result)
        if isinstance(det_thrs, float):
            det_thrs = [det_thrs] * len(result)
        if vis:
            to_file = osp.join(save_root, osp.basename(img))
            to_file = add_prefix_filename_suffix(to_file, suffix='_obb')
            self.vis(img, result, vis_thr=vis_thr, to_file=to_file)
        result = [det[det[..., -1] > det_thr] for det, det_thr
                  in zip(result, det_thrs)]
        if len(result) == 0:
            print('detect: image {} has no object.'.format(img))
        self.img_detected.append(img)
        self.results.append(result)
        return result

    def nms(self, result, nms_th=0.3):
        dets_num = [len(det_cls) for det_cls in result]
        result = np.vstack(result)
        _, ids = nms(result, nms_th)
        total_num = 0
        nms_result = []
        for num in dets_num:
            ids_cls = ids[np.where((total_num <= ids) & (ids < num))[0]]
            nms_result.append(result[ids_cls])
            total_num += num
        return nms_result

    def vis(self, img, bbox_result, vis_thr=0.5,
            to_file='vis.jpg'):
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        inds = np.where(bboxes[:, -1] > vis_thr)[0]
        bboxes = bboxes[inds]
        labels = labels[inds]
        texts = [self.model.CLASSES[index]+'|'+str(round(bbox[-1], 2))
                 for index, bbox in zip(labels, bboxes)]
        img = cvtools.draw_boxes_texts(
            img, bboxes[:, :-1], texts, box_format='polygon')
        cvtools.imwrite(img, to_file)
    
    def vis_feats(self, modules_for_plot):
        h, w = self.cfg.data.train.img_scale
        for name, module in self.model.named_modules():
            if isinstance(module, modules_for_plot):
                draw_features_func = partial(
                    draw_features, work_dir=self.cfg.work_dir)
                module.register_forward_hook(draw_features_func)

    def save_results(self, save):
        str_results = ''
        for i, img in enumerate(self.img_detected):
            result = self.results[i]
            img = osp.basename(img)
            for cls_index, dets in enumerate(result):
                cls = self.model.CLASSES[cls_index]
                for box in dets:
                    bbox_str = ','.join(map(str, map(int, box[:4])))
                    str_results += ' '.join([img, cls, bbox_str]) + '\n'
        with open(save, 'w') as f:
            f.write(str_results)


if __name__ == '__main__':
    config_file = 'configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota_1gpus_mdanet_binary.py'
    pth_file = 'work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota_1gpus_mdanet_binary/epoch_12.pth'
    detector = AerialDetectionOBB(config_file, pth_file)
    detector('/media/data/DOTA/crop/P2701_2926_1597_3949_2620.png', vis=True, 
             save_root='work_dirs/attention_vis/')
    detector.save_results('work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota_1gpus_mdanet_binary/detect_result.txt')
