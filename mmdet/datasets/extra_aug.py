import mmcv
import numpy as np
from numpy import random

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.core.mask.utils import mask_expand, mask_crop


class MixUp(object):
    def __init__(self, p=0.3, lambd=0.5):
        self.lambd = lambd
        self.p = p
        self.img2 = None
        self.boxes2 = None
        self.labels2 = None
        self.masks2 = None

    def __call__(self, img1, boxes1, labels1, masks1=None):
        if random.random() < self.p and self.img2 is not None \
            and img1.shape[1] == self.img2.shape[1]:
            height = max(img1.shape[0], self.img2.shape[0])
            width = max(img1.shape[1], self.img2.shape[1])

            mixup_image = np.zeros([height, width, 3], dtype='float32')
            mixup_image[:img1.shape[0], :img1.shape[1], :] = \
                img1.astype('float32') * self.lambd
            mixup_image[:self.img2.shape[0], :self.img2.shape[1], :] += \
                self.img2.astype('float32') * (1. - self.lambd)
            mixup_image = mixup_image.astype('uint8')

            mixup_boxes = np.vstack((boxes1, self.boxes2))
            mixup_labels = np.hstack((labels1, self.labels2))
            if masks1 is not None:
                mixup_masks = np.vstack((masks1, self.masks2))
        else:
            mixup_image = img1
            mixup_boxes = boxes1
            mixup_labels = labels1
            mixup_masks = masks1
        # 更新img2信息用于mix的样本
        self.img2 = img1
        self.boxes2 = boxes1
        self.labels2 = labels1
        self.masks2 = masks1

        return mixup_image, mixup_boxes, mixup_labels, mixup_masks


class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 color_choose=0):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.color_choose = color_choose

    def __call__(self, img, boxes, labels, masks=None):
        if self.color_choose == 0:
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
        else:
            if self.color_choose == 1:
                # random brightness
                if random.randint(2):
                    delta = random.uniform(-self.brightness_delta,
                                        self.brightness_delta)
                    img += delta
            elif self.color_choose == 2:
                # random contrast first
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha
            else:
                # convert color from BGR to HSV
                img = mmcv.bgr2hsv(img)

                if self.color_choose == 3:
                    # random saturation
                    if random.randint(2):
                        img[..., 1] *= random.uniform(self.saturation_lower,
                                                    self.saturation_upper)
                if self.color_choose == 4:
                    # random hue
                    if random.randint(2):
                        img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                        img[..., 0][img[..., 0] > 360] -= 360
                        img[..., 0][img[..., 0] < 0] += 360

                # convert color from HSV to BGR
                img = mmcv.hsv2bgr(img)

        return img, boxes, labels, masks


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels, masks):
        if random.randint(2):
            return img, boxes, labels, masks

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)
        if masks is not None:
            masks = mask_expand(masks, expand_img.shape[0], 
                                expand_img.shape[1], top, left)
        return img, boxes, labels, masks


class RandomCrop(object):

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, boxes, labels, masks):
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, boxes, labels, masks

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w),
                                  int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                if masks is not None:
                    masks = masks[mask]
                    masks = mask_crop(masks, patch)
                return img, boxes, labels, masks


class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None,
                 mixup=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))
        if mixup is not None:
            self.transforms.append(MixUp(**mixup))

    def __call__(self, img, boxes, labels, masks=None):
        img = img.astype(np.float32)
        for transform in self.transforms:
            if masks is not None:
                img, boxes, labels, masks = transform(img, boxes, labels, masks)
            else:
                img, boxes, labels = transform(img, boxes, labels)
        if masks is not None:
            return img, boxes, labels, masks
        else:
            return img, boxes, labels

