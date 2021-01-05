import mmcv
import numpy as np
from numpy import random

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.core.mask.utils import mask_expand, mask_crop


class MixUp(object):
    """暂未实现在loss上对label按mix加权"""
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


class CutOut(object):
    """CutOut operation.
    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.
    Args:
        n_holes (int | tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
        cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
            shape of dropped regions. It can be `tuple[int, int]` to use a
            fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
            shape from the list.
        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
            candidate ratio of dropped regions. It can be `tuple[float, float]`
            to use a fixed ratio or `list[tuple[float, float]]` to randomly
            choose ratio from the list. Please note that `cutout_shape`
            and `cutout_ratio` cannot be both given at the same time.
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
    """

    def __init__(self,
                 n_holes,
                 cutout_shape=None,
                 cutout_ratio=None,
                 fill_in=(0, 0, 0)):

        assert (cutout_shape is None) ^ (cutout_ratio is None), \
            'Either cutout_shape or cutout_ratio should be specified.'
        assert (isinstance(cutout_shape, (list, tuple))
                or isinstance(cutout_ratio, (list, tuple)))
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        self.n_holes = n_holes
        self.fill_in = fill_in
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape

    def __call__(self, img, bboxes, labels=None, masks=None):
        """Call function to drop some regions of image."""
        h, w, c = img.shape
        n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        if n_holes > len(bboxes):
            n_holes = len(bboxes)
        holes_idxs = random.choice(range(len(bboxes)), n_holes, replace=False)
        for idx in holes_idxs:
            bbox = bboxes[idx]
            bbox_x1, bbox_y1 = bbox[0], bbox[1]
            bbox_x2, bbox_y2 = bbox[2], bbox[3]
            bbox_w = bbox_x2 - bbox_x1 + 1
            bbox_h = bbox_y2 - bbox_y1 + 1
            if bbox_x1 >= bbox_x2 or bbox_y1 >= bbox_y2:
                continue
            x1 = np.random.randint(bbox_x1, bbox_x2)
            y1 = np.random.randint(bbox_y1, bbox_y2)
            cutout_w = random.uniform(self.candidates[0], self.candidates[1])
            cutout_h = random.uniform(self.candidates[0], self.candidates[1])
            if self.with_ratio:
                cutout_w = int(cutout_w * bbox_w)
                cutout_h = int(cutout_h * bbox_h)

            x2 = np.clip(x1 + cutout_w, 0, w)
            y2 = np.clip(y1 + cutout_h, 0, h)
            img[y1:y2, x1:x2, :] = self.fill_in

        return img, bboxes, labels, masks

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(n_holes={self.n_holes}, '
        repr_str += (f'cutout_ratio={self.candidates}, ' if self.with_ratio
                     else f'cutout_shape={self.candidates}, ')
        repr_str += f'fill_in={self.fill_in})'
        return repr_str


class MixCut(object):
    """暂未实现对label的mix"""
    def __init__(self,
                 n_holes,
                 cutout_shape=None,
                 cutout_ratio=None):
        assert (cutout_shape is None) ^ (cutout_ratio is None), \
            'Either cutout_shape or cutout_ratio should be specified.'
        assert (isinstance(cutout_shape, (list, tuple))
                or isinstance(cutout_ratio, (list, tuple)))
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        self.n_holes = n_holes
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        self.img2 = None

    def __call__(self, img1, boxes1, labels1, masks1=None):
        if self.img2 is not None:
            h, w, _ = img1.shape
            n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
            if n_holes > len(boxes1):
                n_holes = len(boxes1)
            holes_idxs = random.choice(range(len(boxes1)), n_holes, replace=False)
            for idx in holes_idxs:
                bbox = boxes1[idx]
                bbox_x1, bbox_y1 = bbox[0], bbox[1]
                bbox_x2, bbox_y2 = bbox[2], bbox[3]
                bbox_w = bbox_x2 - bbox_x1 + 1
                bbox_h = bbox_y2 - bbox_y1 + 1
                if bbox_x1 >= bbox_x2 or bbox_y1 >= bbox_y2:
                    continue
                x1 = np.random.randint(bbox_x1, bbox_x2)
                y1 = np.random.randint(bbox_y1, bbox_y2)
                cutout_w = random.uniform(self.candidates[0], self.candidates[1])
                cutout_h = random.uniform(self.candidates[0], self.candidates[1])
                if self.with_ratio:
                    cutout_w = int(cutout_w * bbox_w)
                    cutout_h = int(cutout_h * bbox_h)

                x2 = np.clip(x1 + cutout_w, 0, w)
                y2 = np.clip(y1 + cutout_h, 0, h)

                # cut from img2
                img2_h, img2_w = self.img2.shape[:2]
                cut2_w, cut2_h = x2 - x1, y2 - y1
                if cut2_w >= img2_w - cut2_w or cut2_h >= img2_h - cut2_h:
                    continue
                img2_x1 = np.random.randint(cut2_w, img2_w - cut2_w)
                img2_y1 = np.random.randint(cut2_h, img2_h - cut2_h)
                img2_cut = self.img2[img2_y1:img2_y1+cut2_h, img2_x1:img2_x1+cut2_w].copy()
                # update img2
                self.img2 = img1.copy()
                try:
                    img1[y1:y2, x1:x2, :] = img2_cut
                except Exception as e:
                    print(e)
        else:
            self.img2 = img1.copy()

        return img1, boxes1, labels1, masks1


class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 color_choose=0,
                 gray_p=0.3):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.color_choose = color_choose
        self.gray_p = gray_p

    def __call__(self, img, boxes, labels, masks=None):
        if self.color_choose == 0:
            if random.uniform() < self.gray_p:
                gray = mmcv.bgr2gray(img)
                img = mmcv.gray2bgr(gray)
                return img, boxes, labels, masks
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
                 mixup=None,
                 cutout=None,
                 mixcut=None):
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
        if cutout is not None:
            self.transforms.append(CutOut(**cutout))
        if mixcut is not None:
            self.transforms.append(MixCut(**mixcut))

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

