import torch
import torch.nn as nn
import numpy as np
from mmdet.core import weighted_smoothl1
from mmdet.ops.piou_loss.pixel_weights import Pious
from mmdet.core import delta2dbbox_v3

from ..registry import LOSSES


def template_w_pixels(width):
    x = torch.tensor(torch.arange(-100, width + 100))
    grid_x = x.float() + 0.5
    return grid_x


@LOSSES.register_module
class PIoULoss(nn.Module):
    def __init__(self, k=10, size=1024, loss_weight=1.0):
        super(PIoULoss, self).__init__()
        self.template = template_w_pixels(size)
        self.PIoU = Pious(k, False)
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight, *args, **kwargs):
        # bboxes = delta2dbbox_v3(rois[:, 1:], bbox_pred, self.target_means,
        #     self.target_stds, img_meta['img_shape'])
        # pred[:, -1] = pred[:, -1] / 180.0 * np.pi
        # target[:, -1] = target[:, -1] / 180.0 * np.pi
        pious = self.loss_weight * self.PIoU(pred, target.data, 
            self.template.cuda(pred.get_device()))
        pious = torch.clamp(pious, 0.1, 1.0)
        pious = -2.0 * torch.log(pious)
        loss = torch.sum(pious)
        loss = loss / (pred.size(0) + 1e-9)
        return loss
