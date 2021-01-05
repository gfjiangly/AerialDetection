import torch.nn as nn
import torch.nn.functional as F
import cvtools 
import cv2.cv2 as cv
import numpy as np
import torch

from ..builder import build_loss


class SEBlock(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
                nn.Linear(in_features=channel, out_features=channel // ratio),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=channel // ratio, out_features=channel),
                nn.Sigmoid()
            )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class Inception(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(in_c, c1, kernel_size=1),
            nn.ReLU()
        )  
        self.p2 = nn.Sequential(
            nn.Conv2d(in_c, c2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=(1, 7), padding=1),
            nn.ReLU(),
            nn.Conv2d(c2[1], c2[2], kernel_size=(7, 1), padding=2),
            nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(in_c, c3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[0], kernel_size=(7, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=(1, 7), padding=2),
            nn.ReLU(),
            nn.Conv2d(c3[1], c3[1], kernel_size=(7, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(c3[1], c3[2], kernel_size=(1, 7), padding=2),
            nn.ReLU()
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_c, c4, kernel_size=1),
            nn.ReLU()
        )
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        return torch.cat((p1,p2,p3,p4), dim=1)


class InceptionAttention(nn.Module):
    def __init__(self):
        super(InceptionAttention, self).__init__()
        self.inception = Inception(
            in_c=256, c1=384, c2=(192,224,256), c3=(192,224,256), c4=128)
        self.conv = nn.Conv2d(1024, 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.inception(x)
        x = self.conv(x)
        return x


class MDANet(nn.Module):
    def __init__(self, 
                 channel, 
                 stages=(True, True, False, False, False),
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(MDANet, self).__init__()
        self.se_block = SEBlock(channel)
        self.inception_attention = InceptionAttention()
        self.stages = stages
        self.loss_mask = build_loss(loss_mask)
        self.pa_mask = []
    
    def init_weights(self):
        pass

    def forward(self, x):
        self.pa_mask = []
        outs = []
        for i, with_mda in enumerate(self.stages):
            if with_mda:
                pa_mask = self.inception_attention(x[i])
                pa_mask_softmax = F.softmax(pa_mask, dim=1)
                pa = pa_mask_softmax[:, [0], :, :]
                ca = self.se_block(x[i])
                out = pa.mul(x[i])
                out *= ca
                self.pa_mask.append(pa_mask)
                outs.append(out)
            else:
                outs.append(x[i])
        return outs

    def get_target(self, gt_masks, device):
        img_masks = []
        for i, masks in enumerate(gt_masks):
            # for j, mask in enumerate(masks):
            #     cv.imwrite('/code/AerialDetection/work_dirs/mask_{}_{}.jpg'.format(i, j), mask*255)
            new_masks = np.sum(masks, axis=0)
            img_masks.append(new_masks)
            # cv.imwrite('/code/AerialDetection/work_dirs/new_mask_{}.jpg'.format(i), new_masks*255)
        img_masks = np.stack(img_masks).astype(np.uint8)
        # img_masks = img_masks[:, :, :, np.newaxis]
        mask_targets = torch.from_numpy(img_masks).float().to(device)
        return mask_targets

    def loss(self, gt_masks, labels):
        mask_targets = self.get_target(
            gt_masks, self.pa_mask[0].device)
        mask_pred = []
        for out in self.pa_mask:
            mda_out = F.interpolate(
                out, size=mask_targets.shape[-2:], 
                mode='bilinear', align_corners = False)
            mask_pred.append(mda_out)
        mask_targets = torch.unsqueeze(mask_targets, dim=1)
        mask_targets = mask_targets.repeat((len(mask_pred), 2, 1, 1))
        mask_pred = torch.cat(mask_pred, dim=0)
        # mask_targets = torch.reshape(mask_targets, [-1, ])
        # mask_pred = torch.reshape(mask_pred, [-1, 2])
        loss_mask = F.binary_cross_entropy_with_logits(
            mask_pred, mask_targets, reduction='mean')[None]
        return dict(loss_mask=loss_mask)
