
# coding: utf-8
import torch.nn.functional as F
import torch
from torch import nn

# shape: [1, 2, 2, 2], 第一个1是batch，第一个2是类别数，后面2x2是特征图大小
pred_output = torch.tensor([[[[0.12,0.36],[0.22,0.66]],[[0.13,0.34],[0.52,-0.96]]]])
# shape: [1, 2, 2]
target = torch.tensor([[[1,0],[0,1]]])


temp1 = F.softmax(pred_output, dim=1)
temp3 = torch.log(temp1)
target = target.long()
loss1 = F.nll_loss(temp3, target)
print('loss1: ', loss1)


loss2 = nn.CrossEntropyLoss()
result2 = loss2(pred_output, target)
print('loss2: ', result2)


# 错误使用，BCEWithLogitsLoss的两个输入shape必须完全一样
loss3 = nn.BCEWithLogitsLoss()
result3 = loss3(pred_output, target)
print('loss3: ', result3)