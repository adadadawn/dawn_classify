"""
coding:utf-8
@Software:PyCharm
@Time:2022/11/11 15:13
@Author:ChenXi
-*- coding: utf-8 -*-
"""

import torch
from torch.nn import L1Loss, MSELoss

input= torch.tensor([1.,2.,3.,4.,5.])
target= torch.tensor([5.,4.,7.,8.,3.])
print(input.shape)
loss= L1Loss()
print(loss(input, target))

loss2= MSELoss()
print(loss2(input, target ))

