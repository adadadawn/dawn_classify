"""
coding:utf-8
@Software:PyCharm
@Time:2022/11/12 10:41
@Author:ChenXi
-*- coding: utf-8 -*-
"""
import torch
import torchvision.models
from torch import nn

vgg16_true= torchvision.models.vgg16(pretrained=True)
vgg16_false= torchvision.models.vgg16(pretrained=False)
vgg16_true.add_module('addlinear',nn.Linear(1000,10))
print(vgg16_true)
vgg16_false.classifier[6]=  nn.Linear(4096,1)
print(vgg16_false)


vgg16= torchvision.models.vgg16(pretrained= False)
# 方式1

torch.save(vgg16,"vgg16_1_path")
# 方式2
torch.save(vgg16.state_dict(),'vgg16_2_path')   #将参数保存为字典形式

from nn_Linear import *
linear=Linear_my()
torch.save(linear,"nn_linear_path")
model= torch.load("nn_linear_path")
print(model)
