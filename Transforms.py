# @File  : Transforms.py
# @Author: dawnchen
# @Time: 2022/11/6 14:52
# -*- coding: utf-8 -*-
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

# transform 像一个工具箱
# python 的用法->tensor 类型
# 通过transform.ToTensor去解决两个问题
# 1.transform如何使用(python)
# 2.为什么要用Tensor数据类型
img_path="dataset/train/bees/95238259_98470c5b10.jpg"
img = Image.open(img_path) #通过路径打开文件
# 想要使用类必须先要实例化
tensor_trans=transforms.ToTensor()       #将类实例化
tensor_img= tensor_trans(img)
# 为什么需要Tensor数据类型
writer= SummaryWriter("logs")
writer.add_scalar("Tensore_img",tensor_img)
writer. close()




