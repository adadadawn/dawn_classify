"""
coding:utf-8
@Software:PyCharm
@Time:2022/11/10 15:50
@Author:ChenXi
-*- coding: utf-8 -*-
"""
import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input= torch.tensor([[1,-5,5],
                     [5,6,3]])
dataset= torchvision.datasets.CIFAR10("./data", train= False, download=True, transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset, batch_size= 64)

class Dawn(nn.Module):
    def __init__(self):
        super(Dawn, self).__init__()
        self. sig=Sigmoid()
    def forward(self, input):
        output= self.sig(input)
        return output
dawn=Dawn()
writer= SummaryWriter("./logs_ReLU")
step=0
for data in dataloader:
    imgs,target= data

    output= dawn(imgs)

    step+=1
writer.close()