"""
coding:utf-8
@Software:PyCharm
@Time:2022/11/10 14:33
@Author:ChenXi
-*- coding: utf-8 -*-
"""

import torch
from torch import nn
from torch.nn import MaxPool2d
input= torch. tensor([[1,2,0,3,1],
                      [0,1,2,1,3],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]],dtype=torch.float)
input= torch.reshape(input,(-1,1,5,5,))
print(input.shape)
class Dawn(nn.Module):
    def __init__(self):
        super(Dawn, self).__init__()
        self.maxpool=MaxPool2d(3, ceil_mode= True)
    def forward(self, input):
        output= self. maxpool(input)
        return output
dawm= Dawn()
output= dawm(input)

print(output)