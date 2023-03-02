"""
coding:utf-8
@Software:PyCharm
@Time:2022/11/14 20:45
@Author:ChenXi
-*- coding: utf-8 -*-
"""
import torch
import torchvision.transforms
from PIL import Image
from torch import nn
img= Image.open("imgs/jennie2.jpg")
img=img.convert("RGB")
image_pth=("imgs/jennie2.jpg")
image= Image.open(image_pth)#image is tensor
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                          torchvision.transforms.ToTensor()])
image= transform(image)
print(image)

class model_dawn(nn.Module):  # 构建网络模型
    def __init__(self):
        super(model_dawn, self).__init__()  # 继承父类函数
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),  # 展平层将数据变成C*1*1格式，最终将特征矩阵转化为特征向量
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model= torch.load('dawn9.pth')
print(model)

image= torch.reshape(image,(1,3,32,32))
print(image.shape)

model.eval()
with torch.no_grad():
    output= model(image)
print(output)
print(output.argmax(1))
