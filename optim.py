"""
coding:utf-8
@Software:PyCharm
@Time:2022/11/11 21:01
@Author:ChenXi
-*- coding: utf-8 -*-
"""
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset= torchvision.datasets.CIFAR10("./data", train= False,transform= torchvision.transforms.ToTensor(), download=True)   #transform是变化模式，train=False表示加载测试集(数据量少)
dataloader= DataLoader(dataset,batch_size= 6)   #每组里有多少张图片

class Con(nn.Module):
    def __init__(self):
        super(Con, self).__init__()     #nn.Module是父类的方法，super继承父类
        self.conv1= Conv2d(in_channels=3, out_channels=6,kernel_size=4)    #设置卷积核in_channels是看输入的层数(图片通道数)， out_channels是输出的通道数(卷积核的个数),kernel_size是卷积核尺寸为4*4
    def forward(self,x):
        x= self.conv1(x)
        return x

con1= Con()     #将类实例化

writer= SummaryWriter("./test_4")   #显示图片
step= 0
loss= nn.CrossEntropyLoss()     #定义损失函数
optim= torch.optim.SGD(con1.parameters(),lr=0.001)      #定义优化器
for data in dataloader:
    imgs, targets= data     #定义两个变量
    writer.add_images("input", imgs,step)   #将input显示
    output=con1(imgs)   #将output卷积
    output= torch.reshape(output,(-1,3,29,29))  #由于输出的是图片所以NCHW的C（通道数）必须是3
    optim.zero_grad()   #将梯度置0
    targets = torch.reshape(output, (-1, 3, 29, 29))    #targets和output的格式要相同
    result_loss=loss(output, targets)   #计算损失函数
    result_loss.backward()  #损失函数反向传播
    optim.step()    #更改梯度
    writer.add_image("output", output,step,dataformats="NCHW")  #让writer理解四维张量
    print(result_loss)
    step+=1
writer.close()


