"""
coding:utf-8
@Software:PyCharm
@Time:2022/11/14 23:17
@Author:ChenXi
-*- coding: utf-8 -*-
"""
"""
coding:utf-8
@Software:PyCharm
@Time:2022/11/14 20:45
@Author:ChenXi
-*- coding: utf-8 -*-
"""
import torch.optim

import torchvision.datasets
from torch import nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./train_model')  # 为后文的可视化做准备
train_data = torchvision.datasets.CIFAR10('./data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)  # 将训练集进行初始化

train_data_size = len(train_data)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("训练集长度为:{}".format(train_data_size))

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)


# ------------------------------------------------------------------------------------
# 构建神经网络
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


model_dawn_1 = model_dawn()  # 将model_dawn函数实例化
model_dawn_1 = model_dawn_1.to(device)
loss_fun = nn.CrossEntropyLoss()  # 将损失函数实例化
loss_fun = loss_fun.to(device)
learning_rate = 1e-2
optimizer = torch.optim.SGD(model_dawn_1.parameters(), lr=learning_rate)  # 优化器实例化
# ------------------------------------------------------------------------------------
# 训练模块
loss_add = 0
step = 0
for times in range(10):
    print('---------------------这是第{}次训练-------------------------'.format(times))
    # 训练步骤开始
    ##
    # 1:优化器梯度清零
    # 2：损失函数计算output和target的loss
    # 3.损失函数反向传播
    # 4：更新权重##
    # ------------------------训练模块---------------------------
    model_dawn_1.train()
    for data in train_dataloader:
        imgs, target = data
        imgs = imgs.to(device)
        target = target.to(device)
        output = model_dawn_1(imgs)
        loss = loss_fun(output, target)  # 计算梯度
        # 优化器优化模型⬇︎
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        # 优化器优化模型⬆︎
        loss_add += loss

        step += 1
        if step % 1000 == 0:
            print('第{}次～～～～损失函数：{},,,损失函数加和:{}'.format(step, loss, loss_add.item()))
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('loss_add', loss_add, step)
        # test
    # ------------------测试模块------------------------
    model_dawn_1.eval()
    test_loss = 0
    test_step = 0
    accurancy_add = 0
    test_dataloader = DataLoader(train_data, batch_size=64)
    with torch.no_grad():  # 以下参数不会自动求导
        for data in train_dataloader:
            imgs, target = data

            imgs = imgs.to(device)
            target = target.to(device)
            output = model_dawn_1(imgs)
            loss = loss_fun(output, target)
            test_loss += loss.item()
            accurancy = (output.argmax(1) == target).sum()  # 测试就是为了计算正确率
            accurancy_add += accurancy
            step += 1

    acc_len = len(test_dataloader)
    correct_rate = accurancy_add / acc_len
    test_step += 1

    torch.save(model_dawn_1, "train.pth")
    print('整体测试损失值：{},,,整体正确率:{}'.format(test_loss, correct_rate))
writer.close()


