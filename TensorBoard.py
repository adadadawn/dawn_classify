# @File  : TensorBoard.py
# @Author: dawnchen
# @Time: 2022/11/5 13:54 
# -*- coding: utf-8 -*-
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writer=SummaryWriter("logs")
image_path= "dataset/train/ants/36439863_0bec9f554f.jpg"
img_PIL= Image.open(image_path)
img_array= np.array(img_PIL)
print(img_array.shape)
writer.add_scalar("test", img_array,2)
for i in range(100):
    writer.add_scalar("y=x",i,i)
writer.close()

