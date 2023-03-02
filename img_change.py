"""
coding:utf-8
@Software:PyCharm
@Time:2022/11/14 21:36
@Author:ChenXi
-*- coding: utf-8 -*-
"""

from PIL import Image

image = Image.open('imgs/jennie_cat.png')
# 将一个4通道转化为rgb三通道
image = image.convert("RGB")
print(image)

