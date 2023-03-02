# @File  : test.py
# @Author: dawnchen
# @Time: 2022/11/6 16:32 
# -*- coding: utf-8 -*-

class Person:
    def __call__(self,name):
        print('__call__'+'Hello'+'\t'+name)

    def hello (self,name):
        print('hello'+name)

person= Person()
person("hhh")
person.hello('list')
for i in range(10):
    print(i)