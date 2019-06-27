# -*- coding: utf-8 -*-
'''
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import random

class RSDataset(Dataset):
    def __init__(self, rootpth='/media/dell/dell/data/遥感',des_size=(224,224),mode='train'):

        self.des_size = des_size
        self.mode = mode

        # 处理对应标签
        assert (mode=='train' or mode=='val' or mode=='test')
        lines = open(osp.join(rootpth,'ClsName2id.txt'),'r',encoding='utf-8').read().rstrip().split('\n')
        self.catetory2idx = {}
        for line in lines:
            print(line)
            line_list = line.strip().split(':')
            self.catetory2idx[line_list[0]] = int(line_list[2])#-1

        # 读取文件名称
        self.file_names = []
        for root,dirs,names in os.walk(osp.join(rootpth,mode)):
            for name in names:
                self.file_names.append(osp.join(root,name))

        # 随机选择一小部分数据做测试
        # self.file_names = random.choices(self.file_names[:200],k=5000)

        # 确定分隔符号
        self.split_char = '/' if '/' in self.file_names[0] else '/'

        # totensor 转换
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx):
        name = self.file_names[idx]
        category = name.split(self.split_char)[-2]
        cate_int = self.catetory2idx[category]
        img = Image.open(name)
        img = img.resize(self.des_size,Image.BILINEAR)
        return self.to_tensor(img),cate_int

    def __len__(self):
        return len(self.file_names)

if __name__ == '__main__':
    aaa = RSDataset(rootpth='/media/dell/dell/data/遥感',mode='test')
    # img,cat = aaa.__getitem__(13000)
    # print(cat)
    # print(img.size())
    # print(img)
    print(len(aaa))
    print(aaa.catetory2idx)
#    print(aaa.__getitem__(2))

'''
b={'停机坪': 0, '停车场': 1, '公园': 2, '公路': 3, '冰岛': 4, '商业区': 5, '墓地': 6, '太阳能发电厂': 7, '居民区': 8, '山地': 9, '岛屿': 10, '工厂': 11, '教堂': 12, '旱地': 13, '机场跑道': 14, '林地': 15, '桥梁': 16, '梯田': 17, '棒球场': 18, '水田': 19, '沙漠': 20, '河流': 21, '油田': 22, '油罐区': 23, '海滩': 24, '温室': 25, '港口': 26, '游泳池': 27, '湖泊': 28, '火车站': 29, '直升机场': 30, '石质地': 31, '矿区': 32, '稀疏灌木地': 33, '立交桥': 34, '篮球场': 35, '网球场': 36, '草地': 37, '裸地': 38, '足球场': 39, '路边停车区': 40, '转盘': 41, '铁路': 42, '风力发电站': 43, '高尔夫球场': 44}
a={'旱地': 1, '水田': 2, '梯田': 3, '草地': 4, '林地': 5, '商业区': 6, '油田': 7, '油罐区': 8, '工厂': 9, '矿区': 10, '太阳能发电厂': 11, '风力发电站': 12, '公园': 13, '游泳池': 14, '教堂': 15, '墓地': 16, '棒球场': 17, '篮球场': 18, '高尔夫球场': 19, '足球场': 20, '温室': 21, '网球场': 22, '居民区': 23, '岛屿': 24, '河流': 25, '停机坪': 26, '直升机场': 27, '机场跑道': 28, '桥梁': 29, '停车场': 30, '公路': 31, '路边停车区': 32, '转盘': 33, '立交桥': 34, '港口': 35, '铁路': 36, '火车站': 37, '裸地': 38, '沙漠': 39, '冰岛': 40, '山地': 41, '石质地': 42, '稀疏灌木地': 43, '海滩': 44, '湖泊': 45}
c={}
for i,k in b.items():
    c[k]=i
#print(c)
for i in c:
    for k in a:
        if c[i] == k:
            c[i] = a[k]
            break
print(c)
x=0
print(c[x])		
'''
b=image_datasets['train'].class_to_idx
c={}
for i,k in b.items():
    c[k]=i
a={'旱地': 1, '水田': 2, '梯田': 3, '草地': 4, '林地': 5, '商业区': 6, '油田': 7, '油罐区': 8, '工厂': 9, '矿区': 10, '太阳能发电厂': 11, '风力发电站': 12, '公园': 13, '游泳池': 14, '教堂': 15, '墓地': 16, '棒球场': 17, '篮球场': 18, '高尔夫球场': 19, '足球场': 20, '温室': 21, '网球场': 22, '居民区': 23, '岛屿': 24, '河流': 25, '停机坪': 26, '直升机场': 27, '机场跑道': 28, '桥梁': 29, '停车场': 30, '公路': 31, '路边停车区': 32, '转盘': 33, '立交桥': 34, '港口': 35, '铁路': 36, '火车站': 37, '裸地': 38, '沙漠': 39, '冰岛': 40, '山地': 41, '石质地': 42, '稀疏灌木地': 43, '海滩': 44, '湖泊': 45}
for i in c:
    for k in a:
        if c[i] == k:
            c[i] = a[k]
            break
'''



































