# -*- coding: utf-8 -*-
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
            self.catetory2idx[line_list[0]] = int(line_list[2])

        # 读取文件名称
        self.file_names = []
        for root,dirs,names in os.walk(osp.join(rootpth,mode)):
            for name in names:
                self.file_names.append(osp.join(root,name))

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
    print(len(aaa))
    print(aaa.catetory2idx)
