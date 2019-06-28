# -*- coding: utf-8 -*-
import os
lines = open(os.path.join('/media/dell/dell/data/遥感','ClsName2id.txt'),'r',encoding='utf-8').read().rstrip().split('\n')
#print(lines)#得到列表
catetory2idx = {}#列表转字典
for line in lines:
    #print(line)
    line_list = line.strip().split(':')
    #print(line_list)
    catetory2idx[line_list[0]] = int(line_list[2])
print(catetory2idx)
a=catetory2idx
