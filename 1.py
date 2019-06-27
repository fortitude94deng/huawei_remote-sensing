#encoding:utf-8
import os
import pandas as pd
data=[]
names = os.listdir("/media/dell/dell/data/遥感/test")  #路径
for name in names:
    data.append([name])
print(data)
a=['id']
test=pd.DataFrame(columns=a,data=data)
#print(test)
test.to_csv("./1.csv",index=None)





















