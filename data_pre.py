import numpy as np
from functools import partial
import pandas as pd
import os
from tqdm import tqdm_notebook, tnrange, tqdm
import sys

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
#from config import config
from collections import OrderedDict
from torchvision import transforms as T
from imgaug import augmenters as iaa
import random
import pathlib
import cv2
# create dataset class
class create_test(Dataset):
    def __init__(self,images_df, base_path,augument=True,mode="train"):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)
        self.images_df = images_df.copy() #csv
        self.augument = augument
        self.images_df.Id = self.images_df.Id.apply(lambda x:base_path / str(x))#.zfill(6))
        self.mode = mode

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self,index):
        X = self.read_images(index)
        if not self.mode == "test":
            y = self.images_df.iloc[index].Target
        else:
            y = str(self.images_df.iloc[index].Id.absolute())
        if self.augument:
            X = self.augumentor(X)
        X = T.Compose([T.ToPILImage(),T.ToTensor()])(X)
        return X.float(),y


    def read_images(self,index):
        row = self.images_df.iloc[index]
        filename = str(row.Id.absolute())
        print(filename)
        images = cv2.imread(filename)#+'.jpg')
        return images

    def augumentor(self,image):
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.SomeOf((0,4),[
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
            ]),
            iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
            #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
            ], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug

test_files = pd.read_csv("/home/dell/Desktop/1.csv")
#train_gen = MultiModalDataset(train_data_list,config.train_data,config.train_vis,mode="train")
test_gen = create_test(test_files,'/media/dell/dell/data/遥感/test/',augument=False,mode="test")
#test_loader = DataLoader(test_gen,1,shuffle=False,pin_memory=True,num_workers=16)
x,y=test_gen[1]
print(x)
print(y)






















