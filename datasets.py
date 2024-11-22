import torch
from torch.utils import data
from PIL import Image
import os
import shutil
import random
import numpy as np
import pandas as pd
import csv


class CustomDataset(data.Dataset):
    def __init__(self, dataset, indices, source_class = None, target_class = None):
        self.dataset = dataset
        self.indices = indices
        self.source_class = source_class
        self.target_class = target_class  
        self.contains_source_class = False
            
    def __getitem__(self, index):
        x, y = self.dataset[int(self.indices[index])][0], self.dataset[int(self.indices[index])][1]
        if y == self.source_class:
            y = self.target_class 
        return x, y 

    def __len__(self):
        return len(self.indices)

class PoisonedDataset(data.Dataset):
    def __init__(self, dataset, source_class = None, target_class = None):
        self.dataset = dataset
        self.source_class = source_class
        self.target_class = target_class  
            
    def __getitem__(self, index):
        x, y = self.dataset[index][0], self.dataset[index][1]
        if y == self.source_class:
            y = self.target_class 
        return x, y 

    def __len__(self):
        return len(self.dataset)


train_dir = os.path.join('./data/gtsrb/csv_train_data', 'train_data.txt')
test_dir = os.path.join('./data/gtsrb/csv_test_data', 'test_data.txt')

def default_loader(path):
    return Image.open(path).convert('RGB')


class GTSRBDataset(data.Dataset):
    def __init__(self, txt_dir, transform=None, loader=default_loader, selected_classes=None):
        self.imgs = []
        self.selected_classes = selected_classes
        self.transform = transform
        self.loader = loader

        with open(txt_dir, 'r') as fn:
            for f in fn:
                f = f.strip('\n')
                words = f.split(',')
                img_path, label = words[0], int(words[1])
                # 如果指定了selected_classes，则过滤并映射标签
                if selected_classes is None or label in selected_classes:
                    new_label = selected_classes.index(label) if selected_classes else label
                    self.imgs.append((img_path, new_label))
                    #print(f"Original label: {label}, Mapped label: {new_label}")# 将图像路径和新标签存入

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # 使用映射后的标签访问图像数据
        img_path, label = self.imgs[index]
        image = self.loader(img_path)
        if self.transform is not None:
            image = self.transform(image)
        #print("image path: "+ str(img_path) + " and label: " + str(label))
        return image, label

# class GTSRBDataset():
#     def __init__(self, txt_dir, transform=None, loader=default_loader):
#         imgs = []
#         with open(txt_dir, 'r') as fn:
#             for f in fn:
#                 f = f.strip('\n')
#                 words = f.split(',')
#                 imgs.append((words[0], int(words[1])))
#         self.loader = loader
#         self.imgs = imgs
#         self.transform = transform
#         self.txt_dir = txt_dir
#
#     def __len__(self):
#         return len(self.imgs)
#
#     def __getitem__(self, index):
#         images, label = self.imgs[index]
#         image = self.loader(images)
#         image = self.transform(image)
#         return image, label


# class IMDBDataset:
#     def __init__(self, reviews, targets):
#         """
#         Argument:
#         reviews: a numpy array
#         targets: a vector array
#
#         Return xtrain and ylabel in torch tensor datatype
#         """
#         self.reviews = reviews
#         self.target = targets
#
#     def __len__(self):
#         # return length of dataset
#         return len(self.reviews)
#
#     def __getitem__(self, index):
#         # given an index (item), return review and target of that index in torch tensor
#         x = torch.tensor(self.reviews[index,:], dtype = torch.long)
#         y = torch.tensor(self.target[index], dtype = torch.float)
#
#         return  x, y

# A method for combining datasets  
def combine_datasets(list_of_datasets):
    return data.ConcatDataset(list_of_datasets)
    