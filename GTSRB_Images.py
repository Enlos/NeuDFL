import random
import glob
from PIL import Image
from random import shuffle
random.seed(7)
import numpy as np
from torchvision import datasets, transforms
import codecs
# import tensorflow as tf
import pandas as pd
from datasets import *
from torch.utils.data import DataLoader, Subset, Dataset
import os


train_dir_from_path = './data/gtsrb/Final_Training/Images'
train_dir_to_path = './data/gtsrb/Processed_Train/Images'
test_dir_from_path = './data/gtsrb/Final_Test/Images'
test_dir_to_path = './data/gtsrb/Processed_Test/Images'



def traintransform(dir_from_path, dir_to_path):
    if not os.path.exists(dir_to_path):
        os.mkdir(dir_to_path)  # 制作Images文件，mkdir默认制作地址的最后一个文件，makesdir默认制作整个路径文件
    if os.path.exists(dir_from_path):
        dir_from_children_names = os.listdir(
            dir_from_path)  # listdir列出此文件夹下面包含的所有子文件，这里Images下面是43个子文件夹00000,00001,00002...
    for dir_from_children_name in dir_from_children_names:  # 00000,00001
        dir_from_children_path = os.path.join(dir_from_path, dir_from_children_name)  # 这里主要是得到路径images/00000
        dir_to_children_path = os.path.join(dir_to_path, dir_from_children_name)  # images/00000
        if not os.path.exists(dir_to_children_path):
            os.mkdir(dir_to_children_path)
        for f in os.listdir(dir_from_children_path):  # 这里进入子文件夹的路径，比如00000下面的00000_00000.ppm，00000_00001.ppm
            if f.endswith('.csv'):  # 每个子文件夹下面有一个csv文件
                csv_dir = os.path.join(dir_from_children_path, f)
                csv_data = pd.read_csv(csv_dir)
                #csv_data_array=np.array(csv_data)
                filenames = [f for f in os.listdir(dir_from_children_path) if f.endswith('.ppm')]  # 00000.00000.ppm...
                for index in range(len(filenames)):
                    (shotname, suffix) = os.path.splitext(filenames[index])  # 分割 00000.00000, .ppm
                    if suffix == '.ppm':
                        file_from_path = os.path.join(dir_from_children_path, filenames[index])  # 得到image的path
                        # images/00000/00000_00000.ppm
                        file_to_path = os.path.join(dir_to_children_path, (shotname + '.png'))
                        # images/00000/00000_00000.png
                        img = Image.open(file_from_path)
                        csv_data_list = np.array(csv_data)[index, :].tolist()[0].split(';')
                        box = (int(csv_data_list[3]), int(csv_data_list[4]), int(csv_data_list[5]),
                               int(csv_data_list[6]))  # roi的四个坐标
                        roi_img = img.crop(box)
                        roi_img.save(file_to_path)


def testtransform(dir_from_path, dir_to_path):
    if not os.path.exists(dir_to_path):
        os.mkdir(dir_to_path)
    if os.path.exists(dir_from_path):
        filenames = os.listdir(dir_from_path)  # 00000.ppm,00001.ppm,00002.ppm
        for filename in filenames:
            if filename.endswith('.csv'):
                csv_dir = os.path.join(dir_from_path, filename)
                csv_data = pd.read_csv(csv_dir)
                csv_data_array = np.array(csv_data)
                dict = {}
                for index in range(csv_data_array.shape[0]):
                    row_data = np.array(csv_data)[index][0]
                    row_data_list = row_data.split(';')
                    sample_file_name = row_data_list[0]
                    sample_label = row_data_list[-1]  # 这里是label信息
                    #     print(sample_label)
                    # dir_file_path=os.path.join(dir_to_path,sample_label)
                    # 这里我将测试集也按label做了子文件夹并分别放置，后来发现没必要，在这里就删了
                    #  if not os.path.exists(dir_file_path):
                    #      os.mkdir(dir_file_path)
                    #  else:
                    #      pass
                    new_sample_file_name = sample_file_name.split('.')[0] + '.png'
                    dict[new_sample_file_name] = sample_label  # 00000.png:16
                for index in range(len(filenames)):
                    (shotname, suffix) = os.path.splitext(filenames[index])
                    if suffix == '.ppm':
                        file_from_path = os.path.join(dir_from_path, filenames[index])  # images/00000.ppm
                        img = Image.open(file_from_path)
                        csv_data_list = np.array(csv_data)[index, :].tolist()[0].split(';')
                        box = (
                        int(csv_data_list[3]), int(csv_data_list[4]), int(csv_data_list[5]), int(csv_data_list[6]))
                        roi_img = img.crop(box)
                        # path = os.path.join(dir_to_path, (dict[shotname + '.png']))
                        # file_to_path = os.path.join(path, (shotname + '.png'))
                        file_to_path = os.path.join(dir_to_path, (shotname + '.png'))
                        roi_img.save(file_to_path)

# 转换训练集和测试集图片
traintransform(train_dir_from_path, train_dir_to_path)
testtransform(test_dir_from_path, test_dir_to_path)