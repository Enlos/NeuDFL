from PIL import Image
import os
import shutil
import random
import numpy as np
import pandas as pd
import csv


#这里我其实生成的是两个txt，不过其实都一样
def makeTrainCSV(dir_root_path,dir_to_path):
    if not os.path.exists(dir_to_path):
        os.makedirs(dir_to_path)
    dir_root_children_names=os.listdir(dir_root_path)
 #   print(dir_root_children_names)
    dict_all_class={}
    #每一个类别的dict：{path,label}
    csv_file_dir=os.path.join(dir_to_path,('train_data'+'.txt'))
    with open(csv_file_dir,'w',newline='') as csvfile:
        for dir_root_children_name in dir_root_children_names:
            dir_root_children_path = os.path.join(dir_root_path, dir_root_children_name)
            if os.path.isfile(dir_root_children_path):
                break
            file_names=os.listdir(dir_root_children_path)
            for file_name in file_names:
                (shot_name,suffix)=os.path.splitext(file_name)
                if suffix=='.png':
                    file_path=os.path.join(dir_root_children_path,file_name)
                    dict_all_class[file_path]=int(dir_root_children_name)
        list_train_all_class=list(dict_all_class.keys())
        random.shuffle(list_train_all_class)
        for path_train_path in list_train_all_class:
            label=dict_all_class[path_train_path]
            example=[]
            example.append(path_train_path)
            example.append(label)
            writer=csv.writer(csvfile)
            writer.writerow(example)
    print('训练集生成的csv文件完毕')
    print('list_train_all_class len:'+str(len(list_train_all_class)))

def makeTestCSV(dir_root_path,dir_to_path):
    if not os.path.exists(dir_to_path):
        os.makedirs(dir_to_path)
    file_names=os.listdir(dir_root_path)
    for file_name in file_names:
        (shot_name,suffix)=os.path.splitext(file_name)
        if suffix=='.csv':
            csv_file_path=os.path.join(dir_root_path,file_name)
            test_csv_data=pd.read_csv(csv_file_path)
            test_csv_data_arr=np.array(test_csv_data)
            dict={}
            for index in range(test_csv_data_arr.shape[0]):
                row_data=np.array(test_csv_data)[index][0]
                row_data_list=row_data.split(';')
                sample_file_name=row_data_list[0]
                sample_label=row_data_list[-1]
                new_sample_file_name=sample_file_name.split('.')[0]+'.png'
                dict[new_sample_file_name]=sample_label
    dict_all_class={}
    csv_file_dir=os.path.join(dir_to_path,('test_data')+'.txt')
    with open(csv_file_dir,'w',newline='') as csvfile:
        for file_name in file_names:
            (shot_name,suffix)=os.path.splitext(file_name)
            if suffix=='.png':
                file_path=os.path.join(dir_root_path,file_name)
                file_name=file_path.split('/')[-1]
                dict_all_class[file_path]=dict[file_name]
        list_test_all_class=list(dict_all_class.keys())
        random.shuffle(list_test_all_class)
        for path_test_path in list_test_all_class:
            label=dict_all_class[path_test_path]
            example=[]
            example.append(path_test_path)
            example.append(label)
            writer=csv.writer(csvfile)
            writer.writerow(example)
    print('测试集生成的csv文件完毕')
    print('list_test_all_class len:'+str(len(list_test_all_class)))

train_dir_root_path = "./data/gtsrb/Processed_Train/Images"  # 替换为你自己的训练集路径
train_dir_to_path = "./data/gtsrb/csv_train_data"  # 替换为生成的CSV文件路径

test_dir_root_path = "./data/gtsrb/Processed_Test/Images"  # 替换为你自己的测试集路径
test_dir_to_path = "./data/gtsrb/csv_test_data"  # 替换为生成的CSV文件路径
makeTrainCSV(train_dir_root_path,train_dir_to_path)
makeTestCSV(test_dir_root_path,test_dir_to_path)
