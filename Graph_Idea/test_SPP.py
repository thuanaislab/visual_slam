#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reimplementation of SSP net paper:
    https://arxiv.org/abs/1712.03452
Created on Mon Jun  7 01:07:19 2021

@author: thuan
"""

import pandas as pd 
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 
import numpy as np
import os 
import random 
import time 





def show_features(img, features):
    # plot the features on the image 
    plt.imshow(img)
    plt.scatter(features[:, 0], features[:, 1], s=5, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

def pd2listD(pandasFile):
    # convert dataFrame to list of dictionaries elements 
    # ex : [{'1': [1, 2]}, {'2': [3, 4]}] purpose is to accelerate data processing 
    # process
    out = []
    length = pandasFile.shape[0]
    for i in range(length):
        out.append({str(i):[pandasFile.iloc[i,0], pandasFile.iloc[i,1]]})
    return out
    
class LoadFeatureData(Dataset):
    def __init__(self, pose_dir, sift_folder, W, H, d1, d2):
        self.poses = pd.read_csv(pose_dir, header = None, sep = " ")
        self.sift_folder = sift_folder
        self.d1 = d1 
        self.d2 = d2
        self.stepw = W/self.d1
        self.steph = H/self.d2
    
    def getInPoints(self, rangew, rangeh, feature_list):
        list_index = []
        length = len(feature_list)
        for i in range(length):
            #print("length {} -- i {} ".format(length,i))
            tmp_x = list(feature_list[i].values())[0][0]
            temp_y = list(feature_list[i].values())[0][1]
            if (tmp_x > rangew[0]) and (tmp_x < rangew[1]):
                if (temp_y > rangeh[0]) and (temp_y < rangeh[1]):
                    list_index.append(i)


        if list_index != []:
            #print(list_index)
            for i in range(len(list_index)):
                feature_list.pop(i)
            return random.choice(list_index) , feature_list
        else:
            return None, feature_list
        
            
    def getFeatureTensor(self, feature_list):
        out_tensor = torch.zeros(133, self.d1, self.d2) 
        starth = 0.0
        list_Rindex = []
        for i in range(self.d2):
            startw = 0.0
            for ii in range(self.d1):
                index, feature_list = self.getInPoints([startw, startw + self.stepw], 
                                         [starth, starth + self.steph], feature_list)
                startw = startw + self.stepw
                if index != None:
                    list_Rindex.append(index)
            starth = starth + self.steph
        return list_Rindex

    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx.tolist()
        siftFile = pd.read_csv(sift_folder + self.poses.iloc[idx,0].replace(".jpg","") + ".txt", 
                               header = None, sep = " ")
        
        feature_list = pd2listD(siftFile)
        
        index = self.getFeatureTensor(feature_list)
        target = self.poses.iloc[idx, 1:8]
        target = np.array([target])
        target = torch.Tensor(target.astype(float))
        
        sample = {"target": target, "index":index}
        
        return sample



if __name__ == "__main__":
    pose_dir = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SIFT/poses.txt"
    img_folder = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SIFT/"
    sift_folder = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SIFT/orig_sift2txt/"
    
    
    poses = pd.read_csv(pose_dir, header = None, sep = " ")
    img_name = poses.iloc[1,0]
    features = pd.read_csv(sift_folder + img_name.replace(".jpg","") + ".txt", header = None, sep = " ")
    # print("Image name {}".format(img_name))
    # print("feature shape {}".format(features.shape))
    
    data = LoadFeatureData(pose_dir, sift_folder, 640, 480, 16, 16)
    start = time.time()
    print("number of points is: {}".format(len(data[1]["index"])))
    print("Total time is: {}".format(time.time()-start))
    # features = features.iloc[data[0]["index"],0:2]
    # features = np.asarray(features)
    # plt.figure()
    # show_features(io.imread(img_folder + img_name), features)
    # plt.plot()
    
    
    