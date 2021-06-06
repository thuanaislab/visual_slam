#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 01:07:19 2021

@author: thuan
"""

import pandas as pd 
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 
import numpy as np


def show_features(img, features):
    plt.imshow(img)
    plt.scatter(features[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


class LoadFeatureData(Dataset):
    def __init__(self, pose_dir, img_folder, sift_folder):
        self.poses = pd.read_csv(pose_dir, header = None, sep = " ")
        self.img_folder = img_folder
        self.sift_folder = sift_folder
        
    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx.tolist()
        img_name = os.path.join(self.img_folder, self.poses.iloc[idx, 0])
        image = io.imread(img_name)

        return img_name

if __name__ == "__main__":
    pose_dir = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SIFT/poses.txt"
    img_folder = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SIFT/"
    sift_folder = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SIFT/orig_sift2txt/"
    
    
    poses = pd.read_csv(pose_dir, header = None, sep = " ")
    img_name = poses.iloc[0,0]
    features = pd.read_csv(sift_folder + img_name.replace(".jpg","") + ".txt", header = None, sep = " ")
    print("Image name {}".format(img_name))
    print("feature shape {}".format(features.shape))
    