#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 00:21:06 2021

@author: thuan
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd 
from .utils import read_image
from .superpoint import SuperPoint

class CRDataset_test(Dataset):
    def __init__(self, poses_path:str, images_path:str, config: dict, device:str, resize = [-1]):
        self.df = pd.read_csv(poses_path, header = None, sep = " ")
        self.images_path = images_path
        self.config = config 
        self.device = device
        self.resize = resize 
        self.superpoint = SuperPoint(self.config).eval().to(device)
        
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        target = self.df.iloc[idx, 1:]
        target = np.array(target).astype(float)
        img_path = self.images_path + self.df.iloc[idx,0]
        _, img, _ = read_image(img_path, self.device, self.resize,0 ,False)
        features = self.superpoint({"image": img})
        # _,_,m,n = img.shape
        features["image"] = img
        # print(img.shape)
        target = torch.Tensor(target)
        for k in features:
             if isinstance(features[k], (list, tuple)):
                features[k] = torch.stack(features[k])
        sample = {'features': features, 'target': target, 'names': self.df.iloc[idx,0]}
        return sample 

class CRDataset_train(Dataset):
    def __init__(self, poses_path:str, images_path:str, device='cuda', resize = [-1]):
        self.df = pd.read_csv(poses_path, header = None, sep = " ")
        self.images_path = images_path
        self.device = 'cuda'
        self.resize = resize 
        
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        target = self.df.iloc[idx, 1:]
        target = np.array(target).astype(float)
        img_path = self.images_path + self.df.iloc[idx,0]
        _, img, _ = read_image(img_path, self.device, self.resize,0 ,False)
        
        target = torch.Tensor(target)
        _,_,m,n = img.shape
        img = img.view(1,m,n)
        
        return img, target, self.df.iloc[idx,0]