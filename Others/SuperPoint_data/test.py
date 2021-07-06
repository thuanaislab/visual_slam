#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 00:30:18 2021

@author: thuan
"""

import cv2
import time
import os
import pandas as pd 


image_path = "/home/thuan/Desktop/Public_dataset/Seven_Scenes/heads/vsfm_seq-02"
sift_path = "/home/thuan/Desktop/Public_dataset/Seven_Scenes/heads/vsfm_seq-02/sift"

for i in range(1000):
    data = pd.read_csv(os.path.join(sift_path, str(i)+".txt"), header=None, sep =" ")
    l = data.shape[0] - 1
    img = cv2.imread(os.path.join(image_path, str(i) + ".jpg"))
    for ii in range(1,l+1):
        img = cv2.circle(img, (data.iloc[ii,0],data.iloc[ii,1]), radius=1, color=(0, 0, 255), thickness=-1)
    
    
    cv2.imshow("ok", img)
    cv2.waitKey(0)



    