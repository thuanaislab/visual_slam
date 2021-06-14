#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 18:40:01 2021

@author: thuan
"""
import sys
sys.path.insert(0, '../')
import torch
import torch.nn as nn 
from .optimizer import Optimizer
import copy 
import os
from .superpoint import SuperPoint
from tqdm import tqdm
import pandas as pd
import numpy as np
import time 

class Trainer(object):
    
    def __init__(self, model, optimizer_config, train_dataset, criterion, configs, superpoint_configs):
        self.model = model
        self.sp_model = SuperPoint(superpoint_configs).eval()
        self.criterion = criterion
        self.configs = configs
        self.n_epochs = self.configs.n_epochs
        self.optimizer = Optimizer(self.model.parameters(), **optimizer_config)
        self.logdir = self.configs.logdir
        self.his_loss = []
        # set random seed 
        torch.manual_seed(self.configs.seed)
        if self.configs.GPUs > 0: 
            torch.cuda.manual_seed(self.configs.seed)
        
        # data loader 
        
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.configs.batch_size, 
                                                        shuffle=self.configs.shuffle, num_workers=self.configs.num_workers)
        self.model = nn.DataParallel(self.model, device_ids=range(self.configs.GPUs))
        if self.configs.GPUs > 0:
            self.model.cuda()
            self.criterion.cuda()
            self.sp_model.cuda()
        
    def save_checkpoints(self, epoch):
        optim_state = self.optimizer.learner.state_dict() 
        checkpoint_dict = {'epoch':epoch, 'model_state_dict': self.model.state_dict(), 
                           'optim_state_dict': optim_state,
                           'criterion_state_dict': self.criterion.state_dict()}
        filename = os.path.join(self.logdir, 'epoch_{:03d}.pth.tar'.format(epoch))
        torch.save(checkpoint_dict, filename)
    
    def train(self):
        save_first_target = True
        number_batch = len(self.train_loader)
        total_time = 0.0
        for epoch in range(1,self.n_epochs+1):
            lr = self.optimizer.adjust_lr(epoch) # adjust learning rate
            # SAVE
            if (epoch % self.configs.snapshot==0):
                self.save_checkpoints(epoch)
                if self.configs.save_pairs:
                    self.save_pairs(epoch)
            # TRAIN
            self.model.train()
            self.optimizer.learner.zero_grad()
            train_loss = 0.0 
            count = 0
            pbar = enumerate(self.train_loader)
            pbar = tqdm(pbar, total=number_batch)
            
            start_time = time.time() # time at begining of each epoch 
            
            if (epoch % self.configs.snapshot==0):
                imgs_list = []
                predict_ar = np.zeros((1,7))
            for batch, (images, poses_gt, imgs) in pbar:
                if self.configs.GPUs > 0:
                    images = images.cuda(non_blocking=True)
                    poses_gt = poses_gt.cuda(non_blocking=True)
                n_samples = images.shape[0]
                with torch.no_grad():
                    super_point_results = self.sp_model.forward_training({"image": images})
                    keypoints = torch.stack(super_point_results['keypoints'], 0)
                    descriptors = torch.stack(super_point_results['descriptors'], 0)
                    scores = torch.stack(super_point_results['scores'], 0)
                _inputs = {
                    "keypoints": keypoints,
                    "descriptors": descriptors,
                    "image": images,
                    "scores": scores}
                predict = self.model(_inputs)
                loss = self.criterion(predict, poses_gt)
                if (epoch % self.configs.snapshot==0):
                    imgs_list = imgs_list + list(imgs)
                    predict_ar = np.concatenate((predict_ar, predict.cpu().detach().numpy()), axis = 0)
                loss.backward()
                self.optimizer.learner.step()
                self.optimizer.learner.zero_grad()
                train_loss += loss.item() * n_samples
                count += n_samples
            total_batch_time = (time.time() - start_time)/60 # time at the end of each epoch 
            total_time += total_batch_time
            train_loss /= count
            if (epoch % self.configs.snapshot==0):
                predict_ar = np.delete(predict_ar, 0, 0)
                file1 = os.path.join(self.logdir, 'prediction_epoch_{:03d}.txt'.format(epoch))
                file3 = os.path.join(self.logdir, 'loss_epoch_{:03d}.txt'.format(epoch))
                m,_ = predict_ar.shape
                assert m == len(imgs_list)
                name_col = np.zeros((m,1)) 
                predict_ar = np.concatenate((name_col, predict_ar), axis = 1)
                predict_ar = pd.DataFrame(predict_ar)
                predict_ar.iloc[:,0] = imgs_list
                predict_ar.to_csv(file1, header=False, index = False, sep = " ")
                pd.DataFrame([train_loss]).to_csv(file3, header=False, index = False, sep = " ")
            self.his_loss.append(train_loss)
            if epoch % self.configs.print_freq == 0:
                print("\nEpoch {} --- Loss: {} --- comp_time: {}\n".format(epoch, train_loss,total_batch_time))
        
        file_his = os.path.join(self.logdir, 'his_loss.txt')
        pd.DataFrame([self.his_loss]).to_csv(file_his, header = False, index = False)
        print("\nTraining Completed  --- Total training time: {}\n".format(total_time))
    
    def save_pairs(self, epoch):
        print("\nSaving the prediction at epoch {}\n".format(epoch))
        imgs_list = []
        self.model.eval()
        pbar = enumerate(self.train_loader)
        predict_ar = np.zeros((1,7))
        for batch, (images, poses_gt, imgs) in pbar:
            imgs_list = imgs_list + list(imgs)
            if self.configs.GPUs > 0:
                images = images.cuda(non_blocking=True)
                poses_gt = poses_gt.cuda(non_blocking=True)
            n_samples = images.shape[0]
            with torch.no_grad():
                super_point_results = self.sp_model.forward_training({"image": images})
                keypoints = torch.stack(super_point_results['keypoints'], 0)
                descriptors = torch.stack(super_point_results['descriptors'], 0)
                scores = torch.stack(super_point_results['scores'], 0)
                _inputs = {
                    "keypoints": keypoints,
                    "descriptors": descriptors,
                    "image": images,
                    "scores": scores}
                predict = self.model(_inputs)
                predict=predict.cpu().detach().numpy()
                predict_ar = np.concatenate((predict_ar, predict), axis = 0)
        # ---- 
        filename = os.path.join(self.logdir, 'prediction_epoch_{:03d}.txt'.format(epoch))
        predict_ar = np.delete(predict_ar, 0, 0)
        m,_ = predict_ar.shape
        assert m == len(imgs_list)
        name_col = np.zeros((m,1)) 
        predict_ar = np.concatenate((name_col, predict_ar), axis = 1)
        predict_ar = pd.DataFrame(predict_ar)
        predict_ar.iloc[:,0] = imgs_list
        predict_ar.to_csv(filename, header=False, index = False)
    
                        
                    
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
