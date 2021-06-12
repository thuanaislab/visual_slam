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
        number_batch = len(self.train_loader)
        for epoch in range(self.n_epochs):
            # SAVE
            if (epoch % self.configs.snapshot==0) and (epoch != 0):
                self.save_checkpoints(epoch)
            # Adjust lr 
            lr = self.optimizer.adjust_lr(epoch) # adjust learning rate
            # TRAIN
            self.model.train()
            self.optimizer.learner.zero_grad()
            train_loss = 0.0 
            count = 0
            pbar = enumerate(self.train_loader)
            pbar = tqdm(pbar, total=number_batch)
            for batch, (images, poses_gt) in pbar:
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
                loss.backward()
                self.optimizer.learner.step()
                self.optimizer.learner.zero_grad()
                train_loss += loss.item() * n_samples
                count += n_samples
            train_loss /= count 
            self.his_loss.append(train_loss)
            if epoch % self.configs.print_freq == 0:
                print("\nEpoch {} --- Lr {} --- Loss: {}\n".format(epoch, lr, train_loss))
        
        file_his = os.path.join(self.logdir, 'his_loss.txt')
        pd.DataFrame(self.his_loss).to_csv(file_his, header = False, index = False)
            
                        
                    
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
