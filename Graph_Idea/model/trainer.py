#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 18:40:01 2021

@author: thuan
"""
import sys
sys.path.insert(0, '../')
import torch.nn as nn 
from optimizer import Optimizer
import copy 
import os
from superpoint import SuperPoint


class Trainer(object):
    
    def __init__(self, model, optimizer_config, train_dataset, criterion, configs, superpoint_configs):
        self.model = model
        self.sp_model = SuperPoint(superpoint_configs.get('superpoint', {})).eval()
        self.criterion = criterion
        self.configs = configs
        self.n_epochs = self.configs.n_epochs
        self.optimizer = Optimizer(self.model.parameters(), method = self.configs.optimizer, 
                                   base_lr = self.configs.lr, weight_decay = self.configs.weight_decay, 
                                   **optimizer_config)
        self.logdir = self.configs.logdir
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
                           'optim_state_dict': optim_state_dict,
                           'criterion_state_dict': self.criterion.state_dict()}
        filename = os.path.join(self.logdir, 'epoch_{:03d}.pth.tar'.format(epoch))
        torch.save(checkpoint_dict, filename)
    
    def train(self):
        
        for epoch in range(self.n_epochs):
            
            if epoch % self.config.snapshot==0:
                self.save_checkpoints(epoch)
        
            lr = self.optimizer.adjust_lr(epoch) # adjust learning rate
            # TRAIN
            self.model.train()
            
            for batch, (images, poses_gt) in enumerate(self.train_loader):
                
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
