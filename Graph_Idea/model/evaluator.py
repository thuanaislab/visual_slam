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
import copy 
import os
from .superpoint import SuperPoint
from tqdm import tqdm
import pandas as pd



class Evaluator(object):
    
    def __init__(self, model, dataset, criterion, configs):
        self.model = model
        self.criterion = criterion
        self.configs = configs
        self.logdir = self.configs.logdir
        self.load_epoch = configs.load_epoch
        self.checkpoint_file = os.path.join(self.logdir, 
                                            'epoch_{:03d}.pth.tar'.format(self.load_epoch))
        
        # data loader 
        
        # self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.configs.batch_size,
        #                                                num_workers=self.configs.num_workers)
        self.data_loader = dataset
        self.model = nn.DataParallel(self.model, device_ids=range(self.configs.GPUs))
        if self.configs.GPUs > 0:
            self.model.cuda()
            self.criterion.cuda()
        self.load_model()
    
    def adapt_load_state_dict(self, state_dict):
        new_state_dict = copy.deepcopy(self.model.state_dict())
        shape_conflicts = []
        missed = []
    
        for k, v in new_state_dict.items():
            if k in state_dict:
                if v.size() == state_dict[k].size():
                    new_state_dict[k] = state_dict[k]
                else:
                    shape_conflicts.append(k)
            else:
                missed.append(k)
    
        if(len(missed) > 0):
            print("Warning: The flowing parameters are missed in checkpoint: ")
            print(missed)
        if (len(shape_conflicts) > 0):
            print(
                "Warning: The flowing parameters are fail to be initialized due to the shape conflicts: ")
            print(shape_conflicts)
    
        self.model.load_state_dict(new_state_dict)
        
        
    
    def load_model(self):
        if os.path.isfile(self.checkpoint_file):
            loc_func = None if self.configs.GPUs > 0 else lambda storage, loc: storage
            print("load ", self.checkpoint_file)
            checkpoint = torch.load(self.checkpoint_file, map_location=loc_func)
            # load model 
            self.adapt_load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            # c_state = checkpoint['criterion_state_dict']
            # append_dict = {k: torch.Tensor([0.0])
            #                for k, _ in self.criterion.named_parameters()
            #                if not k in c_state}
            # c_state.update(append_dict)
            # self.criterion.load_state_dict(c_state)
        else:
            print("Error: Can not load the model at epoch {}".format(self.load_epoch))

    def evaler(self):
        number_batch = len(self.data_loader)
        self.model.eval()
        total_loss = 0.0 
        for i in range(number_batch):
            if self.configs.GPUs > 0:
                poses_gt = self.data_loader[i]['target'].cuda(non_blocking=True).view(1,7)
            else:
                poses_gt = self.data_loader[i]['target'].view(1,7)
            _inputs = self.data_loader[i]['features']
            predict = self.model(_inputs)
            print(self.data_loader[i]['names'])
            print(predict)
            print(poses_gt)
            loss = self.criterion(predict, poses_gt)
            total_loss += loss.item()
            print('loss {}'.format(loss.item()))
            break
        mean_total_loss = total_loss/number_batch
        # print("Recalculation Loss at epoch {}\n Loss: {}".format(self.load_epoch, mean_total_loss))

    