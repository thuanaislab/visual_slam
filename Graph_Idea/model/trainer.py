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
from .evaluator import get_errors, plot_result, qexp
import copy 
import os
from .superpoint import SuperPoint
from tqdm import tqdm
import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt
import copy




class Trainer(object):
    
    def __init__(self, model, optimizer_config, trainLoader, test_dataset, test_target,
                 criterion, configs, superpoint_configs):
        self.model = model
        self.total_params = sum(p.numel() for p in self.model.parameters())
        print("\nTotal parameters: {}".format(self.total_params))
        self.sp_model = SuperPoint(superpoint_configs).eval()
        self.criterion = criterion[0]
        self.configs = configs
        self.n_epochs = self.configs.n_epochs
        self.optimizer = Optimizer(self.model.parameters(), **optimizer_config)
        self.logdir = self.configs.logdir
        self.test_target = test_target
        self.his_loss = []
        self.best_model = None
        self.best_epoch = 0 
        self.best_loss = 1000
        self.best_prediction = None 
        self.best_mean_t = 10
        self.best_mean_q = 10
        # read mean and stdev for un-normalizing predictions
        pose_stats_file = os.path.join(self.configs.dir_stats, 'pose_stats.txt')
        self.pose_m, self.pose_s = np.loadtxt(pose_stats_file)  # mean and stdev
        # set random seed 
        torch.manual_seed(self.configs.seed)
        if self.configs.GPUs > 0: 
            torch.cuda.manual_seed(self.configs.seed)
        
        # data loader 
        
        self.train_loader = torch.utils.data.DataLoader(trainLoader[0], batch_size=self.configs.batch_size, 
                                                        shuffle=self.configs.shuffle, num_workers=self.configs.num_workers)
        if self.configs.mul_step:
            self.train_loader_1 = torch.utils.data.DataLoader(trainLoader[1], batch_size=self.configs.batch_size, 
                                                        shuffle=self.configs.shuffle, num_workers=self.configs.num_workers)
            self.criterion_1 = criterion[1]
        if configs.do_val:
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
                                                            shuffle=0, num_workers=self.configs.num_workers)
            self.L = len(test_dataset)
            print('number of test batch: ', len(self.test_loader))
        else:
            self.test_loader = None 
        print('number of train batch: ', len(self.train_loader))
        
        self.model = nn.DataParallel(self.model, device_ids=range(self.configs.GPUs))
        if self.configs.GPUs > 0:
            self.model.cuda()
            self.criterion.cuda()
            self.sp_model.cuda()
            if self.configs.mul_step:
                self.criterion_1.cuda()
        
    def save_checkpoints(self, epoch):
        optim_state = self.optimizer.learner.state_dict() 
        checkpoint_dict = {'epoch':epoch, 'model_state_dict': self.model.state_dict(), 
                           'optim_state_dict': optim_state,
                           'criterion_state_dict': self.criterion.state_dict()}
        filename = os.path.join(self.logdir, 'epoch_{:03d}.pth.tar'.format(epoch))
        torch.save(checkpoint_dict, filename)
    def save_best(self):
        checkpoint_dict = {'epoch':self.best_epoch, 'model_state_dict': self.best_model, "loss": self.best_loss}
        filename = os.path.join(self.logdir, 'best_result.pth.tar')
        torch.save(checkpoint_dict, filename)
        filename_p = os.path.join(self.logdir, 'best_prediction.txt')
        self.best_prediction.to_csv(filename_p, header=False, index = False, sep = " ")
    
    def train(self):
        save_first_target = True
        number_train_batch = len(self.train_loader)
        
        start_total_time = time.time()
        total_time = 0.0
        his_l1 = []
        his_l2 = [] 
        his_l3 = []
        for epoch in range(1,self.n_epochs+1):
            lr = self.optimizer.adjust_lr(epoch) # adjust learning rate
            # SAVE
            if (epoch % self.configs.snapshot==0):
                if self.configs.save_checkpoint:
                    self.save_checkpoints(epoch)
                plt.plot(self.his_loss, label='total loss')
                plt.show()
                if self.configs.version == 2:
                    plt.plot(his_l1,color='red')
                    plt.show()
                    plt.plot(his_l2,color='y')
                    plt.show()
                    plt.plot(his_l3,color='#008000')
                    plt.show()
                
            # TRAIN
            self.model.train()
            train_loss = 0.0
            t_l1 = 0.0
            t_l2 = 0.0
            t_l3 = 0.0
            count = 0
            pbar = enumerate(self.train_loader)
            pbar = tqdm(pbar, total=number_train_batch)
            
            start_time = time.time() # time at begining of each epoch 
            
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
                if self.configs.version == 1:
                        
                    predict = self.model(_inputs)
                    loss = self.criterion(predict, poses_gt)
                elif self.configs.version == 2:
                    predict, z_e, emb,_ = self.model(_inputs)

                    loss, l1, l2, l3 = self.criterion(predict, poses_gt, z_e, emb)
                    t_l1 += l1.item() * n_samples
                    t_l2 += l2.item() * n_samples
                    t_l3 += l3.item() * n_samples
                self.optimizer.learner.zero_grad()
                loss.backward()
                self.optimizer.learner.step()
                train_loss += loss.item() * n_samples
                count += n_samples

            total_batch_time = (time.time() - start_time)/60 # time at the end of each epoch 
            total_time += total_batch_time
            train_loss /= count
            if self.configs.version == 2:
                t_l1 /= count
                t_l2 /= count
                t_l3 /= count
                his_l1.append(t_l1)
                his_l2.append(t_l2)
                his_l3.append(t_l3)
            self.his_loss.append(train_loss)
            if epoch % self.configs.print_freq == 0:
                print("\nEpoch {} --- Loss: {} --- best_loss: {}\n".format(epoch, train_loss, self.best_loss))
                if self.configs.version == 2:
                    print("\l1 {} --- l2: {} --- l3: {}\n".format(t_l1, t_l2, t_l3))
                if self.configs.do_val:
                    print("\n meand error t {} --- meand error q: {}\n".format(self.best_mean_t, self.best_mean_q))
            
            # EVALUATION
            if self.configs.do_val:
                self.model.eval()
                test_loss = 0.0 
                count = 0
                pred_poses = np.zeros((self.L, 7))  # store all predicted poses
                targ_poses = np.zeros((self.L, 7))  # store all target poses
                pbar = enumerate(self.test_loader)
                number_test_batch = len(self.test_loader)
                pbar = tqdm(pbar, total=number_test_batch)
                imgs_list = []
                predict_ar = np.zeros((1,6))
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
                        if self.configs.version == 1:
                            predict = self.model(_inputs)
                            loss = self.criterion(predict, poses_gt)
                        elif self.configs.version == 2:
                            predict, z_e, emb,_ = self.model(_inputs)
                            loss, _, _, _ = self.criterion(predict, poses_gt, z_e, emb)
                        imgs_list = imgs_list + list(imgs)
                        predict_ar = np.concatenate((predict_ar, predict.cpu().detach().numpy()), axis = 0)
                        test_loss += loss.item() * n_samples
                        count += n_samples
                        #
                        s = predict.size()
                        output = predict.cpu().data.numpy().reshape((-1, s[-1]))
                        target = poses_gt.cpu().data.numpy().reshape((-1, s[-1]))
                    
                        # normalize the predicted quaternions
                        q = [qexp(p[3:]) for p in output]
                        output = np.hstack((output[:, :3], np.asarray(q)))
                        q = [qexp(p[3:]) for p in target]
                        target = np.hstack((target[:, :3], np.asarray(q)))
                    
                        # un-normalize the predicted and target translations
                        output[:, :3] = (output[:, :3] * self.pose_s) + self.pose_m
                        target[:, :3] = (target[:, :3] * self.pose_s) + self.pose_m
                    
                        # take the middle prediction
                        pred_poses[batch, :] = output[int(len(output) / 2)]
                        targ_poses[batch, :] = target[int(len(target) / 2)]
                        
                test_loss /= count
                predict_ar = np.delete(predict_ar, 0, 0)

                
                m,_ = predict_ar.shape
                assert m == len(imgs_list)
                name_col = np.zeros((m,1)) 
                predict_ar = np.concatenate((name_col, predict_ar), axis = 1)
                predict_ar = pd.DataFrame(predict_ar)
                predict_ar.iloc[:,0] = imgs_list
                _,_,meand_t, meand_q = get_errors(pred_poses, targ_poses, False)
                if (epoch % self.configs.snapshot==0):
                    if self.configs.save_checkpoint:
                        file1 = os.path.join(self.logdir, 'prediction_epoch_{:03d}.txt'.format(epoch))
                        file3 = os.path.join(self.logdir, 'loss_epoch_{:03d}.txt'.format(epoch))
                        print("\n Saving model and prediction result\n")
                        predict_ar.to_csv(file1, header=False, index = False, sep = " ")
                        pd.DataFrame([test_loss]).to_csv(file3, header=False, index = False, sep = " ")
                    if self.configs.scatter:
                        target = pd.read_csv(self.configs.test_poses_path, header = None, sep =" ")
                        target = target.iloc[:,1:].to_numpy()
                        predict_plot = predict_ar.iloc[:,1:].to_numpy()
                        plot_result(predict_plot, target, self.test_loader)
                # UPDATE best
                if self.best_loss > test_loss:
                    self.best_loss = test_loss
                    self.best_model = copy.deepcopy(self.model.state_dict())
                    self.best_epoch = epoch
                    self.best_prediction = predict_ar
                    self.best_mean_t = meand_t
                    self.best_mean_q = meand_q
        
        file_his = os.path.join(self.logdir, 'his_loss.txt')
        pd.DataFrame([self.his_loss]).to_csv(file_his, header = False, index = False)
        if self.configs.do_val:
            self.save_best()
        print("\nTraining Completed  --- Total training time: {}\n".format(total_time))
        print("Total time: {} minutes\n".format((time.time()-start_total_time)/60))
    def train_1(self):
        save_first_target = True
        number_train_batch = len(self.train_loader_1)
        
        start_total_time = time.time()
        total_time = 0.0
        his_l1 = []
        his_l2 = [] 
        his_l3 = []
        # first step 
        for epoch in range(1,self.configs.Nepoch_step1+1):
            lr = self.optimizer.adjust_lr(epoch) # adjust learning rate
            # SAVE
            if (epoch % self.configs.snapshot==0):
                if self.configs.save_checkpoint:
                    self.save_checkpoints(epoch)
                plt.plot(self.his_loss, label='total loss')
                plt.show()
                if self.configs.version == 2:
                    plt.plot(his_l1,color='red')
                    plt.show()
                    plt.plot(his_l2,color='y')
                    plt.show()
                    plt.plot(his_l3,color='#008000')
                    plt.show()
                
            # TRAIN
            self.model.train()
            train_loss = 0.0
            t_l1 = 0.0
            t_l2 = 0.0
            t_l3 = 0.0
            count = 0
            pbar = enumerate(self.train_loader_1)
            pbar = tqdm(pbar, total=number_train_batch)
            
            start_time = time.time() # time at begining of each epoch 
            
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
                if self.configs.version == 1:
                    predict = self.model(_inputs)
                    loss = self.criterion(predict, poses_gt)
                elif self.configs.version == 2:
                    predict, z_e, emb,_ = self.model(_inputs)
                    loss, l1, l2, l3 = self.criterion(predict, poses_gt, z_e, emb)
                    t_l1 += l1.item() * n_samples
                    t_l2 += l2.item() * n_samples
                    t_l3 += l3.item() * n_samples
                self.optimizer.learner.zero_grad()
                loss.backward()
                self.optimizer.learner.step()
                train_loss += loss.item() * n_samples
                count += n_samples

            total_batch_time = (time.time() - start_time)/60 # time at the end of each epoch 
            total_time += total_batch_time
            train_loss /= count
            if self.configs.version == 2:
                t_l1 /= count
                t_l2 /= count
                t_l3 /= count
                his_l1.append(t_l1)
                his_l2.append(t_l2)
                his_l3.append(t_l3)
            self.his_loss.append(train_loss)
            if epoch % self.configs.print_freq == 0:
                print("\nEpoch {} --- Loss: {} --- best_loss: {}\n".format(epoch, train_loss, self.best_loss))
                if self.configs.version == 2:
                    print("\l1 {} --- l2: {} --- l3: {}\n".format(t_l1, t_l2, t_l3))
                if self.configs.do_val:
                    print("\n mean error t {} --- mean error q: {}\n".format(self.best_mean_t, self.best_mean_q))
        # Second step 
        number_train_batch = len(self.train_loader)
        for epoch in range(epoch + 1,self.configs.Nepoch_step2+1):
            lr = self.optimizer.adjust_lr(epoch) # adjust learning rate
            # SAVE
            if (epoch % self.configs.snapshot==0):
                if self.configs.save_checkpoint:
                    self.save_checkpoints(epoch)
                plt.plot(self.his_loss, label='total loss')
                plt.show()
                
            # TRAIN
            self.model.train()
            train_loss = 0.0
            count = 0
            pbar = enumerate(self.train_loader)
            pbar = tqdm(pbar, total=number_train_batch)
            
            start_time = time.time() # time at begining of each epoch 
            
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
                predict, _, _, _ = self.model(_inputs)
                loss = self.criterion_1(predict, poses_gt)
                self.optimizer.learner.zero_grad()
                loss.backward()
                self.optimizer.learner.step()
                train_loss += loss.item() * n_samples
                count += n_samples

            total_batch_time = (time.time() - start_time)/60 # time at the end of each epoch 
            total_time += total_batch_time
            train_loss /= count
            self.his_loss.append(train_loss)
            if epoch % self.configs.print_freq == 0:
                print("\nEpoch {} --- Loss: {} --- best_loss: {}\n".format(epoch, train_loss, self.best_loss))
                if self.configs.do_val:
                    print("\n mean error t {} --- mean error q: {}\n".format(self.best_mean_t, self.best_mean_q))
            # EVALUATION
            if self.configs.do_val:
                self.model.eval()
                test_loss = 0.0 
                count = 0
                pred_poses = np.zeros((self.L, 7))  # store all predicted poses
                targ_poses = np.zeros((self.L, 7))  # store all target poses
                pbar = enumerate(self.test_loader)
                number_test_batch = len(self.test_loader)
                pbar = tqdm(pbar, total=number_test_batch)
                imgs_list = []
                predict_ar = np.zeros((1,6))
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
                        if self.configs.version == 1:
                            predict = self.model(_inputs)
                            loss = self.criterion(predict, poses_gt)
                        elif self.configs.version == 2:
                            predict, z_e, emb,_ = self.model(_inputs)
                            loss, _, _, _ = self.criterion(predict, poses_gt, z_e, emb)
                        imgs_list = imgs_list + list(imgs)
                        predict_ar = np.concatenate((predict_ar, predict.cpu().detach().numpy()), axis = 0)
                        test_loss += loss.item() * n_samples
                        count += n_samples
                        #
                        s = predict.size()
                        output = predict.cpu().data.numpy().reshape((-1, s[-1]))
                        target = poses_gt.cpu().data.numpy().reshape((-1, s[-1]))
                    
                        # normalize the predicted quaternions
                        q = [qexp(p[3:]) for p in output]
                        output = np.hstack((output[:, :3], np.asarray(q)))
                        q = [qexp(p[3:]) for p in target]
                        target = np.hstack((target[:, :3], np.asarray(q)))
                    
                        # un-normalize the predicted and target translations
                        output[:, :3] = (output[:, :3] * self.pose_s) + self.pose_m
                        target[:, :3] = (target[:, :3] * self.pose_s) + self.pose_m
                    
                        # take the middle prediction
                        pred_poses[batch, :] = output[int(len(output) / 2)]
                        targ_poses[batch, :] = target[int(len(target) / 2)]
                        
                test_loss /= count
                predict_ar = np.delete(predict_ar, 0, 0)

                
                m,_ = predict_ar.shape
                assert m == len(imgs_list)
                name_col = np.zeros((m,1)) 
                predict_ar = np.concatenate((name_col, predict_ar), axis = 1)
                predict_ar = pd.DataFrame(predict_ar)
                predict_ar.iloc[:,0] = imgs_list
                _,_,meand_t, meand_q = get_errors(pred_poses, targ_poses, False)
                if (epoch % self.configs.snapshot==0):
                    if self.configs.save_checkpoint:
                        file1 = os.path.join(self.logdir, 'prediction_epoch_{:03d}.txt'.format(epoch))
                        file3 = os.path.join(self.logdir, 'loss_epoch_{:03d}.txt'.format(epoch))
                        print("\n Saving model and prediction result\n")
                        predict_ar.to_csv(file1, header=False, index = False, sep = " ")
                        pd.DataFrame([test_loss]).to_csv(file3, header=False, index = False, sep = " ")
                    if self.configs.scatter:
                        target = pd.read_csv(self.configs.test_poses_path, header = None, sep =" ")
                        target = target.iloc[:,1:].to_numpy()
                        predict_plot = predict_ar.iloc[:,1:].to_numpy()
                        plot_result(predict_plot, target, self.test_loader)
                # UPDATE best
                if self.best_loss > test_loss:
                    self.best_loss = test_loss
                    self.best_model = copy.deepcopy(self.model.state_dict())
                    self.best_epoch = epoch
                    self.best_prediction = predict_ar
                    self.best_mean_t = meand_t
                    self.best_mean_q = meand_q
        
        file_his = os.path.join(self.logdir, 'his_loss.txt')
        pd.DataFrame([self.his_loss]).to_csv(file_his, header = False, index = False)
        if self.configs.do_val:
            self.save_best()
        print("\nTraining Completed  --- Total training time: {}\n".format(total_time))
        print("Total time: {} minutes\n".format((time.time()-start_total_time)/60))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
