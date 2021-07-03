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
import torch.nn.functional as F 
import copy 
import os
from .superpoint import SuperPoint
from .utils import quaternion_angular_error
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


def plot_result(pred_poses, targ_poses, data_set):
    # this function is based on https://github.com/NVlabs/geomapnet
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    
    # plot on the figure object
    ss = max(1, int(len(data_set) / 1000))  # 100 for stairs
    # scatter the points and draw connecting line
    x = np.vstack((pred_poses[::ss, 0].T, targ_poses[::ss, 0].T))
    y = np.vstack((pred_poses[::ss, 1].T, targ_poses[::ss, 1].T))
    z = np.vstack((pred_poses[::ss, 2].T, targ_poses[::ss, 2].T))
    for xx, yy, zz in zip(x.T, y.T, z.T):
      ax.plot(xx, yy, zs=zz, c='b')
    ax.scatter(x[0, :], y[0, :], zs=z[0, :], c='r', depthshade=0)
    ax.scatter(x[1, :], y[1, :], zs=z[1, :], c='g', depthshade=0)
    ax.view_init(azim=119, elev=13)
    plt.show()

def get_errors(target, predict, show = True):
    target_t = target[:,:3]
    target_q = target[:,3:]
    predict_t = predict[:,:3]
    predict_q = predict[:,3:]

    #predict_q = F.normalize(torch.from_numpy(predict_q))
    #predict_q = predict_q.numpy()

    t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
    q_criterion = quaternion_angular_error
    t_loss = np.asarray([t_criterion(p, t) for p, t in zip(predict_t,
                                                       target_t)])
    q_loss = np.asarray([q_criterion(p, t) for p, t in zip(predict_q,
                                                           target_q)])
    if show:
        print ('Error in translation: median {:3.2f} m,  mean {:3.2f} m\n' \
            'Error in rotation: median {:3.2f} degrees, mean {:3.2f} degree'.format(np.median(t_loss), np.mean(t_loss),
                            np.median(q_loss), np.mean(q_loss)))
    return np.mean(t_loss), np.mean(q_loss), np.median(t_loss), np.median(q_loss)

def qexp(q):
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))
    return q

class Evaluator(object):
    
    def __init__(self, model, dataset, criterion, configs, superPoint_config, target):
        self.model = model
        self.sp_model = SuperPoint(superPoint_config).eval()
        self.criterion = criterion
        self.configs = configs
        self.target = target
        self.logdir = self.configs.logdir
        self.load_epoch = configs.load_epoch
        if not self.configs.load_best:
            self.checkpoint_file = os.path.join(self.logdir, 
                                                'epoch_{:03d}.pth.tar'.format(self.load_epoch))
        else:
            self.checkpoint_file = os.path.join(self.logdir, 
                                            'best_result.pth.tar'.format(self.load_epoch))
        print(self.checkpoint_file)
        
        # data loader 
        
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.configs.batch_size,
                                                        num_workers=self.configs.num_workers)
        #self.data_loader = dataset
        self.model = nn.DataParallel(self.model, device_ids=range(self.configs.GPUs))
        if self.configs.GPUs > 0:
            self.model.cuda()
            self.sp_model.cuda()
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
            if self.configs.load_best:
                test_loss = checkpoint['loss']
                print('Test_loss: {}'.format(test_loss))
            # c_state = checkpoint['criterion_state_dict']
            # append_dict = {k: torch.Tensor([0.0])
            #                for k, _ in self.criterion.named_parameters()
            #                if not k in c_state}
            # c_state.update(append_dict)
            # self.criterion.load_state_dict(c_state)
        else:
            print("Error: Can not load the model at epoch {}".format(self.load_epoch))

    def evaler(self, is_plot = True):
        self.model.eval()
        train_loss = 0.0 
        count = 0
        pbar = enumerate(self.data_loader)
        number_batch = len(self.data_loader)
        pbar = tqdm(pbar, total=number_batch)
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
                imgs_list = imgs_list + list(imgs)
                predict_ar = np.concatenate((predict_ar, predict.cpu().detach().numpy()), axis = 0)
                train_loss += loss.item() * n_samples
                count += n_samples
        train_loss /= count
        print("--LOSS: ", train_loss)
        predict_ar = np.delete(predict_ar, 0, 0)
        
        m,_ = predict_ar.shape
        assert m == len(imgs_list)
        name_col = np.zeros((m,1)) 
        predict_ar = np.concatenate((name_col, predict_ar), axis = 1)
        predict_ar = pd.DataFrame(predict_ar)
        predict_ar.iloc[:,0] = imgs_list
        mean_t, mean_q = get_errors(predict_ar.iloc[:,1:].to_numpy(), self.target, is_plot)
        
        if is_plot:
            plot_result(predict_ar.iloc[:,1:].to_numpy(), self.target, self.data_loader)
        
        # if (epoch % self.configs.snapshot==0):
        #     file1 = os.path.join(self.logdir, 'prediction_epoch_{:03d}.txt'.format(epoch))
        #     file3 = os.path.join(self.logdir, 'loss_epoch_{:03d}.txt'.format(epoch))
        #     print("\n Saving model and prediction result\n")
        #     predict_ar.to_csv(file1, header=False, index = False, sep = " ")
        #     pd.DataFrame([train_loss]).to_csv(file3, header=False, index = False, sep = " ")

    