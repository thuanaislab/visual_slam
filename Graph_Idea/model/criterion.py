#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 17:56:03 2021

@author: thuan
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class PoseNetCriterion(nn.Module):
    def __init__(self, sx=-3.0, sq=-3.0, learn_smooth_term=True):
        super(PoseNetCriterion, self).__init__()
        self.sx_abs = nn.Parameter(torch.Tensor([sx]), requires_grad = bool(
            learn_smooth_term))
        self.sq_abs = nn.Parameter(torch.Tensor([sq]), requires_grad = bool(
            learn_smooth_term))
        
        #self.loss_func = nn.MSELoss()
        self.loss_func = nn.L1Loss()
    
    def forward(self, poses_pd, poses_gt):
        
       
        s = poses_pd.size()
        num_poses = s[0]
        
        t = poses_pd[:,:3]
        q = poses_pd[:,3:]
        #q = F.normalize(poses_pd[:,3:])
        t_gt = poses_gt[:,:3]
        q_gt = poses_gt[:,3:]
        
        abs_t_loss = self.loss_func(t, t_gt)
        abs_q_loss = self.loss_func(q, q_gt)
        
        pose_loss = torch.exp(-self.sx_abs)*(abs_t_loss) + self.sx_abs \
            + torch.exp(-self.sq_abs)*(abs_q_loss) + self.sq_abs
            
        return pose_loss


class CriterionVersion2(nn.Module):
    def __init__(self, sx=-3.0, sq=-3.0, learn_smooth_term=True, vq_coef = 0.2, commit_coef = 0.4):
        super(CriterionVersion2, self).__init__()
        self.sx_abs = nn.Parameter(torch.Tensor([sx]), requires_grad = bool(
            learn_smooth_term))
        self.sq_abs = nn.Parameter(torch.Tensor([sq]), requires_grad = bool(
            learn_smooth_term))
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        
        #self.loss_func = nn.MSELoss()
        self.loss_func = nn.L1Loss()
    
    def forward(self, poses_pd, poses_gt, z_e, emb):
        
       
        s = poses_pd.size()
        num_poses = s[0]
        
        t = poses_pd[:,:3]
        #q = F.normalize(poses_pd[:,3:])
        q = poses_pd[:,3:]
        t_gt = poses_gt[:,:3]
        q_gt = poses_gt[:,3:]
        
        abs_t_loss = self.loss_func(t, t_gt)
        abs_q_loss = self.loss_func(q, q_gt)
        
        pose_loss = torch.exp(-self.sx_abs)*(abs_t_loss) + self.sx_abs \
            + torch.exp(-self.sq_abs)*(abs_q_loss) + self.sq_abs
        

        # vq_loss = torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1))
        # commit_loss = torch.mean(
        #     torch.norm((emb.detach() - z_e)**2, 2, 1))
        
        vq_loss = self.loss_func(emb, z_e.detach())
        commit_loss = self.loss_func(emb.detach(), z_e)
        
        total_loss = pose_loss + self.vq_coef*vq_loss + self.commit_coef*commit_loss
            
        return total_loss, pose_loss, self.vq_coef*vq_loss, self.commit_coef*commit_loss


class PoseNetCriterionPlus(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, srx=0.0, srq=0.0, learn_beta=False, learn_gamma=False, ratio=1.0):
        super(PoseNetCriterionPlus, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
        self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
        self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)
        self.ratio = ratio

    def forward(self, pred, targ):
        # absolute pose loss
        s = pred.size()
        abs_loss = torch.exp(-self.sax) * self.t_loss_fn(pred[:, :3], targ[:, :3]) + self.sax + \
                   torch.exp(-self.saq) * self.q_loss_fn(pred[:, 3:], targ[:, 3:]) + self.saq
        # get the VOs
        
        targ_vos = calc_vos_simple(targ)
        if targ_vos == None:
            # the case of batch size == 1
            return abs_loss
        pred_vos = calc_vos_simple(pred)
        # VO loss
        s = pred_vos.size()
        vo_loss = torch.exp(-self.srx) * self.t_loss_fn(pred_vos[:, :3], targ_vos[:, :3]) + self.srx + \
                  torch.exp(-self.srq) * self.q_loss_fn(pred_vos[:, 3:], targ_vos[:, 3:]) + self.srq

        # total loss
        loss = abs_loss + self.ratio*vo_loss
        return loss

def calc_vos_simple(poses):
    vos = []
    l = poses.shape[0]
    if l == 1:
        return None
    for i in range(l-1):
        pvos = [poses[i+1,:] - poses[i,:]]
        vos.append(torch.cat(pvos, dim=0))
        if i != 0:
            pvos = [poses[0,:] - poses[i,:]] # i=1 is repeated --> should edit 
            vos.append(torch.cat(pvos, dim=0))
    vos = torch.stack(vos, dim=0)
    return vos

    
    
    
    
    
    
    