#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 14:44:54 2021
Self-attention parts are mainly based on SuperGlue paper https://arxiv.org/abs/1911.11763
@author: thuan 
"""


import torch 
from torch import nn
import torch.nn.functional as F
import copy

BN_MOMENTUM = 0.1

def MLP(channels: list, do_bn=False):
    # Multi layer perceptron 
    n = len(channels)
    layers = []
    for i in range(1,n):
        layers.append(
            nn.Conv1d(channels[i-1], channels[i], kernel_size = 1, bias =True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i], momentum=BN_MOMENTUM))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def normalize_keypoints(kpoints, image_shape):
    # Normalize the keypoints locations based on the image shape
    _, _, height, width = image_shape
    one = kpoints.new_tensor(1) 
    size = torch.stack([one*width, one*height])[None]
    center = size/2
    scaling = size.max(1, keepdim = True).values*0.7 # multiply with 0.7 because of discarded area when extracting the feature points
    return (kpoints- center[:,None,:]) / scaling[:,None,:]

class KeypointEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, keypoints, scores):
        inputs = [keypoints.transpose(1,2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim = 1))

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key)
    pros = torch.nn.functional.softmax(scores, dim=-1)/dim**0.5
    return torch.einsum('bhnm,bdhm->bdhn', pros, value)

class Multi_header_attention(nn.Module):
    """Multiheader attention class"""
    def __init__(self, num_head: int, f_dimension: int):
        super().__init__()
        assert f_dimension % num_head == 0
        self.dim = f_dimension // num_head
        self.num_head = num_head
        self.merge = nn.Conv1d(f_dimension, f_dimension, kernel_size = 1)
        self.proj = nn.ModuleList([copy.deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        query, key, value = [l(x).view(batch_size, self.dim, self.num_head,
            -1) for l,x in zip(self.proj, (query, key, value))]
        x = attention(query, key, value)

        return self.merge(x.contiguous().view(batch_size, self.dim*self.num_head,-1))

class AttentionalPropagation(nn.Module):
    """AttentionalPropagation"""
    def __init__(self, num_head: int, f_dimension: int):
        super().__init__()
        self.attn  = Multi_header_attention(num_head, f_dimension)
        self.mlp = MLP([f_dimension*2, f_dimension*2, f_dimension])
        nn.init.constant_(self.mlp[-1].bias, 0.0)
    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim = 1))

class AttensionalGNN(nn.Module):
    def __init__(self, num_GNN_layers: int, f_dimension: int):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(4,f_dimension)
            for _ in range(num_GNN_layers)])
    def forward(self, descpt):
        for layer in self.layers:
            delta = layer(descpt, descpt)
            descpt = descpt + delta
        return descpt
        

    
class FourDirectionalLSTM(nn.Module):
    def __init__(self, seq_size, origin_feat_size, hidden_size):
        super(FourDirectionalLSTM, self).__init__()
        self.feat_size = origin_feat_size // seq_size
        self.seq_size = seq_size
        self.hidden_size = hidden_size
        self.lstm_rightleft = nn.LSTM(self.feat_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_downup = nn.LSTM(self.seq_size, self.hidden_size, batch_first=True, bidirectional=True)

    def init_hidden_(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_size).to(device),
                torch.randn(2, batch_size, self.hidden_size).to(device))

    def forward(self, x):
        
        batch_size = x.size(0)
        x_rightleft = x.view(batch_size, self.seq_size, self.feat_size)
        x_downup = x_rightleft.transpose(1, 2)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)
        _, (hidden_state_lr, _) = self.lstm_rightleft(x_rightleft, hidden_rightleft)
        _, (hidden_state_ud, _) = self.lstm_downup(x_downup, hidden_downup)
        hlr_fw = hidden_state_lr[0, :, :]
        hlr_bw = hidden_state_lr[1, :, :]
        hud_fw = hidden_state_ud[0, :, :]
        hud_bw = hidden_state_ud[1, :, :]
        return torch.cat([hlr_fw, hlr_bw, hud_fw, hud_bw], dim=1)
    

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.g = nn.Linear(in_channels, in_channels // 8)
        self.theta = nn.Linear(in_channels, in_channels // 8)
        self.phi = nn.Linear(in_channels, in_channels // 8)

        self.W = nn.Linear(in_channels // 8, in_channels)

    def forward(self, x):
        batch_size = x.size(0)
        out_channels = x.size(1)

        g_x = self.g(x).view(batch_size, out_channels // 8, 1)

        theta_x = self.theta(x).view(batch_size, out_channels // 8, 1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, out_channels // 8, 1)
        f = torch.matmul(phi_x, theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.view(batch_size, out_channels // 8)
        W_y = self.W(y)
        z = W_y + x
        return z

class MainModel(nn.Module):

    default_config = {
        'descriptor_dim': 256,
        'keypoint_encoder': [32, 64, 128, 256],
        'num_GNN_layers': 9,
        'num_hidden':2048,
        'num_hiden_2':40,
        'lstm': False,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config,**config}
        print("num_GNN_layers {}".format(self.config['num_GNN_layers']))
        self.keypoints_encoder = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
        self.gnn = AttensionalGNN(self.config['num_GNN_layers'], self.config['descriptor_dim'])
        self.conv1 = nn.Conv1d(256, self.config['num_hidden'], 1)
        #self.conv2 = nn.Conv1d(512, 1024, 1)
        if self.config['lstm']:
            self.fc1 = nn.Linear(self.config['num_hidden']//2, self.config['num_hiden_2'])
            #self.fc2 = nn.Linear(1024,40)
            self.fc3_r = nn.Linear(self.config['num_hidden']//2, 3)
            self.fc3_t = nn.Linear(self.config['num_hidden']//2, 3)
        else:
            self.fc1 = nn.Linear(self.config['num_hidden'], self.config['num_hiden_2'])
            #self.fc2 = nn.Linear(1024,40)
            self.fc3_r = nn.Linear(self.config['num_hiden_2'], 3)
            self.fc3_t = nn.Linear(self.config['num_hiden_2'], 3)
        
        self.bn = nn.BatchNorm1d(2048, momentum=BN_MOMENTUM)
        self.bn1 = nn.BatchNorm1d(512, momentum=BN_MOMENTUM)
        self.bn2 = nn.BatchNorm1d(1024, momentum=BN_MOMENTUM)
        self.bn3 = nn.BatchNorm1d(40, momentum=BN_MOMENTUM)
        if self.config['lstm']:
            self.lstm4dir = FourDirectionalLSTM(seq_size=32, origin_feat_size=2048, hidden_size=256)



    def forward(self, data):
        descpt = data['descriptors']
        keypts = data['keypoints']
        scores = data['scores']

        # normalize keypoints 
        keypts = normalize_keypoints(keypts, data['image'].shape)
        # Keypoint MLP encoder
        key_encodes = self.keypoints_encoder(keypts, scores)
        descpt = descpt + key_encodes
        # Multi layer transformer network
        descpt = self.gnn(descpt) 
        out = F.relu(self.conv1(descpt))
        #out = F.relu(self.bn2(self.conv2(out)))
        out = nn.MaxPool1d(out.size(-1))(out)
        out = nn.Flatten(1)(out)
        if self.config['lstm']:
            out = self.lstm4dir(out)
            out_r = self.fc3_r(out)
            out_t = self.fc3_t(out)
        else:
            out = F.relu(self.fc1(out))
            out_r = self.fc3_r(out)
            out_t = self.fc3_t(out)
        
        

        return torch.cat([out_t, out_r], dim = 1)


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        