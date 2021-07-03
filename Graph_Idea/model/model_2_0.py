#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 01:40:52 2021

@author: thuan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 14:44:54 2021
Self-attention parts are mainly based on SuperGlue paper https://arxiv.org/abs/1911.11763
VQ-VAE is based on 
https://github.com/nadavbh12/VQ-VAE/blob/a360e77d43ec43dd5a989f057cbf8e0843bb9b1f/vq_vae/nearest_embed.py#L90

@author: thuan 
"""


import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function, Variable
import copy
import os 

BN_MOMENTUM = 0.1





class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)

        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(
                emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)

        _, argmin = dist.min(-1)

        shifted_shape = [input.shape[0], *
                         list(input.shape[2:]), input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)
                                      ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) ==
                           latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(
                ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)


# adapted from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py#L25
# that adapted from https://github.com/deepmind/sonnet



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
        

        


class MainModel(nn.Module):

    default_config = {
        'descriptor_dim': 256,
        'keypoint_encoder': [32, 64, 128, 256],
        'num_GNN_layers': 9,
        'num_k': 512,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config,**config}
        print("num_GNN_layers {}".format(self.config['num_GNN_layers']))
        self.keypoints_encoder = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
        self.gnn = AttensionalGNN(self.config['num_GNN_layers'], self.config['descriptor_dim'])
        self.conv1 = nn.Conv1d(256, 2048, 1)
        #self.conv2 = nn.Conv1d(512, 1024, 1)
        
        self.fc1 = nn.Linear(2048, 40)
        #self.fc2 = nn.Linear(1024,40)
        self.fc3_r = nn.Linear(40, 3)
        self.fc3_t = nn.Linear(40, 3)
        
        self.bn = nn.BatchNorm1d(2048, momentum=BN_MOMENTUM)
        self.bn1 = nn.BatchNorm1d(512, momentum=BN_MOMENTUM)
        self.bn2 = nn.BatchNorm1d(1024, momentum=BN_MOMENTUM)
        self.bn3 = nn.BatchNorm1d(40, momentum=BN_MOMENTUM)
        
        # VQ-VAE 
        self.emb = NearestEmbed(self.config['num_k'],self.config['descriptor_dim'])



    def forward(self, data):
        descpt = data['descriptors']
        keypts = data['keypoints']
        scores = data['scores']

        # normalize keypoints 
        keypts = normalize_keypoints(keypts, data['image'].shape)
        # Keypoint MLP encoder
        descpt = descpt + self.keypoints_encoder(keypts, scores)
        # Multi layer transformer network
        descpt = self.gnn(descpt)
        
        z_q, argmin = self.emb(descpt, weight_sg=True)
        
        
        emb, _ = self.emb(descpt.detach())
        # print("descpt.shape", descpt.shape)
        # print("z_q.shape", z_q.shape)
        # out = torch.cat([descpt, z_q], dim = 1)
        # out = descpt + z_q.detach()
        out = F.relu(self.conv1(z_q))
        #out = F.relu(self.bn2(self.conv2(out)))
        out = nn.MaxPool1d(out.size(-1))(out)
        out = nn.Flatten(1)(out)
        
        out = F.relu(self.fc1(out))
        #out = F.relu(self.fc2(out))
        
        out_r = self.fc3_r(out)
        out_t = self.fc3_t(out)

        return torch.cat([out_t, out_r], dim = 1), descpt, emb, argmin
    
    def save_emb(self):
        print("??")
        save_path = os.path.join(self.config['logdir'], 'embs.pth.tar')
        checkpoint_dict = {'model_state_dict': self.emb.state_dict()}
        torch.save(checkpoint_dict, save_path)


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        