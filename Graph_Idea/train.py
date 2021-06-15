#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 23:37:44 2021

@author: thuan
"""

from model import *
from model.trainer import Trainer
from model.self_model import MainModel
from model.criterion import PoseNetCriterion
from model.dataloaders import CRDataset_train
import argparse


parser = argparse.ArgumentParser()


parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--shuffle", type=int, choices=[0, 1], default=0)
parser.add_argument("--num_workers", type=int, default=0,
                    help="The number of threads employed by the data loader")
# optimize
parser.add_argument("--sx", type=float, default=-3,
                    help="Smooth term for translation")
parser.add_argument("--sq", type=float, default=-3,
                    help="Smooth term for rotation")
parser.add_argument("--learn_sxsq", type=int,
                    choices=[0, 1], default=1, help="whether learn sx, sq")
parser.add_argument("--optimizer", type=str,
                    choices=['sgd', 'adam', 'rmsprop'], default='adam', help="The optimization strategy")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Base learning rate.")
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--lr_decay", type=float, default=1,
                    help="The decaying rate of learning rate")

parser.add_argument('--seed', type=int, default=0,
                    help='')
parser.add_argument('--GPUs', type=int, default=2,
                    help='The number of GPUs employed.')
parser.add_argument('--n_epochs', type=int, default=500,
                    help='The # training epochs')

parser.add_argument('--do_val', type=int,
                    choices=[0, 1], default=1, help='Whether do validation when training')

parser.add_argument('--snapshot', type=int, default=20,
                    help='The snapshot frequency')

parser.add_argument('--checkpoint_file', type=str, default=None)
# log
parser.add_argument('--logdir', type=str, default='log',
                    help='The directory of logs')
parser.add_argument('--print_freq', type=int, default=1,
                    help='Print frequency every n epoch')
parser.add_argument('--save_pairs', type=int, default = 0,
                    help = 'save the prediction & target at snapshot')

# dataloader
parser.add_argument('--data_dir', type=str, default=
                    "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SuperGlue/sift/",
                    help='The root dir of image dataset')
parser.add_argument('--poses_path', type=str, default=
                    "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SuperGlue/sift/poses.txt",
                    help='the root dir of label file poses.txt')
parser.add_argument('--resize', type=list, default = [-1],
                    help='resize image into [H,W]. [-1] no change')

# architecture
parser.add_argument('--dropout', type=float, default=0.2,
                    help='The dropout probability')
parser.add_argument('--num_GNN_layers', type=int, default=9,
                    help="number of self attention graph network")

# superpoint 
parser.add_argument('--max_keypoints', type=int, default=1024,
                    help='the number of keypoints per image')
parser.add_argument('--pre_train', type=str, default=
                    '/home/thuan/Desktop/visual_slam/Graph_Idea/weights/superpoint_v1.pth',
                    help = "pre-trained model of superpoint")


args = parser.parse_args()

# dataset 
train_loader = CRDataset_train(args.poses_path, args.data_dir)

# model 
config = {"num_GNN_layers":args.num_GNN_layers}
model = MainModel(config)

# criterion
criterion = PoseNetCriterion(args.sx, args.sq, args.learn_sxsq)


optimizer_configs = {
    'method': args.optimizer,
    'base_lr': args.lr,
    'weight_decay': args.weight_decay,
    'lr_decay': args.lr_decay,
    'lr_stepvalues': [k/4*args.n_epochs for k in range(1, 5)]
}

superPoint_config = {
    'nms_radius': 4,
    'keypoint_threshold':0.0,
    'max_keypoints': args.max_keypoints,
    'pre_train': args.pre_train,
        }

# train 

trainer = Trainer(model, optimizer_configs, train_loader, criterion, args, superPoint_config)
trainer.train()



























