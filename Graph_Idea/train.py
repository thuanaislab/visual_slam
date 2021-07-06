#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 23:37:44 2021

@author: thuan
"""

import model.model_2_0 as v2
from model.trainer import Trainer
import model.self_model as v1
from model.criterion import PoseNetCriterion, CriterionVersion2, PoseNetCriterionPlus
from model.dataloaders import CRDataset_train
import argparse
import pandas as pd
import os.path as osp 


parser = argparse.ArgumentParser()


parser.add_argument("--batch_size", type=int, default=13)
parser.add_argument("--shuffle", type=int, choices=[0, 1], default=1)
parser.add_argument("--num_workers", type=int, default=0,
                    help="The number of threads employed by the data loader")
# optimize
parser.add_argument("--sx", type=float, default=-3.0,
                    help="Smooth term for translation")
parser.add_argument("--sq", type=float, default=-3.0,
                    help="Smooth term for rotation")
parser.add_argument("--srx", type=float, default=-3.0,
                    help="Smooth term for translation")
parser.add_argument("--srq", type=float, default=-3.0,
                    help="Smooth term for rotation")
parser.add_argument("--learn_sxsq", type=int,
                    choices=[0, 1], default=1, help="whether learn sx, sq")
parser.add_argument("--vq_coef", type=float, default=1.0,
                    help="vq coef")
parser.add_argument("--commit_coef", type=float, default=1.0,
                    help="commit_coef")
parser.add_argument("--optimizer", type=str,
                    choices=['sgd', 'adam', 'rmsprop'], default='adam', help="The optimization strategy")
parser.add_argument("--lr", type=float, default=3e-4,
                    help="Base learning rate.")
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--lr_decay", type=float, default=1,
                    help="The decaying rate of learning rate")
# Loss Function
parser.add_argument("--is_VO", type=float, default=0, choices = [0,1],
                    help="is using VO for loss function")
parser.add_argument("--ratio", type=float, default=1.0, choices = [0,1],
                    help="is using VO for loss function")

parser.add_argument('--seed', type=int, default=0,
                    help='')
parser.add_argument('--GPUs', type=int, default=1,
                    help='The number of GPUs employed.')
parser.add_argument('--n_epochs', type=int, default=500,
                    help='The # training epochs')
# evaluate
parser.add_argument('--do_val', type=int,
                    choices=[0, 1], default=1, help='Whether do validation when training')
parser.add_argument('--scatter', type=int,
                    choices=[0, 1], default=1, help='Whether scatter the testing data')

parser.add_argument('--snapshot', type=int, default=5,
                    help='The snapshot frequency')

parser.add_argument('--checkpoint_file', type=str, default=None)
# log
parser.add_argument('--logdir', type=str, default='results/log',
                    help='The directory of logs')
parser.add_argument('--print_freq', type=int, default=1,
                    help='Print frequency every n epoch')
parser.add_argument('--save_checkpoint', type=int, default = 0,
                    help = 'save the prediction & target at snapshot')

# dataloader
parser.add_argument('--data_dir', type=str, default=
                    "",
                    help='The root dir of image dataset')
parser.add_argument('--train_poses_path', type=str, default=
                    "/home/thuan/Desktop/Public_dataset/Seven_Scenes/heads/poses_train.txt",
                    help='the root dir of label file poses.txt')
parser.add_argument('--test_poses_path', type=str, default=
                    "/home/thuan/Desktop/Public_dataset/Seven_Scenes/heads/poses_test.txt",
                    help='the root dir of label file poses.txt')
parser.add_argument('--dir_stats', type=str, default= "/home/thuan/Desktop/Public_dataset/Seven_Scenes/heads/",
                    help = "for loading the statitic file")
parser.add_argument('--resize', type=list, default = [-1],
                    help='resize image into [H,W]. [-1] no change')

# architecture
parser.add_argument('--version', type=int, default=1,
                    help='The version will be trained')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='The dropout probability')
parser.add_argument('--num_GNN_layers', type=int, default=2,
                    help="number of self attention graph network")
parser.add_argument('--lstm', type=int, default=0,choices=[0,1],
                    help="number of self attention graph network")

# superpoint 
parser.add_argument('--max_keypoints', type=int, default=1024,
                    help='the number of keypoints per image')
parser.add_argument('--pre_train', type=str, default=
                    '/home/thuan/Desktop/visual_slam/Graph_Idea/weights/superpoint_v1.pth',
                    help = "pre-trained model of superpoint")
# training fashion
parser.add_argument('--mul_step', type=int, choices=[0,1],
                    default=0, help="do multi steps training or not"
                    )
parser.add_argument('--train_poses_path_1', type=str, default=
                    "/home/thuan/Desktop/Public_dataset/Seven_Scenes/heads/poses_train.txt",
                    help='the root dir of label file poses.txt')
parser.add_argument('--Nepoch_step1', type=int,
                    default=10, help="number epoch for training step 1"
                    )
parser.add_argument('--Nepoch_step2', type=int,
                    default=200, help="number epoch for training step 2"
                    )
args = parser.parse_args()

# dataset 
train_loader = CRDataset_train(args.train_poses_path, args.data_dir, "cuda", args.resize)
if args.mul_step:
    train_loader_1 = CRDataset_train(args.train_poses_path_1, args.data_dir)
    criterion_1 = PoseNetCriterion(args.sx, args.sq, args.learn_sxsq) 
else:
    train_loader_1 = None 
    criterion_1 = None 
    
if args.do_val:
    test_loader = CRDataset_train(args.test_poses_path, args.data_dir)
else:
    test_loader = None 

# model 
config = {"num_GNN_layers":args.num_GNN_layers, "logdir": args.logdir, "lstm":args.lstm}
if args.version == 1:
    model = v1.MainModel(config)
    if not args.is_VO:
        criterion = PoseNetCriterion(args.sx, args.sq, args.learn_sxsq)
    else:
        print("VO loss is activated")
        kwargs = dict(sax=args.sx, saq=args.sq,srx=args.srx, srq=args.srq, learn_beta=True, learn_gamma=True,ratio = args.ratio)
        criterion = PoseNetCriterionPlus(**kwargs)
elif args.version == 2:
    model = v2.MainModel(config)
    criterion = CriterionVersion2(args.sx, args.sq, args.learn_sxsq, args.vq_coef, args.commit_coef)



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

test_target = pd.read_csv(args.test_poses_path, header = None, sep =" ")
test_target = test_target.iloc[:,1:].to_numpy()

trainLoader = [train_loader, train_loader_1]
cri = [criterion, criterion_1]

trainer = Trainer(model, optimizer_configs, trainLoader , test_loader, test_target, cri, args, superPoint_config)
if args.mul_step:
    trainer.train_1()
else:
    trainer.train()



























