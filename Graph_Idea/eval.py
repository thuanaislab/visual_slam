#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 14:38:24 2021

@author: thuan
"""

from model import *
from model.evaluator import Evaluator, get_errors, plot_result
from model.self_model import MainModel
from model.criterion import PoseNetCriterion
from model.dataloaders import CRDataset_train
import argparse
import pandas as pd
import torch.nn.functional as F 
import torch



parser = argparse.ArgumentParser()


parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--shuffle", type=int, choices=[0, 1], default=0)
parser.add_argument("--num_workers", type=int, default=0,
                    help="The number of threads employed by the data loader")
# optimize
parser.add_argument("--sx", type=float, default=-0,
                    help="Smooth term for translation")
parser.add_argument("--sq", type=float, default=-0,
                    help="Smooth term for rotation")
parser.add_argument("--learn_sxsq", type=int,
                    choices=[0, 1], default=0, help="whether learn sx, sq")


parser.add_argument('--GPUs', type=int, default=1,
                    help='The number of GPUs employed.')
parser.add_argument('--load_epoch', type=int, default=1000,
                    help='The epoch number will be loaded')

# log
parser.add_argument('--logdir', type=str, default='log',
                    help='The directory of logs')

# dataloader
parser.add_argument('--data_dir', type=str, default=
                    "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SuperGlue/sift/",
                    help='The root dir of image dataset')
parser.add_argument('--poses_path', type=str, default=
                    "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SuperGlue/sift/poses.txt",
                    help='the root dir of label file poses.txt')
parser.add_argument('--resize', type=list, default = [-1],
                    help='resize image into [H,W]. [-1] no change')

# superpoint 
parser.add_argument('--max_keypoints', type=int, default=1024,
                    help='the number of keypoints per image')
parser.add_argument('--pre_train', type=str, default=
                    '/home/thuan/Desktop/visual_slam/Graph_Idea/weights/superpoint_v1.pth',
                    help = "pre-trained model of superpoint")
# architecture
parser.add_argument('--num_GNN_layers', type=int, default=9,
                    help="number of self attention graph network")
# Results 
parser.add_argument('--prediction_result', type = str, default=
                    "/home/thuan/Desktop/visual_slam/Graph_Idea/log/prediction_epoch_1000.txt",
                    help='path to prediction poses')

args = parser.parse_args()

if args.GPUs > 0:
    device = 'cuda'
else:
    device = 'cpu'

superPoint_config = {
    'nms_radius': 4,
    'keypoint_threshold':0.0,
    'max_keypoints': args.max_keypoints,
    'pre_train': args.pre_train,
        }

# dataset 
data_loader = CRDataset_train(args.poses_path, args.data_dir)

# model 
config = {"num_GNN_layers":args.num_GNN_layers}
model = MainModel(config)

# criterion
criterion = PoseNetCriterion(args.sx, args.sq, args.learn_sxsq)


# eval
target = pd.read_csv(args.poses_path, header = None, sep =" ")
predict = pd.read_csv(args.prediction_result, header = None, sep =" ")
target = target.iloc[:,1:].to_numpy()
predict = predict.iloc[:,1:].to_numpy()


plot_result(predict, target, data_loader)


target_t = target[:,:3]
target_q = target[:,3:]
predict_t = predict[:,:3]
predict_q = predict[:,3:]
# get_errors(predict, target)

print(predict_q.shape)
predict_q = F.normalize(torch.from_numpy(predict_q))
predict_q = predict_q.numpy()
print(predict_q.shape)
get_errors(target_t, target_q, predict_t, predict_q)

# eval_ = Evaluator(model, data_loader, criterion, args, superPoint_config)
# eval_.evaler()





