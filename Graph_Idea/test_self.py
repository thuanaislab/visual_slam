import self_model as md 
import torch 
from superpoint import SuperPoint
from utils import read_image
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np

img_path_1 = "/home/thuan/Desktop/teset_Data_representation/SuperGluePretrainedNetwork/1.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
resize = [-1] #[640, 480]

config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold':0.005,
        'max_keypoints': 1024
    },
    'main_model':{
        'weight': 'indoor'
    }
}


class CRDataset(Dataset):
    def __init__(self, poses_path:str, images_path:str, config: dict, device:str, resize = [-1]):
        self.df = pd.read_csv(poses_path, header = None, sep = " ")
        self.images_path = images_path
        self.config = config 
        self.device = device
        self.resize = resize 
        self.superpoint = SuperPoint(self.config.get('superpoint', {})).eval().to(device)
        
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        target = self.df.iloc[idx, 1:]
        target = np.array(target).astype(float)
        img_path = images_path + self.df.iloc[idx,0]
        _, img, _ = read_image(img_path, self.device, self.resize,0 ,False)
        features = self.superpoint({"image": img})
        features["image"] = img
        target = torch.Tensor(target)
        for k in features:
             if isinstance(features[k], (list, tuple)):
                features[k] = torch.stack(features[k])
        sample = {'features': features, 'target': target}
        
        return sample 
        
poses_path = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SuperGlue/sift/poses.txt"
images_path = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SuperGlue/sift/"
load_data = CRDataset(poses_path, images_path, config, device)

model = md.MainModel(config['main_model']).eval().to(device)
model(load_data[1]["features"])


