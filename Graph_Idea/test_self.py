import self_model as md
import torch 
from superpoint import SuperPoint
from utils import read_image
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 

device = "cuda" if torch.cuda.is_available() else "cpu"
resize = [-1] #[640, 480]

config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold':0.0,
        'max_keypoints': 1024
    },
    'main_model':{
        'weight': 'indoor'
    }
}


class CRDataset_test(Dataset):
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

class CRDataset_train(Dataset):
    def __init__(self, poses_path:str, images_path:str, device:str, resize = [-1]):
        self.df = pd.read_csv(poses_path, header = None, sep = " ")
        self.images_path = images_path
        self.device = device
        self.resize = resize 
        
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        target = self.df.iloc[idx, 1:]
        target = np.array(target).astype(float)
        img_path = images_path + self.df.iloc[idx,0]
        _, img, _ = read_image(img_path, self.device, self.resize,0 ,False)
        
        target = torch.Tensor(target)
        _,_,m,n = img.shape
        img = img.view(1,m,n)
        
        return img, target 

poses_path = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SuperGlue/sift/poses.txt"
images_path = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SuperGlue/sift/"
load_data = CRDataset_train(poses_path, images_path, device)
# load_data_test = CRDataset_test(poses_path, images_path, config, device)
model = md.MainModel(config['main_model']).train().to(device)
superpoint = SuperPoint(config.get('superpoint', {})).eval().to(device)

criterion = md.Criterion().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(load_data, batch_size = 6, num_workers = 0, shuffle = False)

# model.eval()
# model(load_data_test[0]["features"])


number_batch = len(train_loader)
his_losses = []
for epoch in range(80):
    optimizer.zero_grad()
    pbar = enumerate(train_loader)
    pbar = tqdm(pbar, total=number_batch)
    count = 0
    train_loss = 0.0
    for i, (images, poses_gt) in pbar:
        images = images.to(device)
        poses_gt = poses_gt.to(device)
        n_samples = images.shape[0]
        with torch.no_grad():
            super_point_results = superpoint.forward_training({"image": images})
            keypoints = torch.stack(super_point_results['keypoints'], 0)
            descriptors = torch.stack(super_point_results['descriptors'], 0)
            scores = torch.stack(super_point_results['scores'], 0)
        superglue_inputs = {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "image": images,
            "scores": scores}
        out = model(superglue_inputs)
        total_loss = criterion(out, poses_gt)
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += total_loss.item() * n_samples
        count += n_samples
    train_loss /= count
    his_losses.append(train_loss)
    print("\nLoss Epoch {} is {}\n".format(epoch, train_loss))
    
plt.plot(his_losses)
plt.show()

a = pd.DataFrame(his_losses)
#a.to_csv("/home/thuan/Dropbox/MASTERWORK/Robotics/Experiments/Experiment_1/his_loss_1_2.txt")

















