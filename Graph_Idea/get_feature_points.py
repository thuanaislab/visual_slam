import model as md 
import torch 
from superpoint import SuperPoint
from utils import read_image
import numpy as np 


img_path_1 = "/home/thuan/Desktop/teset_Data_representation/SuperGluePretrainedNetwork/1.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
resize = [640, 480]

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

super_point_model = SuperPoint(config.get('superpoint', {})).eval().to(device)

image0, inp0, scales0 = read_image(
    img_path_1, device, resize, 0, False)

pred = super_point_model({'image': inp0})


pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}


def sort_scores(scores):
	# sort the score values following the decrease of the importance. 
	# return the new scores list and the original index. 
	sorted_score = np.sort(scores)
	sorted_index = np.argsort(scores)

	sorted_score = np.flip(sorted_score)
	sorted_index = np.flip(sorted_index)
	return sorted_score, sorted_index

def sort_feature_Or_descriptors(data, sorted_index):
	length, dim = data.shape
	assert length == len(sorted_index) # the data length need to be equal length of 
									   # sorted_index 
	new_data = np.zeros((length,dim))
	for i in range(length):
		new_data[i,:] = data[sorted_index[i],:]
	return new_data


new_a, index = sort_scores(pred['scores'])
print(new_a)
print(index)
# print(b)
print(sort_feature_Or_descriptors(pred['descriptors'].T,index))

