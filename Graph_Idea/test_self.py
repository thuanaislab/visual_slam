import model_self as md 
import torch 
from superpoint import SuperPoint
from utils import read_image


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
pred['image'] = inp0
for k in pred:
	if isinstance(pred[k], (list, tuple)):
		pred[k] = torch.stack(pred[k])

model = md.MainModel(config['main_model']).eval().to(device)

model(pred)



# layers = [3, 10, 20]
# test = torch.randn(1,3,6)
# model = md.MLP(layers)
# out = model(test)
# print(out.shape)