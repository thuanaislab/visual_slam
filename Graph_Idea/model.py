
import torch 
from torch import nn
import copy

def MLP(channels: list, do_bn=True):
	# Multi layer perceptron 
	n = len(channels)
	layers = []
	for i in range(1,n):
		layers.append(
			nn.Conv1d(channels[i-1], channels[i], kernel_size = 1, bias =True))
		if i < (n-1):
			if do_bn:
				layers.append(nn.BatchNorm1d(channels[i]))
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
		'num_GNN_layers': 18,
	}

	def __init__(self, config):
		super().__init__()
		self.config = {**self.default_config,**config}
		self.keypoints_encoder = KeypointEncoder(
			self.config['descriptor_dim'], self.config['keypoint_encoder'])
		self.gnn = AttensionalGNN(self.config['num_GNN_layers'], self.config['descriptor_dim'])


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

		print(descpt[:,:,0])

		return descpt

