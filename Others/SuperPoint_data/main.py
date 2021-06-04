from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2 
import matplotlib.cm as cm
import pandas as pd 
import os.path

from models.matching import Matching
from models.utils import (make_matching_plot, read_image)


img_path_1 = "1.png"
img_path_2 = "2.png"
global_image_folder = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images/"
image_infor = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images.txt"
consecutive = 3
discrete = [7,15]

save_sift_txt = "/home/thuan/Desktop/visual_slam/Data_for_superglue/sift_txt/"
save_matching_txt = "/home/thuan/Desktop/visual_slam/Data_for_superglue/matching.txt"


device = 'cuda' if torch.cuda.is_available() else 'cpu'
resize = [-1] #[640, 480]


########################################################################################################
########################################################################################################


def sort_scores(scores):
  # sort the score values following the decrease of the importance. 
  # return the new scores list and the original index. 
  sorted_score = np.sort(scores)
  sorted_index = np.argsort(scores)

  sorted_score = np.flip(sorted_score)
  sorted_index = np.flip(sorted_index)
  return sorted_score, sorted_index

def sort_feature_Or_descriptors(data, sorted_index):
  match_mode = False
  if len(data.shape) == 1:
    length = len(data)
    match_mode = True
  else:
    length, dim = data.shape
  assert length == len(sorted_index) # the data length need to be equal length of 
                     # sorted_index 
  if not match_mode:
    new_data = np.zeros((length,dim))
    for i in range(length):
      new_data[i,:] = data[sorted_index[i],:]
  else: # match mode
    new_data = np.zeros(length)
    for i in range(length):
      new_data[i] = data[sorted_index[i]]
  return new_data

def get_correct_new_matching_index(old_match0, new_index1):
  length = len(old_match0)
  new_match0 = -np.ones(length)
  for i in range(length):
    if old_match0[i] == -1:
      continue
    else:
      new_match0[i] = np.where(new_index1 == old_match0[i])[0][0]
  return new_match0

def get_list_matching_pairs(matches):
  length = len(matches)
  list_0 = ''
  list_1 = ''
  for i in range(length):
    if matches[i] == - 1:
      continue
    else:
        if i == (length-1):
            list_0 = list_0 + str(i)
            list_1 = list_1 + str(int(matches[i]))
        else:
            list_0 = list_0 + str(i) + ' ' 
            list_1 = list_1 + str(int(matches[i])) + ' '
  return list_0, list_1


def save_feature_locations(feature_points, path):
  m,n = feature_points.shape
  add_length = np.zeros((m+1,n))
  add_length[1:,:]=feature_points
  add_length[0,0] = m 
  temp = pd.DataFrame(add_length)
  temp.to_csv(path, header = False, index = False, sep = " ")


def save_matching_sfm(path, list0, list1, imgPath0, imgPath1):
    if os.path.isfile(path):
        #print("file name is existed")
        f = open(path, "a")
        f.write(imgPath0 + " " + imgPath1 + "\n")
        f.write(list0+ "\n")
        f.write(list1+ "\n")
    else:
        f = open(path, "w")
        f.write(imgPath0 + " " + imgPath1 + "\n")
        f.write(list0+ "\n")
        f.write(list1+ "\n")
    f.close()
    
def generate_imagePairs(consecutive = consecutive, discrete = discrete, infor_path = image_infor):
    infor = pd.read_csv(infor_path)
    list_img = infor.iloc[:,1]
    g_length = len(list_img)
    out_list = []
    for i in range(g_length):
        for ii in range(1,consecutive+1):
            if (i+ii) < g_length:
                out_list.append([list_img[i],list_img[i+ii]])
        for ii in discrete:
            if (i + ii) < g_length:
                out_list.append([list_img[i],list_img[i+ii]])
    return out_list
            
def checking(list_pairs):
    # used for verify the generate_imagePairs function. 
    i = 0
    while(True):
        temp = list_pairs.pop(i)
        if (temp in list_pairs) or (temp.reverse() in list_pairs):
            print("ERORRRRRRRR______")
        if (len(list_pairs) == 1):
            break
def extract_name(filename):
    out = ''
    for i in filename:
        if i == '.':
            break
        else:
            out = out + i
    return out 
    
########################################################################################################
########################################################################################################

config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': "indoor",
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}
matching = Matching(config).eval().to(device)

# Do the for loop in here for list of image pairs

listImagePairs = generate_imagePairs()
length_imgPairs = len(listImagePairs)
save_feature_list = []
for i in range(length_imgPairs):
    name_img_path_1 = listImagePairs[i][0]
    cur_savename1 = extract_name(name_img_path_1)

    name_img_path_2 = listImagePairs[i][1]
    cur_savename2 = extract_name(name_img_path_2)
    img_path_1 = global_image_folder + name_img_path_1
    img_path_2 = global_image_folder + name_img_path_2

    # Load the image pair.
    image0, inp0, scales0 = read_image(
        img_path_1, device, resize, 0, False)
    image1, inp1, scales1 = read_image(
        img_path_2, device, resize, 0, False)
    if image0 is None or image1 is None:
        print('Problem reading image pair: {} {}'.format(
            input_dir/name0, input_dir/name1))
        exit(1)
    
    
    
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
    
    
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    dscpt0, dscpt1 = pred['descriptors0'], pred['descriptors1']
    scores0, scores1 = pred['scores0'], pred['scores1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    
    
    # sort------
    # for image_0 
    n_scores0,nIn_scores0 = sort_scores(scores0)
    n_kpts0 = sort_feature_Or_descriptors(kpts0, nIn_scores0)
    n_dscpt0 = sort_feature_Or_descriptors(dscpt0.T, nIn_scores0)
    # for image_1
    n_scores1, nIn_scores1 = sort_scores(scores1)
    n_kpts1 = sort_feature_Or_descriptors(kpts1, nIn_scores1)
    n_dscpt1 = sort_feature_Or_descriptors(dscpt1.T, nIn_scores1)
    # completed sort 
    
    # next step is correct the matches order. 
    n_matches = sort_feature_Or_descriptors(matches, nIn_scores0)
    n_conf = sort_feature_Or_descriptors(conf, nIn_scores0)
    # --------------- 
    n_matches = get_correct_new_matching_index(n_matches, nIn_scores1)
    
    valid = n_matches > -1 
    mkpts0 = n_kpts0[valid]
    mkpts1 = n_kpts1[n_matches[valid].astype(int)]
    mconf = n_conf[valid]
    
    list_0, list_1 = get_list_matching_pairs(n_matches)
    if not cur_savename1 in save_feature_list:
        save_feature_locations(n_kpts0, save_sift_txt + cur_savename1 + '.txt')
        save_feature_list.append(cur_savename1)
    if not cur_savename2 in save_feature_list:
        save_feature_locations(n_kpts0, save_sift_txt + cur_savename2 + '.txt')
        save_feature_list.append(cur_savename2)
    save_matching_sfm(save_matching_txt, list_0, list_1, name_img_path_1, name_img_path_2)
    

# color = cm.jet(mconf)
# text = [
#                 'SuperGlue',
#                 'Keypoints',
#                 'Matches:'
#             ]

# make_matching_plot(image0, image1, n_kpts0, n_kpts1, mkpts0, mkpts1,
#                        color, text, path = None, show_keypoints=True,
#                        fast_viz=False, opencv_display=True,
#                        opencv_title='matches', small_text = text)

