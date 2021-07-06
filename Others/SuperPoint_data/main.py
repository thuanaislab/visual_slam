from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2 
import matplotlib.cm as cm
import pandas as pd 
import os.path as osp 
import os

from models.matching import Matching
from models.utils import (make_matching_plot, read_image)

### These information will not be changed 
dictionary_train = {"chess": ["seq-01", "seq-02", "seq-04", "seq-06"], "fire": 
                     ["seq-01", "seq-02"], "heads": [ "seq-02"], "office": 
                     ["seq-01", "seq-03", "seq-04", "seq-05", "seq-08", "seq-10"], 
                     "pumpkin": ["seq-02", "seq-03", "seq-06", "seq-08"], "redkitchen":
                     ["seq-01", "seq-02", "seq-05", "seq-07", "seq-08", "seq-11", "seq-13"],
                     "stairs": ["seq-02", "seq-03", "seq-05", "seq-06"]}
seven_scenses_path = "/home/thuan/Desktop/Public_dataset/Seven_Scenes"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resize = [-1] #[640, 480]
####### ---------


scene = "heads"
seq = "seq-02"
infor_file = "poses_train.txt"
global_image_folder = osp.join(seven_scenses_path, scene, seq)


consecutive = 3
discrete = [7,15]



########################################################################################################
########################################################################################################

def create_vsfm_data(seq, dataloader, start_point, convert = False):
    # This function is used to convert .png images folder to 
    # .jpg folder and change the images name from 1.jpg, .. N.jpg
    # which will be used in Visual SFM tool 
    # input: seq : (int) - sequence number 
    #        dataloader: (pandas) - the poses infor
    #        rangeSeq: (list of 2 elements) - start and stop point
    # seq is the sequence number (int)
    folder = osp.join(seven_scenses_path, scene, "vsfm_"+ str(seq))
    # Create the vsfm image folder 
    if convert:
        try:
            os.makedirs(folder)
            print("Created folder path", folder)
        except: 
            print("Can not create the folder path ", folder)
        infor_file = "infor_file.txt"
        l = 1000
        stop_point = start_point+l
        temp = np.empty((l,2))
        temp = pd.DataFrame(temp)
        new_names = [str(i)+".jpg" for i in range(l)]
        temp.iloc[:,0] = dataloader.iloc[start_point:stop_point,0]
        temp.iloc[:,1] = new_names
        temp.to_csv(folder + "/" +  infor_file,header=False, index = False, sep = " ")
        for i in range(l):
            image = cv2.imread(temp.iloc[i,0])
            saveFile = folder + "/" + temp.iloc[i,1]
            cv2.imwrite(saveFile, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return folder


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
  number_match = 0
  for i in range(length):
    if matches[i] == - 1:
      continue
    else:
        number_match = number_match + 1
        if i == (length-1):
            list_0 = list_0 + str(i)
            list_1 = list_1 + str(int(matches[i]))
        else:
            list_0 = list_0 + str(i) + ' ' 
            list_1 = list_1 + str(int(matches[i])) + ' '
  return list_0, list_1, number_match


def save_feature_locations(feature_points, path):
  m,n = feature_points.shape
  add_length = np.zeros((m+1,n))
  add_length[1:,:]=feature_points
  add_length[0,0] = m 
  add_length = add_length. astype(int)
  temp = pd.DataFrame(add_length)
  temp.to_csv(path, header = False, index = False, sep = " ")


def save_matching_sfm(path, list0, list1, imgPath0, imgPath1, num_matches):
    if os.path.isfile(path):
        #print("file name is existed")
        f = open(path, "a")
        f.write(imgPath0 + " " + imgPath1 + " " + str(num_matches) +  "\n")
        f.write(list0+ "\n")
        f.write(list1+ "\n")
    else:
        f = open(path, "w")
        f.write(imgPath0 + " " + imgPath1 + " " + str(num_matches) + "\n")
        f.write(list0+ "\n")
        f.write(list1+ "\n")
    f.close()
    
def generate_imagePairs(consecutive = consecutive, discrete = discrete, infor_path = ""):
    infor = pd.read_csv(infor_path, header = None, sep = " ")
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
        'keypoint_threshold': 0.0,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': "indoor",
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}
matching = Matching(config).eval().to(device)

list_seq = dictionary_train[scene]
dataloader = pd.read_csv(osp.join(seven_scenses_path,scene, infor_file), header = None, sep = " ") 
for step in range(len(list_seq)):
    folder = create_vsfm_data(list_seq[step], dataloader, 0, True)
    save_sift_txt = osp.join(folder, "sift")
    try:
        os.makedirs(save_sift_txt)
    except:
        print("Can not create the folder ", save_sift_txt)
    save_matching_txt = folder + "/" + "matching.txt"
    infor_path = osp.join(folder, "infor_file.txt")
    listImagePairs = generate_imagePairs(infor_path = infor_path)
    length_imgPairs = len(listImagePairs)
    save_feature_list = []
    for i in range(length_imgPairs):
        name_img_path_1 = listImagePairs[i][0]
        cur_savename1 = extract_name(name_img_path_1)
    
        name_img_path_2 = listImagePairs[i][1]
        cur_savename2 = extract_name(name_img_path_2)
        img_path_1 = osp.join(folder, name_img_path_1)
        img_path_2 = osp.join(folder, name_img_path_2)
    
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
        
        list_0, list_1, num_matches = get_list_matching_pairs(n_matches)
        if not cur_savename1 in save_feature_list:
            save_feature_locations(n_kpts0, osp.join(save_sift_txt, cur_savename1 + '.txt'))
            save_feature_list.append(cur_savename1)
        if not cur_savename2 in save_feature_list:
            save_feature_locations(n_kpts1,osp.join(save_sift_txt, cur_savename2 + '.txt'))
            save_feature_list.append(cur_savename2)
        save_matching_sfm(save_matching_txt, list_0, list_1, name_img_path_1, name_img_path_2, num_matches)

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

