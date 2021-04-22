import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt 
import pandas as pd 
import os 


data_file = "/home/thuan/Desktop/Paper2/augmented_idea/aalto_dataset_git/kitchen1/seq_01_poses"
global_path = "/home/thuan/Desktop/Paper2/augmented_idea/aalto_dataset_git/kitchen1/"

save_name = "left_augmentation"

saved_folder = global_path + save_name

try:
	os.mkdir(saved_folder)
except OSError:
    print ("Creation of the directory %s failed" % saved_folder)
else:
    print ("Successfully created the directory %s " % saved_folder)

csv_data = pd.read_csv(data_file, sep = " ")
list_path = csv_data.iloc[:,0]



img = cv.imread('test.png')
rows,cols,ch = img.shape


def cal_points(rows, cols, side = "left", val=1/32):
	# the side parameter need to be "left" or "right"

	if side != "left" or side != "right":
		print("__")
	
	o_point_1 = [int(cols/4), int(rows/4)]
	o_point_2 = [int((3/4)*cols), int(rows/4)]
	o_point_4 = [int((3/4)*cols), int((3/4)*rows)]
	o_point_3 = [int((1/4)*cols), int((3/4)*rows)]

	if side == "right":
		point_1 = [int(cols/4), int(rows/4)]
		point_2 = [int((3/4)*cols), int((1/4 + val)*rows)]
		point_4 = [int((3/4)*cols), int((3/4 - val)*rows)]
		point_3 = [int((1/4)*cols), int((3/4)*rows)]
	else:
		point_1 = [int(cols/4), int((1/4 + val)*rows)]
		point_2 = [int((3/4)*cols), int((1/4)*rows)]
		point_4 = [int((3/4)*cols), int((3/4)*rows)]
		point_3 = [int((1/4)*cols), int((3/4 - val)*rows)]

	return [o_point_1, o_point_2, o_point_3, o_point_4], [point_1, point_2, point_3, point_4]

# generating the data 

for i in range(len(list_path))


print(rows,cols,ch)

pts1, pts2 = cal_points(rows, cols)

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(cols, rows))
