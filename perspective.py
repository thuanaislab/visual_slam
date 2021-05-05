import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt 
import pandas as pd 
import os 


# data_file = "/home/thuan/Desktop/Paper2/augmented_idea/aalto_dataset_git/kitchen1/seq_01_poses"
global_path = "/home/thuan/Desktop/visual_slam/Data/Original/deer_robot/"
data_pose = global_path + "poses2"
data_file = global_path + "cam0/data.csv"
save_path = "/home/thuan/Desktop/visual_slam/Data/Augmentation/deer_robot/"

save_name = "left_augmentation"
data_folder_name = "cam0/data"
saved_folder = save_path + save_name +'/'

save_infor_path = save_path  + 'poses.txt'

try:
	os.mkdir(saved_folder)
except OSError:
    print ("Creation of the directory %s failed" % saved_folder)
else:
    print ("Successfully created the directory %s " % saved_folder)

try:
	os.mkdir(saved_folder + data_folder_name)
except OSError:
    print ("Creation of the directory %s failed" % saved_folder)
else:
    print ("Successfully created the directory %s " % saved_folder)

print('_______________________________________________________________________________________')

# read the image paths. 
csv_data = pd.read_csv(data_file)
#print(csv_data.head(5))
list_path = csv_data.iloc[:,1]
# read the pose paths. 
pose_data = pd.read_csv(data_pose, sep = " ", header = None)

#print("pose data, ", pose_data.head(5))

new_pose_data = {'path':[],'1':[],'2':[],'3':[],'4':[],'5':[],'6':[], '7':[]}

def cal_points(rows, cols, side = "left", val=1/32):
	# the side parameter need to be "left" or "right"

	if side != "left" and side != "right":
		print("ERROR: the side need to be right or left")
	
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

for i in range(len(list_path)):
	if i%2 == 0:
		file_name = list_path[i]

		#print("file_name: ", file_name)

		img = cv.imread(global_path + data_folder_name + '/' + file_name)
		rows, cols, ch = img.shape

		pts1, pts2 = cal_points(rows, cols)

		pts1 = np.float32(pts1)
		pts2 = np.float32(pts2)
		M = cv.getPerspectiveTransform(pts1,pts2)
		dst = cv.warpPerspective(img,M,(cols, rows))
		# save the result 
		cv.imwrite(saved_folder + file_name, dst)

		new_pose_data['path'].append(file_name)
		new_pose_data['1'].append(pose_data.iloc[i,1])
		new_pose_data['2'].append(pose_data.iloc[i,2])
		new_pose_data['3'].append(pose_data.iloc[i,3])
		new_pose_data['4'].append(pose_data.iloc[i,4])
		new_pose_data['5'].append(pose_data.iloc[i,5])
		new_pose_data['6'].append(pose_data.iloc[i,6])
		new_pose_data['7'].append(pose_data.iloc[i,7])

df = pd.DataFrame (new_pose_data, columns = ['path','1','2', '3', '4', '5', '6', '7'])

df.to_csv(save_infor_path, sep = ' ', header = False, index = False)

