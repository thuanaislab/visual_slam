import pandas as pd 
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


estimated = pd.read_csv("/home/thuan/Desktop/estimate_poses.txt", header = None, sep = " ")
ground_truth = pd.read_csv("/home/thuan/Desktop/visual_slam/Data/Augmentation/deer_robot/poses.txt", header = None, sep = " ")

line_e = estimated.iloc[:,:3]
line_g = ground_truth.iloc[:,1:4]

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')


ax.plot(line_g.iloc[:,0], line_g.iloc[:,1], line_g.iloc[:,2], label='ground truth trajectory')
ax.plot(line_e.iloc[:,0], line_e.iloc[:,1], line_e.iloc[:,2], label='estimated of augmentation data ')
ax.legend()

plt.show()
