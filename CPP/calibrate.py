import numpy as np 
import pandas as pd  

file_path = "/home/thuan/Desktop/Paper2/augmented_idea/aalto_dataset_git/kitchen1/seq_01_poses"

read_file = pd.read_csv(file_path, header = None, sep = " ")

read_file = read_file.drop([0], axis = 1)

print(read_file.head(5))

print(read_file.iloc[0,:])

