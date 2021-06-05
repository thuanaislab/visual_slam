import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt 

a = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SIFT/cameras_v2.txt"
b = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SIFT/orig_sift2txt/"

def extract_name(filename):
    out = ''
    for i in filename:
        if i == '.':
            break
        else:
            out = out + i
    return out 

def readCamP(camera_v2, savePath):
    # read the camera position from camera_v2.txt
    index = [16, 19, 5, 14] # [number of images, first path, distance from path to 
                            #  camera position, distance to next path from current path] 
    i = 0
    nImages = 0
    iPath = index[1]
    iP = iPath + index[2]
    iR = iP + 1
    with open(camera_v2) as f:
        for line in f:
            if i == index[0]:
                nImages = int(line)
            if nImages != 0:
                if i == iPath:
                    print(line)
                    iPath = iPath + index[3]
            i = i + 1
    print(nImages)


def scatter_trajectory(data):
    plt.scatter(data.iloc[:,5], data.iloc[:,6])
    plt.title('Scatter plot pythonspot.com')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    





def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new

def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def read_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales