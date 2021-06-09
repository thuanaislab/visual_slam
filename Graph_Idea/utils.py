import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt 

a = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SuperGlue/sift/svm.nvm.cmvs/00/cameras_v2.txt"
#a = "/home/thuan/Desktop/visual_slam/Data_for_superglue/cameras_v2.txt"
b = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SuperGlue/sift/poses.txt"




def extract_name(filename):
    # to get the name of image file
    # ex: "123.jpg" -> "123"
    out = ''
    for i in filename:
        if i == '.':
            break
        else:
            out = out + i
    return out 

def str2floatL(strline):
    ''' 
    Convert of a list as "1 2 3" into [1, 2, 3]
    Used to read the output data of visual SfM. camera_v2.txt
    ''' 
    out = []
    temp = ""
    for i in strline:
        if (i== " ") or (i == "\n"):
            out.append(float(temp))
            temp = ""
        else:
            temp = temp + i
            
    if temp != "":
        out.append(float(temp))
        temp = ""
    return out 
def filterlink(link):
    # used to read the camera_v2.txt file 
    link = link.replace("/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images_SuperGlue/sift/","")
    link = link.replace("\n","")
    return link

def readCamP(camera_v2, savePath):
    # read the camera position from camera_v2.txt
    index = [16, 19, 3, 14] # [number of images, first path, distance from path to 
                            #  camera position, distance to next path from current path] 
    i = 0
    nImages = 0
    iPath = index[1]
    iP = iPath + index[2]
    df_index = 0
    with open(camera_v2) as f:
        for line in f:
            if i == index[0]:
                nImages = int(line)
                df = pd.DataFrame(index=range(nImages),columns=range(8))
            if nImages != 0:
                if i == iPath:
                    iPath = iPath + index[3]
                    df.iloc[df_index,0] = filterlink(line)
                if i == iP:
                    iP = iP + index[3]
                    df.iloc[df_index,1:4] = str2floatL(line)
                if i == (iP-index[3]+3):
                    df.iloc[df_index,4:8] = str2floatL(line)
                    df_index = df_index + 1
            i = i + 1
    if savePath != "":
        df.to_csv(savePath, header = False, index = False, sep = " ")
    #print(df)
    #scatter_trajectory(df)


def scatter_trajectory(data, out_link):
    # sort the data into correct order 
    # for the purpose of trajectory plot
    # ex: 
    '''
        22.jpg 1 2 3
        1.jpg  2 1 1 
        ... 
        ==> 
        1.jpg  2 1 1
        22.jpg 1 2 3 
        ... 
    '''
    
    df = pd.DataFrame(index=range(365),columns=range(7))
    t = 0;
    for i in range(371):
        for ii in range(365):
            if i == int(data.iloc[ii,0].replace(".jpg","")):
                df.iloc[t,0:7] =  data.iloc[ii,1:8]
                t = t + 1
    print(df)
    df.to_csv(out_link, header = False, sep = " ", index = False)
    
    '''plt.scatter(data.iloc[:,1], data.iloc[:,2])
    plt.title('Scatter plot pythonspot.com')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()'''
    
    
def getRealCor(gTruth=1, vSfM=1):
    # get the real world coordinate system of TUM dataset 
    # which will be used to convert visualSfM to world system using GCP function 
    # Input:
    #       gTruth: the groundtruth file of TUM dataset
    #       vSfM: is the saving abriviation of TUMdataset
    # output: 
    #   a .gcp file such that each line gives: filename X Y Z
    vSfM = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images.txt"
    gTruth =  "/home/thuan/Downloads/rgbd_dataset_freiburg2_desk/groundtruth.txt"
    save_file = "/home/thuan/Desktop/TUM_images_SIFT/gcp3.gcp"
    g = pd.read_csv(gTruth, sep = " ")
    length_g, _ = g.shape 
    v = pd.read_csv(vSfM, sep = ",")
    step = 2
    nImg = 20
    df = pd.DataFrame(index=range(nImg),columns=range(4)) # filename X Y Z
    i = 0 
    ii = 0
    while (i < nImg):
        tmp_img = v.iloc[ii,1] # ex: 1.jpg
        tmp_time = v.iloc[ii,0].replace(".jpg","")
        tmp_time = float(tmp_time)
        for tmp_i in range(length_g):
            if tmp_i == 0:
                if (tmp_time < g.iloc[tmp_i,0]):
                    df.iloc[i,0] = tmp_img
                    df.iloc[i,1:4] = g.iloc[tmp_i,1:4]
                    break
            else:
                if (tmp_time > g.iloc[tmp_i-1,0]) and (tmp_time < g.iloc[tmp_i,0]):
                    df.iloc[i,0] = tmp_img
                    if (tmp_time - g.iloc[tmp_i-1,0]) > (g.iloc[tmp_i,0] - tmp_time):
                        df.iloc[i,1:4] = g.iloc[tmp_i,1:4]
                    else:
                        df.iloc[i,1:4] = g.iloc[tmp_i-1,1:4]
                    break
        i = i + 1
        ii = ii + step
    df.to_csv(save_file, header=False, sep = " ", index = False)



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