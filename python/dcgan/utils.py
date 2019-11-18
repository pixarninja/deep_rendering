import cv2 as cv2
import glob as glob
import numpy as np
import os as os
import random
import torch

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def clear_dir(path):
    if os.path.exists(path):
        filelist = glob.glob(path + '*')
        for file in filelist:
            os.remove(file)
    else:
        os.mkdir(path)
        
def alter_image(img, alpha, beta):
    # Add noise.
    noise = np.random.normal(loc=0, scale=1, size=img.shape).astype('float32')
    img = cv2.addWeighted(img, alpha, noise, 1 - alpha, 0)

    # Gaussian blur.
    img = cv2.GaussianBlur(img, (beta, beta), 0)

    return torch.from_numpy(np.asarray(img).transpose(2, 0, 1))