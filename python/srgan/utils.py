import cv2 as cv2
import glob as glob
import numpy as np
import os as os
import random
import torch

def alter_image(img, alpha, beta, pair = None):
    if pair is not None:
        alpha = random.uniform(alpha, pair[0])
        beta = int(random.uniform(beta, pair[1]))
        if beta % 2 != 1:
            beta += 1

    # Add noise.
    noise = np.random.normal(loc=0, scale=1, size=img.shape).astype('float32')
    img = cv2.addWeighted(img, alpha, noise, 1 - alpha, 0, dtype=cv2.CV_32F)

    # Gaussian blur.
    img = cv2.GaussianBlur(img, (beta, beta), 0)

    return torch.from_numpy(np.asarray(img).transpose(2, 0, 1))
    
def clear_dir(path):
    if os.path.exists(path):
        filelist = glob.glob(path + '*')
        for file in filelist:
            os.remove(file)
    else:
        os.mkdir(path)

def normalize_images(images):
    return (np.array(images) - np.array(images).min(0)) / np.array(images).ptp(0)
