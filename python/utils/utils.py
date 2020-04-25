import cv2 as cv2
import glob as glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os as os
import random
import torch

def alter_image(img, alpha, beta):
    # Add noise.
    noise = np.random.normal(loc=0, scale=1, size=img.shape).astype('float32')
    img = cv2.addWeighted(img, alpha, noise, 1 - alpha, 0)

    # Gaussian blur.
    img = cv2.GaussianBlur(img, (beta, beta), 0)

    return torch.from_numpy(np.asarray(img).transpose(2, 0, 1))

def clear_dir(path):
    if os.path.exists(path):
        filelist = glob.glob(path + '/*')
        for f in filelist:
            try:
                os.remove(f)
            except:
                if len(os.listdir(f)) > 0:
                    print('rm ' + f)
                    clear_dir(f)
    else:
        os.mkdir(path)

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def evaluate_images(img_str_real, img_str_fake, img_str_out):
    # Read in images from input file paths.
    img_real = cv2.imread(img_str_real)
    img_real = cv2.resize(img_real, (0,0), fx=0.5, fy=0.5) 
    height_1, width_1 = img_real.shape[:2]
    img_fake = cv2.imread(img_str_fake)
    img_fake = cv2.resize(img_fake, (0,0), fx=0.5, fy=0.5)
    height_2, width_2 = img_fake.shape[:2]

    # Find minimum height and width.
    height = min(height_1, height_2)
    width = min(width_1, width_2)

    # Evaluate images.
    img_xor = cv2.bitwise_xor(img_real, img_fake)
    pixel_sum = np.sum(img_xor)
    img_out = cv2.bitwise_not(cv2.cvtColor(img_xor, cv2.COLOR_BGR2GRAY))

    # Print evaluation and save output image.
    cv2.imwrite(img_str_out + '.png', img_out)
    cv2.imwrite(img_str_out + '_real.png', img_real)
    cv2.imwrite(img_str_out + '_fake.png', img_fake)
    return 1 - (pixel_sum / (3 * 255.0 * width * height))

# Definition for plotting values.
def plot_together(values, colors, labels, title, path):
    samples = len(values[0])
    x = [i for i in range(samples)]
    x_axis = np.linspace(0, samples, samples, endpoint=True)
    patches = []
    
    # Plot values and fits.
    for i in range(len(values)):
        avg = np.average(values[i])
        plt.plot(x_axis, values[i], color=colors[i], alpha=0.33)
        plt.plot(np.unique(x), np.poly1d(np.polyfit(x, values[i], 1))(np.unique(x)), color=colors[i], linestyle='--')
        patches.append(mpatches.Patch(color=colors[i]))
        print(title + '[' + str(i) + ']: ' + str(avg))
    
    # Finish plot.
    plt.legend(patches, labels, loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Difference Ratio')
    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([0, samples])
    axes.set_ylim([0, 1])
    plt.tight_layout()
    
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path)
    plt.clf()
    
# Definition for plotting loss.
def plot_loss_single(value, color, label, title, path):
    samples = len(value)
    x = [i for i in range(samples)]
    x_axis = np.linspace(0, samples, samples, endpoint=True)
    
    # Plot values.
    avg = np.average(value)
    plt.plot(x_axis, value, color=color, alpha=0.33)
    plt.axhline(y=avg, color=color, xmin=0, xmax=samples, linestyle='--')
    loss_patch = mpatches.Patch(color=color)
    print(title + ': ' + str(avg))
    
    # Finish plot.
    plt.legend([loss_patch], [label], loc='lower right')
    plt.xlabel('Sample')
    plt.ylabel(label)
    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([0, samples])
    axes.set_ylim([0, 1])
    plt.tight_layout()
    
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path)
    plt.clf()
    
# Definition for plotting all loss.
def plot_loss_all(values, colors, labels, title, path):
    samples = len(values[0])
    x = [i for i in range(samples)]
    x_axis = np.linspace(0, samples, samples, endpoint=True)
    patches = []
    
    # Plot values.
    for i in range(len(values)):
        plt.plot(x_axis, values[i], color=colors[i], alpha=0.33)
        patches.append(mpatches.Patch(color=colors[i]))
    
    # Plot average lines.
    min = 1
    max = 0
    for i in range(len(values)):
        if min > np.min(values[i]):
            min = np.min(values[i])
        if max < np.max(values[i]):
            max = np.max(values[i])
        avg = np.average(values[i])
        plt.axhline(y=avg, color=colors[i], xmin=0, xmax=samples, linestyle='--', linewidth=0.5)
        
    # Finish plot.
    plt.legend(patches, labels, loc='lower right')
    plt.xlabel('Sample')
    plt.ylabel('Difference Ratio')
    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([0, samples])
    axes.set_ylim([0.725, 1])
    plt.tight_layout()
    
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path)
    plt.clf()
