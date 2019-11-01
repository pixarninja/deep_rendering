# python evaluate_models.py

import cv2
import glob
import numpy as np
import os
import utils
import random

def evaluate_training(in_paths, models, niter, eval_folder, out_paths, colors):
    # Initialize prefix variables.
    out_path_1 = eval_folder + out_paths[0]
    out_path_2 = eval_folder + out_paths[1]
    real_prefix = 'real_'
    fake_prefix = 'fake_'
    t1_prefix = 'real_to_fake_1_'
    t2_prefix = 'real_to_fake_2_'

    # Initialize lists.
    values_1 = []
    values_2 = []

    # Clear output path images.
    utils.make_dir(eval_folder)
    utils.clear_dir(out_path_1)
    utils.clear_dir(out_path_2)

    # Start main loop.
    for epoch in range(niter):
        epoch_str = ('%03d' % epoch)
        
        img_str_real = in_paths[0] + real_prefix + epoch_str + '.png'
        img_str_fake = in_paths[0] + fake_prefix + epoch_str + '.png'
        img_str_real_to_fake = out_path_1 + t1_prefix + epoch_str + '.png'
        values_1.append(utils.evaluate_images(img_str_real, img_str_fake, img_str_real_to_fake))
        
        img_str_real = in_paths[1] + real_prefix + epoch_str + '.png'
        img_str_fake = in_paths[1] + fake_prefix + epoch_str + '.png'
        img_str_real_to_fake = out_path_2 + t2_prefix + epoch_str + '.png'
        values_2.append(utils.evaluate_images(img_str_real, img_str_fake, img_str_real_to_fake))

    values = [values_1, values_2]
    title = str('Training Evaluation of %s and %s' % (models[0], models[1]))
    graph_str_out = eval_folder + title + '.png'

    utils.plot_together(values, colors, models, title, graph_str_out)
    
def evaluate_testing(in_path, model, eval_folder, out_path, color):
    # Initialize prefix variables.
    out_path = eval_folder + out_path
    altr_prefix = '/low_res/'
    real_prefix = '/high_res_real/'
    fake_prefix = '/high_res_fake/'
    altr_to_fake_prefix = 'alt_to_fake_'
    real_to_fake_prefix = 'real_to_fake_'

    # Initialize lists.
    real_to_fake = []
    best_pairs = {}
    worst_pairs = {}

    # Clear output path images.
    utils.make_dir(eval_folder)
    utils.clear_dir(out_path)
    utils.clear_dir(out_path + 'sorted/')
    
    altr_filelist = glob.glob(in_path + altr_prefix + '*')
    real_filelist = glob.glob(in_path + real_prefix + '*')
    fake_filelist = glob.glob(in_path + fake_prefix + '*')

    # Start main loop.
    for i in range(len(real_filelist)):
        img_str_altr = altr_filelist[i]
        img_str_real = real_filelist[i]
        img_str_fake = fake_filelist[i]
    
        img_str_real_to_fake = out_path + real_to_fake_prefix + str(i) + '.png'
        difference = utils.evaluate_images(img_str_real, img_str_fake, img_str_real_to_fake)

        real_to_fake.append(difference)
        best_pairs[i] = difference
        worst_pairs[i] = difference

    title = str('Testing Evaluation of %s Model' % model)
    graph_str_out = eval_folder + title + '.png'
    utils.plot_loss_single(real_to_fake, color, 'Difference Ratio', title, graph_str_out)
    
    # Export the best and worst images
    sorted_pairs = sorted(worst_pairs.items(), key=lambda x: x[1])
    for i in range(10):
        worst_img = cv2.imread(altr_filelist[sorted_pairs[i][0]])
        best_img = cv2.imread(altr_filelist[sorted_pairs[len(sorted_pairs) - i - 1][0]])
        
        cv2.imwrite(out_path + 'sorted/worst_' + str(i) + '.png', worst_img)
        cv2.imwrite(out_path + 'sorted/best_' + str(i) + '.png', best_img)
    
    return real_to_fake

# Global references.
eval_path = 'evaluation/'
epochs = 100
dcgan_colors = ['cornflowerblue', 'lightseagreen', 'slateblue']
srgan_colors = ['orange', 'mediumvioletred', 'darkorchid']

# DCGAN and SRGAN training evaluations
evaluate_training(['dcgan/outputx32-90-3/', 'srgan/outputx32-90-3/'],
                  ['DCGAN-90-3', 'SRGAN-90-3'], epochs, eval_path,
                  ['dcgan/', 'srgan/'],
                  [dcgan_colors[0], srgan_colors[0]])
evaluate_training(['dcgan/outputx32-90-7/', 'srgan/outputx32-90-7/'],
                  ['DCGAN-90-7', 'SRGAN-90-7'], epochs, eval_path,
                  ['dcgan/', 'srgan/'],
                  [dcgan_colors[1], srgan_colors[1]])
evaluate_training(['dcgan/outputx32-75-7/', 'srgan/outputx32-75-7/'],
                  ['DCGAN-75-7', 'SRGAN-75-7'], epochs, eval_path,
                  ['dcgan/', 'srgan/'],
                  [dcgan_colors[2], srgan_colors[2]])


# SRGAN testing evaluations.
values = []
labels = ['90-3', '90-7', '75-7']
title = 'Testing Evaluation of SRGAN Model'
graph_str_out = eval_path + title + '.png'

values.append(evaluate_testing('srgan/outputx32-90-3/', 'SRGAN-90-3', eval_path, 'srgan/testing_90-3/', srgan_colors[0]))
values.append(evaluate_testing('srgan/outputx32-90-7/', 'SRGAN-90-7', eval_path, 'srgan/testing_90-7/', srgan_colors[1]))
values.append(evaluate_testing('srgan/outputx32-75-7/', 'SRGAN-75-7', eval_path, 'srgan/testing_75-7/', srgan_colors[2]))

# SRGAN final plot.
srgan_colors = ['gold', 'mediumvioletred', 'darkorchid']
utils.plot_loss_all(values, srgan_colors, labels, title, graph_str_out)

