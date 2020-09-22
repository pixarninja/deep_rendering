import argparse
import csv
import cv2 as cv2
import numpy as np
import os as os
import pandas as pd
import re as re
from utils import plot_together

def graph_loss(values, title, eval_folder, axes, colors, labels, legend):
    graph_str_out = eval_folder + title + '.png'
    plot_together(values, colors, labels, title, axes, graph_str_out, legend)
    print('Output to {}'.format(graph_str_out))

def evaluate_pretraining():
    # Initalize application
    eval_folder = '../attngan/output/'
    paths = ['birds/birds_100_250/pre_train.csv', 'frame/frame_emb256_cap1/pre_train.csv', 'frame/frame_emb256_subdiv/pre_train.csv']
    data = []

    for path in paths:
        with open(eval_folder + path) as f:
            reader = csv.reader(f)
            data.append( list(reader) )

    # Collect loss data
    values = []
    for i in range(len(data)):
        for loss_data in data[i]:
            losses = []
            for loss in loss_data:
                losses.append( re.sub('[^0-9]', '', loss) )
            avg_loss = np.average( np.array([float(losses[1]), float(losses[2])], dtype=float) )

            if len(values) <= i:
                values.append([])
            values[i].append(avg_loss)

    for i in range(0, len(values)):
        values[i] = values[i][0:251]
    
    # print(np.average(np.array(values[1])))
    # print(np.average(np.array(values[2])))

    # Normalize losses
    max = np.matrix((values)).max()
    for i in range(len(values)):
        values[i] = (np.array(values[i]) / float(max)).tolist()

    colors = ['darkslategrey', 'mediumvioletred', 'seagreen'] # ['seagreen', 'mediumaquamarine', 'teal', 'darkslateblue']
    frame_labels = ['Birds Pretraining', 'Frame Pretraining Initial', 'Frame Pretraining Subdiv']
    out_path = '../eval/generation/'
    axes = ['Epoch', 'Average Normalized Loss']
    graph_loss(values, 'Evaluation of Pretraining Models', out_path, axes, colors, frame_labels, 'lower right')

def evaluate_gentraining():
    # Initalize application
    eval_folder = '../attngan/output/'
    paths = ['birds/birds_100_250/gen_train.csv', 'frame/frame_emb256_cap1/gen_train.csv', 'frame/frame_emb256_subdiv/gen_train.csv']
    data = []

    for path in paths:
        with open(eval_folder + path) as f:
            reader = csv.reader(f)
            data.append( list(reader) )

    # Collect loss data for both generator and discriminator
    values = []
    for i in range(0, len(data) * 2, 2):
        for loss_data in data[int(i / 2)]:
            losses = []
            for loss in loss_data:
                losses.append( re.sub('[^0-9]', '', loss) )

            if len(values) <= i:
                values.append([])
            values[i].append(float(losses[1]))
            if len(values) <= i + 1:
                values.append([])
            values[i + 1].append(float(losses[2]))

    for i in range(0, len(values)):
        values[i] = values[i][0:251]

    # Normalize losses
    max_D = np.matrix(values[0] + values[2] + values[4]).max()
    max_G = np.matrix(values[1] + values[3] + values[5]).max()
    for i in range(0, len(values), 2):
        values[i] = (np.array(values[i]) / float(max_D)).tolist()
        values[i + 1] = (np.array(values[i + 1]) / float(max_G)).tolist()

    # print(np.average(np.array(values[1])))
    # print(np.average(np.array(values[3])))

    colors = ['darkslategrey', 'cornflowerblue', 'red', 'darkmagenta', 'orange', 'teal'] # ['seagreen', 'mediumaquamarine', 'teal', 'darkslateblue']
    labels = ['Birds D', 'Birds G', 'Initial D', 'Initial G', 'Subdiv D', 'Subdiv G']
    out_path = '../eval/generation/'
    axes = ['Epoch', 'Normalized Loss']
    graph_loss(values, 'Evaluation of Generation Models', out_path, axes, colors, labels, 'upper right')

    # Collect loss data for both generator and discriminator
    values = []
    for i in range(0, len(data)):# * 2, 2):
        for loss_data in data[int(i)]:# / 2)]:
            losses = []
            for loss in loss_data:
                losses.append( re.sub('[^0-9]', '', loss) )
            avg_loss = np.average( np.array([float(losses[1]), float(losses[2])], dtype=float) )

            if len(values) <= i:
                values.append([])
            values[i].append(avg_loss)

    for i in range(0, len(values)):
        values[i] = values[i][0:251]

    # Normalize losses
    max = np.matrix((values)).max()
    for i in range(len(values)):
        values[i] = (np.array(values[i]) / float(max)).tolist()

    colors = ['darkslategrey', 'mediumvioletred', 'seagreen'] # ['seagreen', 'mediumaquamarine', 'teal', 'darkslateblue']
    labels = ['Birds Pretraining', 'Frame Pretraining Simple', 'Frame Pretraining Subdiv']
    out_path = '../eval/generation/'
    axes = ['Epoch', 'Normalized Loss']
    graph_loss(values, 'Averaged Loss of Generation Models', out_path, axes, colors, labels, 'upper right')

def image_from_blocks():
    dim = int(opt.blockDim)
    captions = int(opt.captions)
    x_res = int(opt.xRes)
    y_res = int(opt.yRes)
    x_div = int(x_res / dim)
    y_div = int(y_res / dim)

    # Initialize file variables
    f_original = opt.targetFrame
    eval_folder = '../attngan/output/frame/{}/gen_train/{}/'.format(opt.inputSpecifier, opt.inputPrefix)

    # Iterate over all captions
    for c in range(0, captions):
        avg = []
        img_original = cv2.imread(f_original)[0 : y_res, 0 : x_res]
        img_out = np.full((y_res, x_res * 2, 3), 0, dtype=int)
        index = 1

        # Iterate over image space
        for row in range(0, y_div):
            data = []
            for col in range(0, x_div):
                # Find input index and store image
                index = row * x_div + col + 1
                f_in = eval_folder + '{}/0_s_{:d}_g1.png'.format(index, c)
                if not f_in or not os.path.exists(f_in):
                    data.append([0, 0, 0])
                    continue

                # Place image in output
                img_in = cv2.resize(cv2.imread(f_in), (dim, dim))
                x_offset = col * dim
                y_offset = row * dim
                img_out[y_offset : (y_offset + dim), x_offset : (x_offset + dim)] = img_in
                data.append(img_in.mean(axis=0).mean(axis=0))
            avg.append(data)

        # Write full-resolution images
        img_out[0 : y_res, x_res : x_res * 2] = img_original
        img_out = cv2.resize(np.array(img_out, dtype='uint8'), (x_res * 2, y_res))
        f_out = eval_folder + '{}_frame_full.png'.format(c)
        cv2.imwrite(f_out, img_out)
        print('Wrote image: ' + f_out)

        # # Write half-resolution images
        # img_out = cv2.resize(cv2.imread(f_out), (int(x_res), int(y_res / 2)))
        # img_out = cv2.resize(np.array(img_out, dtype='uint8'), (x_res * 2, y_res))
        # f_out = eval_folder + '{}_frame_half.png'.format(c)
        # cv2.imwrite(f_out, img_out)
        # print('Wrote image: ' + f_out)

        # # Write tenth-resolution images
        # img_out = cv2.resize(cv2.imread(f_out), (int(x_res / 5), int(y_res / 10)))
        # img_out = cv2.resize(np.array(img_out, dtype='uint8'), (x_res * 2, y_res))
        # f_out = eval_folder + '{}_frame_tenth.png'.format(c)
        # cv2.imwrite(f_out, img_out)
        # print('Wrote image: ' + f_out)

        # # Write fortieth-resolution images
        # img_out = cv2.resize(cv2.imread(f_out), (int(x_res / 20), int(y_res / 40)))
        # img_out = cv2.resize(np.array(img_out, dtype='uint8'), (x_res * 2, y_res))
        # f_out = eval_folder + '{}_frame_fortieth.png'.format(c)
        # cv2.imwrite(f_out, img_out)
        # print('Wrote image: ' + f_out)

        # # Write min-resolution images
        # img_out = cv2.resize(cv2.imread(f_out), (x_div, y_div))
        # img_out = cv2.resize(np.array(img_out, dtype='uint8'), (x_res * 2, y_res))
        # f_out = eval_folder + '{}_frame_minimum.png'.format(c)
        # cv2.imwrite(f_out, img_out)
        # print('Wrote image: ' + f_out)

        # Write average-resolution images
        f_out = eval_folder + '{}_frame_avg.png'.format(c)
        img_out = np.full((y_div, x_div * 2, 3), 0, dtype=int)
        img_in = np.array(avg, dtype='uint8')
        img_out[0 : y_div, 0 : x_div] = img_in
        img_original = cv2.resize(cv2.imread(f_original), (x_div, y_div))
        img_out[0 : y_div * dim, x_div : (x_div * 2)] = img_original
        img_out = cv2.resize(np.array(img_out, dtype='uint8'), (x_res * 2, y_res))
        cv2.imwrite(f_out, img_out)
        print('Wrote image: ' + f_out)

def run_evaluations():
    #evaluate_pretraining()
    #evaluate_gentraining()
    image_from_blocks()

parser = argparse.ArgumentParser()
parser.add_argument('--inputSpecifier', default='frame_16_all', help='input GAN path specifier')
parser.add_argument('--inputPrefix', default='netG_epoch_50', help='input GAN path prefix')
parser.add_argument('--targetFrame', default='../datasets/Frame/images/003.jpg', help='target frame source')
parser.add_argument('--blockDim', type=int, default=4, help='dimension of frameblocks')
parser.add_argument('--captions', type=int, default=1, help='number of captions to output')
parser.add_argument('--xRes', type=int, default=1920, help='resolution of target image in the x dimension')
parser.add_argument('--yRes', type=int, default=1080, help='resolution of target image in the y dimension')

opt = parser.parse_args()
print(opt)

run_evaluations()