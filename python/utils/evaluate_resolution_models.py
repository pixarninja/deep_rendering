# python evaluate_models.py

import cv2
import glob
import numpy as np
import os
import utils
import random

def evaluate_training(in_paths, model, niter, eval_folder, out_paths, colors, labels):
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
        img_str_real_to_fake = out_path_1 + t1_prefix + epoch_str
        values_1.append(utils.evaluate_images(img_str_real, img_str_fake, img_str_real_to_fake))
        
        img_str_real = in_paths[1] + real_prefix + epoch_str + '.png'
        img_str_fake = in_paths[1] + fake_prefix + epoch_str + '.png'
        img_str_real_to_fake = out_path_2 + t2_prefix + epoch_str
        values_2.append(utils.evaluate_images(img_str_real, img_str_fake, img_str_real_to_fake))

    values = [values_1, values_2]
    axes = ['Epoch', 'Difference Ratio']
    title = str('Training Evaluation of %s Model' % (model))
    graph_str_out = eval_folder + title + '.png'

    utils.plot_together(values, colors, labels, title, axes, graph_str_out)
    return values
    
def evaluate_testing(in_path, model, eval_folder, out_path, color):
    # Initialize prefix variables.
    out_path = eval_folder + out_path
    altr_prefix = '/low_res/'
    real_prefix = '/high_res_real/'
    fake_prefix = '/high_res_fake/'
    real_to_fake_prefix = 'real_to_fake_'

    # Initialize lists.
    real_to_fake = []

    # Clear output path images.
    utils.make_dir(eval_folder)
    utils.make_dir(out_path)
    
    altr_filelist = glob.glob(in_path + altr_prefix + '*')
    real_filelist = glob.glob(in_path + real_prefix + '*')
    fake_filelist = glob.glob(in_path + fake_prefix + '*')

    # Start main loop.
    for i in range(len(real_filelist)):
        img_str_altr = altr_filelist[i]
        img_str_real = real_filelist[i]
        img_str_fake = fake_filelist[i]
    
        img_str_real_to_fake = out_path + real_to_fake_prefix + str(i)
        difference = utils.evaluate_images(img_str_real, img_str_fake, img_str_real_to_fake)
        real_to_fake.append(difference)

    title = str('Testing Evaluation of %s Model' % model)
    graph_str_out = eval_folder + title + '.png'
    utils.plot_loss_single(real_to_fake, color, model, title, graph_str_out)
    
    return real_to_fake

def srgan_eval(eval_path, epochs, frame_colors, cifar_colors):
    # SRGAN training evaluations.
    model = 'SRGAN'
    values = []
    labels = []
    prefix = 'srgan/'
    utils.make_dir(eval_path + prefix)
    block_dim = 32

    alpha = 90
    beta = 3
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    labels.append('FRAME-{}'.format(tag))
    labels.append('CIFAR-{}'.format(tag))
    values.append(evaluate_training([prefix + 'frame_outputx' + tag + '/', prefix + 'cifar_outputx' + tag + '/'],
                      model + '-' + tag, epochs, eval_path,
                      [prefix + 'frame/', prefix + 'cifar/'],
                      [frame_colors[0], cifar_colors[0]],
                      ['FRAME', 'CIFAR']))

    alpha = 90
    beta = 7
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    labels.append('FRAME-{}'.format(tag))
    labels.append('CIFAR-{}'.format(tag))
    values.append(evaluate_training([prefix + 'frame_outputx' + tag + '/', prefix + 'cifar_outputx' + tag + '/'],
                      model + '-' + tag, epochs, eval_path,
                      [prefix + 'frame/', prefix + 'cifar/'],
                      [frame_colors[1], cifar_colors[1]],
                      ['FRAME', 'CIFAR']))

    alpha = 75
    beta = 7
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    labels.append('FRAME-{}'.format(tag))
    labels.append('CIFAR-{}'.format(tag))
    values.append(evaluate_training([prefix + 'frame_outputx' + tag + '/', prefix + 'cifar_outputx' + tag + '/'],
                      model + '-' + tag, epochs, eval_path,
                      [prefix + 'frame/', prefix + 'cifar/'],
                      [frame_colors[2], cifar_colors[2]],
                      ['FRAME', 'CIFAR']))

    tag = 'range'
    labels.append('FRAME-Random')
    labels.append('CIFAR-Random')
    values.append(evaluate_training([prefix + 'frame_outputx' + tag + '/', prefix + 'cifar_outputx' + tag + '/'],
                      model + '-Random', epochs, eval_path,
                      [prefix + 'frame/', prefix + 'cifar/'],
                      [frame_colors[3], cifar_colors[3]],
                      ['FRAME', 'CIFAR']))

    all_values = []
    for vals in values:
        for item in vals:
            all_values.append(item)

    all_colors = []
    for i in range(len(frame_colors)):
        all_colors.append(frame_colors[i])
        all_colors.append(cifar_colors[i])

    utils.plot_loss_all(all_values, all_colors, labels, 'Training Evaluation for SRGAN Model', eval_path + 'Training Evaluation SRGAN.png')

    # SRGAN testing evaluations.
    values = []
    labels = ['90-3', '90-7', '75-7']
    title = 'Testing Evaluation of SRGAN Model (FRAME)'
    label = model + '-FRAME-'
    graph_str_out = eval_path + title + '.png'

    alpha = 90
    beta = 3
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    values.append(evaluate_testing('{}frame_outputx{}/'.format(prefix, tag), label + '{}-{}'.format(alpha, beta), eval_path, 'srgan/frame_testing_{}-{}/'.format(alpha, beta), frame_colors[0]))
    alpha = 90
    beta = 7
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    values.append(evaluate_testing('{}frame_outputx{}/'.format(prefix, tag), label + '{}-{}'.format(alpha, beta), eval_path, 'srgan/frame_testing_{}-{}/'.format(alpha, beta), frame_colors[1]))
    alpha = 75
    beta = 7
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    values.append(evaluate_testing('{}frame_outputx{}/'.format(prefix, tag), label + '{}-{}'.format(alpha, beta), eval_path, 'srgan/frame_testing_{}-{}/'.format(alpha, beta), frame_colors[2]))
    tag = 'range'
    values.append(evaluate_testing('{}frame_outputx{}/'.format(prefix, tag), label + 'Random'.format(alpha, beta), eval_path, 'srgan/frame_testing_Random/', frame_colors[3]))

    # SRGAN-FRAME final plot.
    utils.plot_loss_all(values, frame_colors, labels, title, graph_str_out)

    values = []
    labels = ['90-3', '90-7', '75-7', 'Range']
    title = 'Testing Evaluation of SRGAN Model (CIFAR)'
    label = model + '-CIFAR-'
    graph_str_out = eval_path + title + '.png'

    alpha = 90
    beta = 3
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    values.append(evaluate_testing('{}cifar_outputx{}/'.format(prefix, tag), label + '{}-{}'.format(alpha, beta), eval_path, 'srgan/cifar_testing_{}-{}/'.format(alpha, beta), cifar_colors[0]))
    alpha = 90
    beta = 7
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    values.append(evaluate_testing('{}cifar_outputx{}/'.format(prefix, tag), label + '{}-{}'.format(alpha, beta), eval_path, 'srgan/cifar_testing_{}-{}/'.format(alpha, beta), cifar_colors[1]))
    alpha = 75
    beta = 7
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    values.append(evaluate_testing('{}cifar_outputx{}/'.format(prefix, tag), label + '{}-{}'.format(alpha, beta), eval_path, 'srgan/cifar_testing_{}-{}/'.format(alpha, beta), cifar_colors[2]))
    tag = 'range'
    values.append(evaluate_testing('{}cifar_outputx{}/'.format(prefix, tag), label + 'Random'.format(alpha, beta), eval_path, 'srgan/cifar_testing_Random/', cifar_colors[3]))

    # SRGAN-CIFAR final plot.
    utils.plot_loss_all(values, frame_colors, labels, title, graph_str_out)

def attention_eval(eval_path, epochs, frame_colors, cifar_colors):
    # Transformer training evaluations.
    model = 'Transformer'
    values = []
    labels = []
    prefix = 'sparse_attention/output/'
    outf_path = 'attention/'
    utils.make_dir(eval_path + outf_path)
    block_dim = 32

    alpha = 90
    beta = 3
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    labels.append('FRAME-{}'.format(tag))
    labels.append('CIFAR-{}'.format(tag))
    values.append(evaluate_training([prefix + 'frame_outputx' + tag + '/', prefix + 'cifar_outputx' + tag + '/'],
                      model + '-' + tag, epochs, eval_path,
                      [outf_path + 'frame/', outf_path + 'cifar/'],
                      [frame_colors[0], cifar_colors[0]],
                      ['FRAME', 'CIFAR']))

    alpha = 90
    beta = 7
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    labels.append('FRAME-{}'.format(tag))
    labels.append('CIFAR-{}'.format(tag))
    values.append(evaluate_training([prefix + 'frame_outputx' + tag + '/', prefix + 'cifar_outputx' + tag + '/'],
                      model + '-' + tag, epochs, eval_path,
                      [outf_path + 'frame/', outf_path + 'cifar/'],
                      [frame_colors[1], cifar_colors[1]],
                      ['FRAME', 'CIFAR']))

    alpha = 75
    beta = 7
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    labels.append('FRAME-{}'.format(tag))
    labels.append('CIFAR-{}'.format(tag))
    values.append(evaluate_training([prefix + 'frame_outputx' + tag + '/', prefix + 'cifar_outputx' + tag + '/'],
                      model + '-' + tag, epochs, eval_path,
                      [outf_path + 'frame/', outf_path + 'cifar/'],
                      [frame_colors[2], cifar_colors[2]],
                      ['FRAME', 'CIFAR']))

    tag = 'range'
    labels.append('FRAME-Random')
    labels.append('CIFAR-Random')
    values.append(evaluate_training([prefix + 'frame_outputx' + tag + '/', prefix + 'cifar_outputx' + tag + '/'],
                      model + '-Random', epochs, eval_path,
                      [outf_path + 'frame/', outf_path + 'cifar/'],
                      [frame_colors[3], cifar_colors[3]],
                      ['FRAME', 'CIFAR']))

    all_values = []
    for vals in values:
        for item in vals:
            all_values.append(item)

    all_colors = []
    for i in range(len(frame_colors)):
        all_colors.append(frame_colors[i])
        all_colors.append(cifar_colors[i])

    utils.plot_loss_all(all_values, all_colors, labels, 'Training Evaluation for Tranformer Model', eval_path + 'Training Evaluation Transformer.png')

    # Transformer testing evaluations.
    values = []
    labels = ['90-3', '90-7', '75-7', 'Range']
    title = 'Testing Evaluation of Transformer Model (FRAME)'
    label = model + '-FRAME-'
    graph_str_out = eval_path + title + '.png'

    alpha = 90
    beta = 3
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    values.append(evaluate_testing('{}frame_outputx{}/'.format(prefix, tag), label + '{}-{}'.format(alpha, beta), eval_path, 'attention/frame_testing_{}-{}/'.format(alpha, beta), frame_colors[0]))
    alpha = 90
    beta = 7
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    values.append(evaluate_testing('{}frame_outputx{}/'.format(prefix, tag), label + '{}-{}'.format(alpha, beta), eval_path, 'attention/frame_testing_{}-{}/'.format(alpha, beta), frame_colors[1]))
    alpha = 75
    beta = 7
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    values.append(evaluate_testing('{}frame_outputx{}/'.format(prefix, tag), label + '{}-{}'.format(alpha, beta), eval_path, 'attention/frame_testing_{}-{}/'.format(alpha, beta), frame_colors[2]))
    tag = 'range'
    values.append(evaluate_testing('{}frame_outputx{}/'.format(prefix, tag), label + 'Random'.format(alpha, beta), eval_path, 'attention/frame_testing_Random/', frame_colors[3]))

    # Transformer-FRAME final plot.
    utils.plot_loss_all(values, frame_colors, labels, title, graph_str_out)

    values = []
    labels = ['90-3', '90-7', '75-7']
    title = 'Testing Evaluation of Transformer Model (CIFAR)'
    label = model + '-CIFAR-'
    graph_str_out = eval_path + title + '.png'

    alpha = 90
    beta = 3
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    values.append(evaluate_testing('{}cifar_outputx{}/'.format(prefix, tag), label + '{}-{}'.format(alpha, beta), eval_path, 'attention/cifar_testing_{}-{}/'.format(alpha, beta), cifar_colors[0]))
    alpha = 90
    beta = 7
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    values.append(evaluate_testing('{}cifar_outputx{}/'.format(prefix, tag), label + '{}-{}'.format(alpha, beta), eval_path, 'attention/cifar_testing_{}-{}/'.format(alpha, beta), cifar_colors[1]))
    alpha = 75
    beta = 7
    tag = '%02d-%02d-%d' % ( block_dim, alpha, beta )
    values.append(evaluate_testing('{}cifar_outputx{}/'.format(prefix, tag), label + '{}-{}'.format(alpha, beta), eval_path, 'attention/cifar_testing_{}-{}/'.format(alpha, beta), cifar_colors[2]))
    tag = 'range'
    values.append(evaluate_testing('{}cifar_outputx{}/'.format(prefix, tag), label + 'Random'.format(alpha, beta), eval_path, 'attention/cifar_testing_Random/', cifar_colors[3]))

    # Transformer-CIFAR final plot.
    utils.plot_loss_all(values, frame_colors, labels, title, graph_str_out)

# Global references.
eval_path = 'evaluation/'
epochs = 95
frame_colors = ['orange', 'chocolate', 'crimson', 'darkmagenta']
cifar_colors = ['seagreen', 'mediumaquamarine', 'teal', 'darkslateblue']

srgan_eval(eval_path, epochs, frame_colors, cifar_colors)
attention_eval(eval_path, epochs, frame_colors, cifar_colors)
