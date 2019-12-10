# Taken from: https://github.com/aitorzip/PyTorch-SRGAN
# python train.py --blockDim 32 --alpha 0.9 --beta 3 --cuda

import argparse
import cv2 as cv2
import numpy as np
import os as os
import sys as sys
import utils as utils
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from models import Generator, Discriminator, FeatureExtractor
from tensorboard_logger import configure, log_value

from keras.datasets import cifar10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blockDim', type=int, default=64, help='size of block to use')
    parser.add_argument('--useRange', action='store_true', help='whether or not to use a random range')
    parser.add_argument('--alpha', type=float, default=0.75, help='noise constant to use')
    parser.add_argument('--beta', type=int, default=7, help='blur constant to use')
    parser.add_argument('--alphaPair', type=float, default=0.9, help='noise constant range to use')
    parser.add_argument('--betaPair', type=int, default=3, help='blur constant range to use')
    parser.add_argument('--generation', type=int, default=100, help='epochs to wait between writing images')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--upSampling', type=int, default=1, help='low to high resolution scaling factor')
    parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
    parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--generatorWeights', type=str, default='', help="path to generator weights (to continue training)")
    parser.add_argument('--discriminatorWeights', type=str, default='', help="path to discriminator weights (to continue training)")
    parser.add_argument('--inType', default='frame', help='input type, one of the following: [frame, cifar]')

    opt = parser.parse_args()
    print(opt)
    
    pair = None
    tag = '%d-%d-%d' % (opt.blockDim, int(opt.alpha * 100), opt.beta)
    if opt.useRange:
        opt.alpha = 0.75
        opt.beta = 7
        pair = [opt.alphaPair, opt.betaPair]
        tag = 'range'
    outc_path = ('%s_checkpointsx%s/' % (opt.inType, tag))
    outf_path = ('%s_outputx%s/' % (opt.inType, tag))

    utils.clear_dir(outc_path)
    utils.clear_dir(outf_path)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Load data.
    if opt.inType == 'frame':
        transform = transforms.Compose([transforms.RandomCrop(opt.blockDim), transforms.ToTensor()])
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

        data_prefix = '/home/pixarninja/Git/dynamic_frame_generator/python/training/' + str(opt.blockDim) + '/'
        dataset = datasets.ImageFolder(root=data_prefix + 'validation/', transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

        # Generate training data.
        x_train = []
        y_train = []
        for i, data in enumerate(dataloader, 0):
            high_res_real = data[0]
            if np.shape(high_res_real)[0] != opt.batchSize:
                continue

            # Downsample images to low resolution.
            for j in range(opt.batchSize):
                x_train.append(utils.alter_image(high_res_real[j].numpy().transpose(1, 2, 0), opt.alpha, opt.beta, pair=pair))
                y_train.append(normalize(high_res_real[j]))

        x_train = torch.stack(x_train)
        y_train = torch.stack(y_train)

    elif opt.inType == 'cifar':
        (x_data, _), _ = cifar10.load_data()

        # Generate training data.
        x_train = []
        y_train = []
        for i, data in enumerate(x_data, 0):
            data = data / 255.

            # Downsample image to low resolution.
            x_train.append(utils.alter_image(np.array(data), opt.alpha, opt.beta, pair=pair))
            y_train.append(torch.from_numpy(data.transpose(2, 0, 1)))

        x_train = torch.stack(x_train)
        y_train = torch.stack(y_train)

    else:
        print('ERROR: Input data type not recognized')
        exit(1)

    # Plot 25 sample images.
    plt.figure(figsize=(10,10))
    for i in range(25):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(np.asarray(x_train[i]).transpose(1, 2, 0))
            plt.title(str(i))
            plt.axis('off')
    plt.savefig('./{}_samplesx{}.png'.format(opt.inType, tag))
    plt.close('all')

    n_samples = int(x_train.shape[0] / opt.batchSize)
    print('\nBatch Size: {}, Batches: {}'.format(opt.batchSize, n_samples))

    generator = Generator(16, opt.upSampling)
    if opt.generatorWeights != '':
        generator.load_state_dict(torch.load(opt.generatorWeights))
    print(generator)

    discriminator = Discriminator()
    if opt.discriminatorWeights != '':
        discriminator.load_state_dict(torch.load(opt.discriminatorWeights))
    print(discriminator)

    # For the content loss
    feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
    print(feature_extractor)
    content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()

    ones_const = Variable(torch.ones(opt.batchSize, 1))

    # if gpu is to be used
    if opt.cuda:
        generator.cuda()
        discriminator.cuda()
        feature_extractor.cuda()
        content_criterion.cuda()
        adversarial_criterion.cuda()
        ones_const = ones_const.cuda()

    optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)

    configure('logs/' + '-' + str(opt.batchSize) + '-' + str(opt.generatorLR) + '-' + str(opt.discriminatorLR), flush_secs=5)

    low_res = torch.FloatTensor(opt.batchSize, 3, opt.blockDim, opt.blockDim)

    # Pre-train generator using raw MSE loss
    print('Generator pre-training')
    for epoch in range(2):
        mean_generator_content_loss = 0.0

        for i in range(n_samples):
            low_res = x_train[i * opt.batchSize:(i + 1) * opt.batchSize]
            high_res_real = y_train[i * opt.batchSize:(i + 1) * opt.batchSize]

            # Generate real and fake inputs
            if opt.cuda:
                high_res_real = Variable(high_res_real.cuda())
                high_res_fake = generator(Variable(low_res).cuda())
            else:
                high_res_real = Variable(high_res_real)
                high_res_fake = generator(Variable(low_res))

            high_res_real = high_res_real.float()
            high_res_fake = high_res_fake.float()

            ######### Train generator #########
            generator.zero_grad()

            generator_content_loss = content_criterion(high_res_fake, high_res_real)
            mean_generator_content_loss += generator_content_loss.data

            generator_content_loss.backward()
            optim_generator.step()

            ######### Status and display #########
            sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f' % (epoch + 1, 2, i, n_samples, generator_content_loss.data))

        sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f\n' % (epoch + 1, 2, i, n_samples, mean_generator_content_loss / n_samples))
        log_value('generator_mse_loss', mean_generator_content_loss / n_samples, epoch)

    # Do checkpointing
    torch.save(generator.state_dict(), '%s/generator_pretrain.pth' % outc_path)

    # SRGAN training
    optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR*0.1)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR*0.1)

    print('SRGAN training')
    for epoch in range(opt.nEpochs):
        mean_generator_content_loss = 0.0
        mean_generator_adversarial_loss = 0.0
        mean_generator_total_loss = 0.0
        mean_discriminator_loss = 0.0

        for i in range(n_samples):
            low_res = x_train[i * opt.batchSize:(i + 1) * opt.batchSize]
            high_res_real = y_train[i * opt.batchSize:(i + 1) * opt.batchSize]

            # Generate real and fake inputs
            if opt.cuda:
                high_res_real = Variable(high_res_real.cuda())
                high_res_fake = generator(Variable(low_res).cuda())
                target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7).cuda()
                target_fake = Variable(torch.rand(opt.batchSize,1)*0.3).cuda()
            else:
                high_res_real = Variable(high_res_real)
                high_res_fake = generator(Variable(low_res))
                target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7)
                target_fake = Variable(torch.rand(opt.batchSize,1)*0.3)

            high_res_real = high_res_real.float()
            high_res_fake = high_res_fake.float()

            ######### Train discriminator #########
            discriminator.zero_grad()

            discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                                 adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
            mean_discriminator_loss += discriminator_loss.data
            
            discriminator_loss.backward()
            optim_discriminator.step()

            ######### Train generator #########
            generator.zero_grad()

            real_features = Variable(feature_extractor(high_res_real).data)
            fake_features = feature_extractor(high_res_fake)

            generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
            mean_generator_content_loss += generator_content_loss.data
            generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), ones_const)
            mean_generator_adversarial_loss += generator_adversarial_loss.data

            generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss
            mean_generator_total_loss += generator_total_loss.data
            
            generator_total_loss.backward()
            optim_generator.step()   
            
            ######### Status and display #########
            sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (epoch + 1, opt.nEpochs, i, n_samples,
            discriminator_loss.data, generator_content_loss.data, generator_adversarial_loss.data, generator_total_loss.data))
            if i == n_samples - 1:
                vutils.save_image(low_res,
                        '%s/alt_%03d.png' % (outf_path, epoch),
                        normalize=True)
                vutils.save_image(high_res_real,
                        '%s/real_%03d.png' % (outf_path, epoch),
                        normalize=True)
                vutils.save_image(high_res_fake,
                        '%s/fake_%03d.png' % (outf_path, epoch),
                        normalize=True)

        sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (epoch + 1, opt.nEpochs, i, n_samples,
        mean_discriminator_loss / n_samples, mean_generator_content_loss / n_samples, 
        mean_generator_adversarial_loss / n_samples, mean_generator_total_loss / n_samples))

        log_value('generator_content_loss', mean_generator_content_loss / n_samples, epoch)
        log_value('generator_adversarial_loss', mean_generator_adversarial_loss / n_samples, epoch)
        log_value('generator_total_loss', mean_generator_total_loss / n_samples, epoch)
        log_value('discriminator_loss', mean_discriminator_loss / n_samples, epoch)

        # Do checkpointing
        torch.save(generator.state_dict(), '%s/generator_final.pth' % outc_path)
        torch.save(discriminator.state_dict(), '%s/discriminator_final.pth' % outc_path)
