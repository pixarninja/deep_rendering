# Taken from: https://github.com/aitorzip/PyTorch-SRGAN
# python test.py --blockDim 32 --alpha 0.9 --beta 3 --cuda

import argparse
import os as os
import numpy as np
import sys as sys
import utils as utils

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models import Generator, Discriminator, FeatureExtractor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blockDim', type=int, default=64, help='size of block to use')
    parser.add_argument('--alpha', type=float, default=0.75, help='noise contant to use')
    parser.add_argument('--beta', type=int, default=7, help='blur contant to use')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--upSampling', type=int, default=1, help='low to high resolution scaling factor')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--generatorWeights', type=str, default='generator_final.pth', help="path to generator weights (to continue training)")
    parser.add_argument('--discriminatorWeights', type=str, default='discriminator_final.pth', help="path to discriminator weights (to continue training)")

    opt = parser.parse_args()
    print(opt)
    
    inc_path = ('checkpointsx%d-%d-%d/' % (opt.blockDim, int(opt.alpha * 100), opt.beta))
    outf_path = ('outputx%d-%d-%d/' % (opt.blockDim, int(opt.alpha * 100), opt.beta))
    if not os.path.exists(inc_path):
        print('Error: input checkpoint path %s does not exist. First generate the files using train.py.' % inc_path)
        exit()
    
    utils.clear_dir(outf_path + 'high_res_fake/')
    utils.clear_dir(outf_path + 'high_res_real/')
    utils.clear_dir(outf_path + 'low_res/')

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    transform = transforms.Compose([transforms.RandomCrop(opt.blockDim),
                                    transforms.ToTensor()])

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])

    # Equivalent to un-normalizing ImageNet (for correct visualization)
    unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

    # Replace loader with hardcoded values.
    data_prefix = 'C:/Users/wesha/Git/dynamic_frame_generator/python/training/' + str(opt.blockDim) + '/'
    dataset = datasets.ImageFolder(root=data_prefix + 'testset/', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=int(opt.workers))

    generator = Generator(16, opt.upSampling)
    if opt.generatorWeights != '':
        generator.load_state_dict(torch.load(inc_path + opt.generatorWeights))
    print(generator)

    discriminator = Discriminator()
    if opt.discriminatorWeights != '':
        discriminator.load_state_dict(torch.load(inc_path + opt.discriminatorWeights))
    print(discriminator)

    # For the content loss
    feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
    print(feature_extractor)
    content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()

    target_real = Variable(torch.ones(opt.batchSize,1))
    target_fake = Variable(torch.zeros(opt.batchSize,1))

    # if gpu is to be used
    if opt.cuda:
        generator.cuda()
        discriminator.cuda()
        feature_extractor.cuda()
        content_criterion.cuda()
        adversarial_criterion.cuda()
        target_real = target_real.cuda()
        target_fake = target_fake.cuda()

    low_res = torch.FloatTensor(opt.batchSize, 3, opt.blockDim, opt.blockDim)

    print('Test started...')
    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0

    # Set evaluation mode (not training)
    generator.eval()
    discriminator.eval()

    for i, data in enumerate(dataloader):
        # Generate data
        high_res_real = data[0]
        print(' ... ' + str(np.shape(high_res_real)))
        if np.shape(high_res_real)[0] != opt.batchSize:
            continue

        # Downsample images to low resolution
        for j in range(opt.batchSize):
            low_res[j] = utils.alter_image(high_res_real[j].numpy().transpose(1, 2, 0), opt.alpha, opt.beta)
            high_res_real[j] = normalize(high_res_real[j])

        # Generate real and fake inputs
        if opt.cuda:
            high_res_real = Variable(high_res_real.cuda())
            high_res_fake = generator(Variable(low_res).cuda())
        else:
            high_res_real = Variable(high_res_real)
            high_res_fake = generator(Variable(low_res))
        
        ######### Test discriminator #########

        discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                                adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
        mean_discriminator_loss += discriminator_loss.data

        ######### Test generator #########

        real_features = Variable(feature_extractor(high_res_real).data)
        fake_features = feature_extractor(high_res_fake)

        generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
        mean_generator_content_loss += generator_content_loss.data
        generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), target_real)
        mean_generator_adversarial_loss += generator_adversarial_loss.data

        generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss
        mean_generator_total_loss += generator_total_loss.data

        ######### Status and display #########
        sys.stdout.write('\r[%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (i, len(dataloader),
        discriminator_loss.data, generator_content_loss.data, generator_adversarial_loss.data, generator_total_loss.data))

        for j in range(opt.batchSize):
            vutils.save_image(unnormalize(high_res_fake[j]),
                    '%shigh_res_fake/%d.png' % (outf_path, i*opt.batchSize + j),
                    normalize=False)
            vutils.save_image(unnormalize(high_res_real[j]),
                    '%shigh_res_real/%d.png' % (outf_path, i*opt.batchSize + j),
                    normalize=False)
            vutils.save_image(low_res[j],
                    '%slow_res/%d.png' % (outf_path, i*opt.batchSize + j),
                    normalize=False)

    sys.stdout.write('\r[%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (i, len(dataloader),
    mean_discriminator_loss/len(dataloader), mean_generator_content_loss/len(dataloader), 
    mean_generator_adversarial_loss/len(dataloader), mean_generator_total_loss/len(dataloader)))