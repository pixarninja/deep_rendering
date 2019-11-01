# Taken from: https://github.com/aitorzip/PyTorch-SRGAN
# python train.py --blockDim 32 --alpha 0.9 --beta 3 --cuda

import argparse
import cv2 as cv2
import numpy as np
import os as os
import sys as sys
import utils as utils

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser()
parser.add_argument('--blockDim', type=int, default=64, help='size of block to use')
parser.add_argument('--alpha', type=float, default=0.75, help='noise contant to use')
parser.add_argument('--beta', type=int, default=7, help='blur contant to use')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--generation', type=int, default=100, help='epochs to wait between writing images')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--upSampling', type=int, default=1, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--generatorWeights', type=str, default='', help="path to generator weights (to continue training)")
parser.add_argument('--discriminatorWeights', type=str, default='', help="path to discriminator weights (to continue training)")

opt = parser.parse_args()
print(opt)

outc_path = ('checkpointsx%d-%d-%d/' % (opt.blockDim, int(opt.alpha * 100), opt.beta))
outf_path = ('outputx%d-%d-%d/' % (opt.blockDim, int(opt.alpha * 100), opt.beta))
utils.clear_dir(outc_path)
utils.clear_dir(outf_path)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transform = transforms.Compose([transforms.RandomCrop(opt.blockDim),
                                transforms.ToTensor()])

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

# Replace loader with hardcoded values.
data_prefix = 'C:/Users/wesha/Git/dynamic_frame_generator/python/training/' + str(opt.blockDim) + '/'
dataset = datasets.ImageFolder(root=data_prefix + 'validation/', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

nc=3

device = torch.device("cuda:0" if opt.cuda else "cpu")
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x

class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return swish(self.shuffler(self.conv(x)))

class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

if __name__ == '__main__':
    generator = Generator(16, 1)
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

        for i, data in enumerate(dataloader, 0):
            # Generate data
            high_res_real = data[0]
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

            ######### Train generator #########
            generator.zero_grad()

            generator_content_loss = content_criterion(high_res_fake, high_res_real)
            mean_generator_content_loss += generator_content_loss.data

            generator_content_loss.backward()
            optim_generator.step()

            ######### Status and display #########
            sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f' % (epoch + 1, 2, i, len(dataloader), generator_content_loss.data))
            if i % opt.generation == 0:
                vutils.save_image(low_res,
                        '%slow_res.png' % outf_path,
                        normalize=True)
                vutils.save_image(high_res_real,
                        '%shigh_res_real.png' % outf_path,
                        normalize=True)
                vutils.save_image(high_res_fake,
                        '%shigh_res_fake.png' % outf_path,
                        normalize=True)

        sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f\n' % (epoch + 1, 2, i, len(dataloader), mean_generator_content_loss/len(dataloader)))
        log_value('generator_mse_loss', mean_generator_content_loss/len(dataloader), epoch)

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

        for i, data in enumerate(dataloader):
            # Generate data
            high_res_real, _ = data
            print('... ' + str(np.shape(high_res_real)))
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
                target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7).cuda()
                target_fake = Variable(torch.rand(opt.batchSize,1)*0.3).cuda()
            else:
                high_res_real = Variable(high_res_real)
                high_res_fake = generator(Variable(low_res))
                target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7)
                target_fake = Variable(torch.rand(opt.batchSize,1)*0.3)
            
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
            sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (epoch + 1, opt.nEpochs, i, len(dataloader),
            discriminator_loss.data, generator_content_loss.data, generator_adversarial_loss.data, generator_total_loss.data))
            if i % opt.generation == 0:
                vutils.save_image(low_res,
                        '%slow_res.png' % outf_path,
                        normalize=True)
                vutils.save_image(high_res_real,
                        '%shigh_res_real.png' % outf_path,
                        normalize=True)
                vutils.save_image(high_res_fake,
                        '%shigh_res_fake.png' % outf_path,
                        normalize=True)

        sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (epoch + 1, opt.nEpochs, i, len(dataloader),
        mean_discriminator_loss/len(dataloader), mean_generator_content_loss/len(dataloader), 
        mean_generator_adversarial_loss/len(dataloader), mean_generator_total_loss/len(dataloader)))

        log_value('generator_content_loss', mean_generator_content_loss/len(dataloader), epoch)
        log_value('generator_adversarial_loss', mean_generator_adversarial_loss/len(dataloader), epoch)
        log_value('generator_total_loss', mean_generator_total_loss/len(dataloader), epoch)
        log_value('discriminator_loss', mean_discriminator_loss/len(dataloader), epoch)

        # Do checkpointing
        torch.save(generator.state_dict(), '%s/generator_final.pth' % outc_path)
        torch.save(discriminator.state_dict(), '%s/discriminator_final.pth' % outc_path)
