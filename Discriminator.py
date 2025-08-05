import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from SPPmodel import SPPLayer

class ConvLayer(nn.Module):  # inchannels->conv2d->dropout->relu->outchannels
    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True, bn=True, dropout=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = dropout
        self.Dropout = nn.Dropout2d(p=0.5)
        self.relu = relu
        self.bn = bn
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.batch_norm = nn.BatchNorm2d(self.out_channels, eps=1e-05, momentum=0.1, affine=True,
        #                                  track_running_stats=True)
        self.batch_norm = nn.InstanceNorm2d(self.out_channels,affine=True)      #wgan说是用BN效果不好
        self.batch_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.bn is True:
            out = self.batch_norm(out)
        if self.relu is True:
            out = self.batch_relu(out)
        if self.dropout is True:
            out = self.Dropout(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #self.img = img
        # H = self.img.shape[2]
        # W = self.img.shape[2]

        self.discriminator = nn.Sequential(
            ConvLayer(1, out_channels=16, kernel_size=3, stride=1, relu=True, bn=True, dropout=False),
            ConvLayer(in_channels=16, out_channels=32, kernel_size=3, stride=1, relu=True, bn=True, dropout=False),
            ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, relu=True, bn=True, dropout=False),
            ConvLayer(in_channels=64, out_channels=1, kernel_size=3, stride=1, relu=True, bn=True, dropout=False),
        )
        self.spp = SPPLayer([4, 2, 1], pool_type='max_pool')
        self.fc1 = nn.Linear(21, 1)     # 在这一块，由于我最后把channel合并成1了，所以经过三个尺度的pooling得到了16+4+1=21，如果最后channel增大了，就改为c*21
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        out = self.discriminator(img)  # (16,1,H,W)
        out = self.spp(out)     #(16,21)
        # out = out.view(out.size(0), -1)  # (16,H*W)
        score = self.fc1(out)  # (16,1)
        # score = self.sigmoid(score)         #wgan要去掉sigmoid
        return score


class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()

        self.conv = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride= 1,padding=1),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(16, 32,  kernel_size=3, stride=1,padding=1),
                nn.InstanceNorm2d(32),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1,padding=1),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64,  kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2)
            )
        )
        self.spp = SPPLayer([8, 4, 1], pool_type='max_pool')
        self.fc = nn.Linear(64*81, 1)     # 在这一块，由于我最后把channel合并成1了，所以经过三个尺度的pooling得到了16+4+1=21，如果最后channel增大了，就改为c*21
        self.flatten = nn.Flatten()

    def forward(self, img):
        x = self.conv(img)  # (16,1,H,W)
        x = self.spp(x)
        out = self.fc(x)  # (16,1)
        return out
