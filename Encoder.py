import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

eps = 1e-8

# Convolution operation
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
        self.batch_norm = nn.BatchNorm2d(self.out_channels)
        self.Relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        out = self.reflection_pad(x)

        out = self.conv2d(out)

        if self.bn is True:
            out = self.batch_norm(out)
        if self.relu is True:
            out = self.Relu(out)
            # print('out1:',out)
        # if self.tanh is True:
        #     out = self.Tanh(out) / 2 + 0.5
        if self.dropout is True:
            out = self.Dropout(out)
        return out


# Dense convolution unit
class DenseConv2d(nn.Module):  # inchannels->conv2d->dropout->relu->outchannels + inchannels
    def __init__(self, in_channels, out_channels, kernel_size, stride, relu, bn, dropout):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride, relu, bn, dropout)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# Dense Block unit
class DenseBlock(nn.Module):  # DenseBlock==input->DB1->DB2->DB3->DB4->out
    def __init__(self, in_channels, kernel_size, stride, relu, bn, dropout):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride, relu, bn, dropout),
                       DenseConv2d(in_channels + out_channels_def, out_channels_def, kernel_size, stride, relu, bn,
                                   dropout),
                       # DenseConv2d(in_channels + out_channels_def * 2, out_channels_def, kernel_size, stride, relu, bn,
                       #             dropout),
                       # DenseConv2d(in_channels + out_channels_def * 3, out_channels_def, kernel_size, stride, relu, bn,
                       #             dropout)
                       ]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)#48
        return out

class Encoder(nn.Module):  # inchannels->conv2d->dropout->relu->outchannels:120
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = ConvLayer(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.DB1 = DenseBlock(in_channels=16, kernel_size=3, stride=1, relu=True, bn=True,dropout=False)
        self.conv2 = DenseConv2d(in_channels=48, out_channels=16, kernel_size=5, stride=1,relu=True,bn=True,dropout=False)
        self.DB2 = DenseBlock(in_channels=64, kernel_size=3, stride=1, relu=True, bn=True,dropout=False)
        self.conv3 = DenseConv2d(in_channels=96, out_channels=24, kernel_size=5, stride=1,relu=True,bn=True,dropout=False)
        # self.conv4 = ConvLayer(in_channels=8, out_channels=4, kernel_size=3, stride=1)
        self.tanh = nn.Tanh()
        self.encoder = nn.Sequential(
            self.conv1,
            self.DB1,
            self.conv2,
            self.DB2,
            self.conv3,
            # self.conv4
        )

    def forward(self, x):
        out = self.encoder(x)
        return out

class Encoder2(nn.Module):  # inchannels->conv2d->dropout->relu->outchannels:120
    def __init__(self):
        super(Encoder2, self).__init__()

        self.conv1 = ConvLayer(in_channels=1, out_channels=16, kernel_size=3, stride=1,relu=True,bn=True,
                               dropout=False)
        self.conv2 = ConvLayer(in_channels=16, out_channels=64, kernel_size=3, stride=1, relu=True,bn=True,
                               dropout=False)
        self.conv3 = ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, relu=True, bn=True,
                               dropout=False)
        self.conv4 = ConvLayer(in_channels=64, out_channels=4, kernel_size=1, stride=1, relu=True, bn=True,
                               dropout=False)
        self.encoder = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4
        )

    def forward(self, x):
        out = self.encoder(x)
        return out

class Encoder3(nn.Module):  # inchannels->conv2d->dropout->relu->outchannels:120
    def __init__(self):
        super(Encoder3, self).__init__()

        self.conv1 = ConvLayer(in_channels=1, out_channels=16, kernel_size=3, stride=1,relu=True,bn=True,
                               dropout=False)
        self.conv2 = ConvLayer(in_channels=16, out_channels=32, kernel_size=3, stride=1, relu=True,bn=True,
                               dropout=False)
        self.conv3 = ConvLayer(in_channels=32, out_channels=48, kernel_size=3, stride=1, relu=True, bn=True,
                               dropout=False)
        self.conv4 = ConvLayer(in_channels=48, out_channels=64, kernel_size=5, stride=1, relu=True, bn=True,
                               dropout=False)
        self.conv5 = ConvLayer(in_channels=64, out_channels=80, kernel_size=3, stride=1, relu=True, bn=True,
                               dropout=False)
        self.conv6 = ConvLayer(in_channels=80, out_channels=96, kernel_size=3, stride=1, relu=True, bn=True,
                               dropout=False)
        self.conv7 = ConvLayer(in_channels=96, out_channels=120, kernel_size=3, stride=1, relu=True, bn=True,
                               dropout=False)
        self.encoder = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
            self.conv7
        )

    def forward(self, x):
        out = self.encoder(x)
        return out

class ResBlock(nn.Module):  # inchannels->conv2d->dropout->relu->outchannels:120
    def __init__(self,inchannel):
        super(ResBlock, self).__init__()
        self.DB1 = DenseBlock(in_channels=inchannel, kernel_size=3, stride=1, relu=True, bn=True,dropout=False)

    def forward(self, x):
        main = self.DB1(x)
        res = torch.abs(x - main)
        return main, res

# class Encoder3(nn.Module):  # inchannels->conv2d->dropout->relu->outchannels:120
#     def __init__(self):
#         super(Encoder3, self).__init__()
#
#         self.conv1 = ConvLayer(in_channels=1, out_channels=16, kernel_size=3, stride=1)
#         self.RB1 = ResBlock(inchannel=16)#out=48
#         self.RB2 = ResBlock(inchannel=48)#out=80
#         self.conv2 = ConvLayer(in_channels=80, out_channels=4, kernel_size=1, stride=1, relu=True, bn=True,
#                                dropout=False)
#         self.conv3 = ConvLayer(in_channels=128, out_channels=4, kernel_size=1, stride=1, relu=True, bn=True,
#                                dropout=False)
#
#         self.encoder = nn.Sequential(
#             self.conv1,
#             self.conv2,
#             self.conv3,
#             self.conv4
#         )
#
#     def forward(self, x):
#         out = self.encoder(x)
#         return out