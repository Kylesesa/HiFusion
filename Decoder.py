import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

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
        self.batch_norm = nn.BatchNorm2d(self.out_channels, eps=1e-05, momentum=0.1, affine=True,
                                         track_running_stats=True)
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

class Decoder(nn.Module):  # 240->1
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            # ConvLayer(in_channels=320, out_channels=320, kernel_size=3, stride=1, relu=True, bn=True, dropout=False),
            ConvLayer(in_channels=240, out_channels=160, kernel_size=3, stride=1, relu=True, bn=True, dropout=False),
            ConvLayer(in_channels=160, out_channels=40, kernel_size=3, stride=1, relu=True, bn=True, dropout=False),
            ConvLayer(in_channels=40, out_channels=20, kernel_size=3, stride=1, relu=True, bn=True, dropout=False),
            ConvLayer(in_channels=20, out_channels=10, kernel_size=3, stride=1, relu=True, bn=True, dropout=False),
            ConvLayer(in_channels=10, out_channels=1, kernel_size=3, stride=1, relu=False, bn=True, dropout=False)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):  # 160->1
        out = self.decoder(x)
        out = self.tanh(out)/2 + 0.5
        return out

class Decoder2(nn.Module):  # 240->1
    def __init__(self):
        super(Decoder2, self).__init__()

        # self.decoder = nn.Sequential(
        #     ConvLayer(in_channels=120, out_channels=120, kernel_size=3, stride=1, relu=True, bn=True, dropout=False),
        #     ConvLayer(in_channels=120, out_channels=40, kernel_size=3, stride=1, relu=True, bn=True, dropout=False),
        #     ConvLayer(in_channels=40, out_channels=20, kernel_size=3, stride=1, relu=True, bn=True, dropout=False),
        #     ConvLayer(in_channels=20, out_channels=10, kernel_size=3, stride=1, relu=True, bn=True, dropout=False),
        #     ConvLayer(in_channels=10, out_channels=1, kernel_size=1, stride=1, relu=False, bn=True, dropout=False)
        # )
        self.decoder = nn.Sequential(
            ConvLayer(in_channels=4, out_channels=64, kernel_size=3, stride=1, relu=True, bn=True, dropout=False),
            ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, relu=True, bn=True, dropout=False),
            ConvLayer(in_channels=64, out_channels=16, kernel_size=3, stride=1, relu=True, bn=True, dropout=False),
            ConvLayer(in_channels=16, out_channels=1, kernel_size=1, stride=1, relu=False, bn=True, dropout=False),

        )
        self.tanh = nn.Tanh()

    def forward(self, x):  # 160->1
        out = self.decoder(x)
        out = self.tanh(out)/2 + 0.5
        return out