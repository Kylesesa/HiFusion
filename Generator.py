import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from Encoder import *
eps = 1e-8


class FuseBlock(nn.Module):  # C=80
    '''
    定义两个特征提取器分别提取红外图像的特征图和可见光图像的特征图
    '''

    def __init__(self):
        super(FuseBlock, self).__init__()

        self.encoder1 = Encoder()
        self.encoder2 = Encoder()

    def feature(self, vis, ir):

        vis_main_feature = self.encoder1(vis)
        ir_main_feature = self.encoder2(ir)

        return vis_main_feature, ir_main_feature

    def fusion(self, vis, ir):  # C=80
        [vis_main_feature, ir_main_feature] = self.feature(vis, ir)


        main_feature = torch.cat([vis_main_feature, ir_main_feature], dim=1)
        # res_feature = torch.concat([vis_res_feature, ir_res_feature], dim=1)

        return main_feature
        # return vis_main_feature, ir_main_feature            #160,160

    def forward(self, vis, ir):  # 160+160==320
        main_feature = self.fusion(vis, ir)

        return main_feature


# class FuseBlock2(nn.Module):  # C=80
#     '''
#     一个特征提取器，分别提取红外和可见光图像的特征，这么做是为了了使得得到的特征图具有联系
#     '''
#
#     def __init__(self):
#         super(FuseBlock2, self).__init__()
#         denseblock = DenseBlock
#
#         self.conv = ConvLayer(in_channels=1, out_channels=16, kernel_size=5, stride=1)
#         self.DB = denseblock(in_channels=16, kernel_size=3, stride=1, relu=True, bn=True, dropout=False)
#
#     def feature(self, vis, ir):
#         x1 = self.conv(vis)
#         x2 = self.conv(ir)
#
#         vis_main_feature = self.DB(x1)
#         vis_res_feature = torch.abs(vis - vis_main_feature)  # 采用绝对值的方法得到残差图
#         # vis_res_feature = (vis - vis_main_feature)**2      #考虑到可能出现0点无法求导的情况，所以使用平方。
#
#         ir_main_feature = self.DB(x2)
#         ir_res_feature = torch.abs(ir - ir_main_feature)
#         # ir_res_feature = (ir - ir_main_feature)**2
#
#         return vis_main_feature, vis_res_feature, ir_main_feature, ir_res_feature
#
#     def fusion(self, vis, ir):  # C=80
#         [vis_main_feature, vis_res_feature, ir_main_feature, ir_res_feature] = self.feature(vis, ir)
#         gmp = nn.AdaptiveMaxPool2d(output_size=(1, 1))
#         gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#
#         vis_gmp = gmp(vis_main_feature)  # [0,1]
#         ir_gmp = gmp(ir_main_feature)
#         vis_gap = gap(vis_res_feature)  # [0,1]
#         ir_gap = gap(ir_res_feature)
#
#         gmp_add_weight = vis_gmp + ir_gmp  # [0,2]
#         gap_add_weight = vis_gap + ir_gap
#
#         vis_gmp_weight = torch.sigmoid(torch.exp(torch.log(vis_gmp) - torch.log(gmp_add_weight + eps)))
#         ir_gmp_weight = torch.sigmoid(torch.exp(torch.log(ir_gmp) - torch.log(gmp_add_weight + eps)))
#         vis_gap_weight = torch.sigmoid(torch.exp(torch.log(vis_gap) - torch.log(gap_add_weight + eps)))
#         ir_gap_weight = torch.sigmoid(torch.exp(torch.log(ir_gap) - torch.log(gap_add_weight + eps)))
#
#         # vis_gmp = torch.exp(gmp(vis_main_feature))#[1,e]
#         # ir_gmp = torch.exp(gmp(ir_main_feature))
#         # vis_gap = torch.exp(gap(vis_res_feature))  #[1,e]
#         # ir_gap = torch.exp(gap(ir_res_feature))
#         #
#         # vis_gmp_weight = vis_gmp/(vis_gmp+ir_gmp)
#         # ir_gmp_weight = ir_gmp/(vis_gmp+ir_gmp)
#         # vis_gap_weight = vis_gap/(vis_gap+ir_gap)
#         # ir_gap_weight = ir_gap/(vis_gap+ir_gap)
#
#         main_feature = vis_main_feature * vis_gmp_weight + ir_main_feature * ir_gmp_weight
#         res_feature = vis_res_feature * vis_gap_weight + ir_res_feature * ir_gap_weight
#
#         return main_feature, res_feature
#
#     def forward(self, vis, ir):  # 80+80->160
#         [main_feature, res_feature] = self.fusion(vis, ir)
#         concat_result = torch.concat([main_feature, res_feature], dim=1)
#
#
#         return concat_result


class Decoder(nn.Module):  # 160->1
    def __init__(self):
        super(Decoder, self).__init__()
        # self.cbam = CBAM(160)
        # self.eca = ECA(160)
        # self.bam = BAMBlock(channel=320)

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
        # x = self.cbam(x)
        # x = self.eca(x)
        # x = self.bam(x)
        out = self.decoder(x)
        out = self.tanh(out)/2 + 0.5
        return out

class Generator(nn.Module):  # (24,1,84,84)+(24,1,84,84)->(24,1,84,84)
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder = FuseBlock()
        self.decoder = Decoder()

    def forward(self, vis, ir):
        feature_map = self.encoder(vis, ir)
        result = self.decoder(feature_map)
        return result



#测试融合网络参数量
# g= Generator()
# para = sum([np.prod(list(p.size())) for p in g.parameters()])
# type_size = 1
# print('Model {} : params: {:4f}M'.format(g._get_name(), para * type_size / 1000 / 1000))