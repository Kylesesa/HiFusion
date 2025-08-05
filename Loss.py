import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from utils import *
from ssim import *

class g_loss(nn.Module):
    def __init__(self):
        super(g_loss, self).__init__()

    def forward(self, v, i, img):
        # loss = grad_loss(v, i, img, window_size=3) + 2 * grad_loss(v, i, img, window_size=7) +\
        #        4 * grad_loss(v, i, img, window_size=15)+ 8 * grad_loss(v, i, img, window_size=31) + \
        #        16 * grad_loss(v, i, img, window_size=61)
        loss = grad_loss(v, i, img, window_size=3) + grad_loss(v, i, img, window_size=7) + \
               grad_loss(v, i, img, window_size=15) + grad_loss(v, i, img, window_size=31) + \
               grad_loss(v, i, img, window_size=61)
        return loss

class s_loss2(nn.Module):
    def __init__(self):
        super(s_loss2, self).__init__()

    def forward(self, v, i, img):
        g_v = grad(v)
        g_i = grad(i)
        g_map = (g_v+g_i)

        g_f = grad(img)
        # loss = torch.norm((g_map - g_f))
        loss = torch.sqrt(torch.sum(torch.square(g_map - g_f), dim=[2, 3]) + 1e-8)

        return loss

class b_loss(nn.Module):
    def __init__(self):
        super(b_loss, self).__init__()

    def forward(self, v, i, img):
        # loss = bright_loss(v, i, img, window_size=3) + 2 * bright_loss(v, i, img, window_size=7) \
        #         + 4 * bright_loss(v, i, img, window_size=15) + 8 * bright_loss(v, i, img, window_size=31) \
        #         + 16 * bright_loss(v, i, img, window_size=61)
        loss = bright_loss(v, i, img, window_size=3) + bright_loss(v, i, img, window_size=7) \
               + bright_loss(v, i, img, window_size=15) + bright_loss(v, i, img, window_size=31) \
               + bright_loss(v, i, img, window_size=61)
        return loss

class c_loss2(nn.Module):
    def __init__(self):
        super(c_loss2, self).__init__()

    def forward(self, v, i, img):
        b_v = bright2(v)
        b_i = bright2(i)
        b_map = b_v + b_i
        normalized_map = (b_map - torch.min(b_map)) / (torch.max(b_map) - torch.min(b_map))

        b_f = bright2(img)
        # loss = torch.norm((normalized_map - b_f))
        loss = torch.sqrt(torch.sum(torch.square(normalized_map - b_f), dim=[2, 3]) + 1e-8)
        return loss


class conloss(nn.Module):
    def __init__(self):
        super(conloss, self).__init__()
        self.c1 = con_loss(3)
        self.c2 = con_loss(7)
        self.c3 = con_loss(15)
        self.c4 = con_loss(31)
        self.c5 = con_loss(61)

    def forward(self, v, i, img):
        loss = 1*self.c1(v, i, img) + 1*self.c2(v, i, img) + \
               1*self.c3(v, i, img) + 1*self.c4(v, i, img) + \
               1*self.c5(v, i, img)

        # loss1 = torch.sqrt(torch.sum(torch.square(img - v), dim=[2, 3]) + 1e-8)
        # loss2 = torch.sqrt(torch.sum(torch.square(img - i), dim=[2, 3]) + 1e-8)
        return loss

class ms_ssim_loss(nn.Module):
    def __init__(self):
        super(ms_ssim_loss, self).__init__()
        self.ms_ssim = MS_SSIM()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    def forward(self, v, i, img):
        av = self.avgpool(v)
        bi = self.avgpool(i)

        a = av / (av + bi + 1e-8)
        b = bi / (av + bi + 1e-8)
        loss1 = 1 - self.ms_ssim(v, img)
        loss2 = 1 - self.ms_ssim(i, img)
        loss = a * loss1 + b * loss2
        return loss



class inf_loss(nn.Module):
    def __init__(self):
        super(inf_loss, self).__init__()
        self.c_loss = conloss()
    def infmap(self, v, i, encoder):
        infs_v = encoder(v)
        infs_i = encoder(i)
        return infs_i,infs_v

    def forward(self, v, i, img, encoder):
        infs_i, infs_v = self.infmap(v,i,encoder)
        infs_img = encoder(img)
        b, c, h, w = infs_img.shape
        loss_total = 0

        for i in range(c):
            inf_v = infs_v[:, i, :, :]
            inf_i = infs_i[:, i, :, :]
            inf_img = infs_img[:, i, :, :]

            inf_v = inf_v.unsqueeze(1)
            inf_i = inf_i.unsqueeze(1)
            inf_img = inf_img.unsqueeze(1)

            loss = self.c_loss(inf_v, inf_i, inf_img).mean()
            loss_total = loss_total + loss

        return loss_total/c


class L_con(nn.Module):
    def __init__(self):
        super(L_con, self).__init__()
        self.ssim_loss = ms_ssim_loss()
        self.c_loss = conloss()

    def forward(self, img, v, i):

        L_SSIM = self.ssim_loss(v, i, img).mean()
        L_C = self.c_loss(v,i,img).mean()

        return (1 * L_C + 25 * L_SSIM), L_SSIM
#1:89

class L_inf(nn.Module):
    def __init__(self):
        super(L_inf, self).__init__()
        self.i_loss = inf_loss()

    def forward(self, img, v, i, encoder):
        L_I = self.i_loss(v, i, img,encoder).mean()

        return L_I
#0.2:28
# class L_con1(nn.Module):
#     def __init__(self, eta):  # eta = 1.2
#         super(L_con1, self).__init__()
#         self.eta = eta
#
#     def forward(self, img, v, i):
#         # L_ir_con = torch.pow(torch.pow((img-i),2).sum(),0.5)
#         L_ir_con = torch.norm((img-i))
#
#         r = img-v
#         #[H,W] = r.shape[2:4]
#         tv_x = torch.pow((r[:, :, :, :-1] - r[:, :, :, 1:]), 2).mean()
#         tv_y = torch.pow((r[:, :, :-1, :] - r[:, :, 1:, :]), 2).mean()
#         L_vis_con = tv_x + tv_y
#
#         s1 = torch.exp(ssim(v, img))  # SSIM图[-1,1],避免负数的情况（虽然影响不大），通过e映射到[1/e,e]
#         s2 = torch.exp(ssim(i, img))
#
#         score = torch.mean((9*s1 + s2)/10, dim=(2, 3))
#         L_ssim = (torch.exp(torch.tensor(1.0)) - score).mean()
#
#         return (L_ir_con/16 + 10*L_vis_con + 100*L_ssim).mean(), L_ssim
#     # 在DDcGAN中，ir的内容损失，给了系数1/16，vis给了1.2，这会使得loss严重偏向优化vis，结果更像vis
#     # 这可能比我的系数，效果更好
#
# class L_con2(nn.Module):
#     def __init__(self, eta):  # eta = 1.2
#         super(L_con2, self).__init__()
#         self.eta = eta
#         self.s_loss2 = s_loss2()
#         self.c_loss2 = c_loss2()
#         # self.l_loss = L_grad()
#
#     def forward(self, img, v, i):
#         L_grad= self.s_loss2(v, i, img).mean()  # 将batch中的所有值平均
#         L_bright = self.c_loss2(v, i, img).mean()
#         print('L_bright',L_bright)
#         print('L_grad',L_grad)
#
#         # return (L_content + self.eta * L_ssim), L_ssim
#         return (L_grad+L_bright), L_grad  # 这里我把二者系数改为一致，认为内容和结构一致是一样重要的


class L_adv(nn.Module):
    def __init__(self):
        super(L_adv, self).__init__()
        self.eps = 1e-8

    def forward(self, score_Gv, score_Gi):

        # loss =(0.5 * (score_Gv-1) ** 2 + 0.5 * (score_Gi-1) ** 2).mean()#lsgan
        loss = -(torch.mean(score_Gv) + torch.mean(score_Gi)) #wgan
        # #print('G',loss)
        return loss

class L_G(nn.Module):
    def __init__(self):
        super(L_G, self).__init__()

        self.L_con = L_con()
        self.L_adv = L_adv()
        self.L_inf = L_inf()

    def forward(self, v, i, img, score_Gv, score_Gi,encoder):

        l_con, l_ssim = self.L_con(img, v, i)
        l_inf = self.L_inf(v, i, img, encoder)
        l_adv = self.L_adv(score_Gv, score_Gi)

        return (l_adv * 0.12 + 1 * l_con + 0.2 * l_inf), l_ssim
#0.12:1:0.2

class L_Dv(nn.Module):
    def __init__(self):
        super(L_Dv, self).__init__()
        self.eps = 1e-8

    def forward(self, score_v, score_Gv):
        # print(score_Gv)
        # print(score_v)
        # return (-torch.log(score_v)).mean() + (-torch.log(1 - score_Gv + self.eps)).mean()
        # return (0.5 * (score_v - 1) ** 2 + 0.5 * (score_Gv) ** 2).mean()  # LSGAN

        loss_critic = torch.mean(score_Gv) - torch.mean(score_v)
        #print('Dv:', loss_critic)
        return loss_critic          #wgan



class L_Di(nn.Module):
    def __init__(self):
        super(L_Di, self).__init__()
        self.eps = 1e-8

    def forward(self, score_i, score_Gi):
        # return (-torch.log(score_i)).mean() + (-torch.log(1 - score_Gi + self.eps)).mean()
        # return (0.5 * (score_i - 1) ** 2 + 0.5 * (score_Gi) ** 2).mean()  # LSGAN

        loss_critic = torch.mean(score_Gi) - torch.mean(score_i)
        #print('Di:',loss_critic)
        return loss_critic  # wgan
