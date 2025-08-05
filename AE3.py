import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import *
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from ssim import *
from utils import *
from Encoder import *
from Decoder import *


#----------------------------------------------------网络结构--------------------------------------------------#
class AE(nn.Module):  # 1->120->1
    def __init__(self):
        super(AE, self).__init__()
        self.en = Encoder2().to(device)
        self.de = Decoder2().to(device)


    def forward(self, img):  # 4->1
        feature = self.en(img)
        r_img = self.de(feature)
        return r_img


#----------------------------------------------------损失函数--------------------------------------------------#
def loss1(img, r_img, window_size = 3):
    g_i = grad(img, window_size)
    g_ri = grad(r_img, window_size)
    loss = torch.sqrt(torch.sum(torch.square(g_i - g_ri), dim=[2, 3]) + 1e-8)

    return loss

def loss2(img, r_img, window_size = 1):
    b_i = bright2(img, window_size)
    b_ri = bright2(r_img, window_size)
    loss = torch.sqrt(torch.sum(torch.square(b_i - b_ri), dim=[2, 3]) + 1e-8)

    return loss

class g_loss(nn.Module):
    def __init__(self):
        super(g_loss, self).__init__()
    def forward(self, img,r_img):
        loss = 16*loss1(img, r_img, window_size=3)+4*loss1(img, r_img, window_size=11)+loss1(img, r_img, window_size=31)
        return loss

class b_loss(nn.Module):
    def __init__(self):
        super(b_loss, self).__init__()
    def forward(self, img,r_img):
        loss = 16*loss2(img, r_img, window_size=3)+4*loss2(img, r_img, window_size=11)+loss2(img, r_img, window_size=31)
        return loss

class ms_ssim_loss(nn.Module):
    def __init__(self):
        super(ms_ssim_loss, self).__init__()
        self.ms_ssim = MS_SSIM()

    def forward(self,img, r_img):
        loss = 1 - self.ms_ssim(r_img, img)
        return loss

class F_loss(nn.Module):
    def __init__(self):
        super(F_loss, self).__init__()
    def forward(self, img,r_img):
        loss = loss2(img, r_img, window_size=1)
        return loss

class L_content(nn.Module):
    def __init__(self):  # eta = 1.2
        super(L_content, self).__init__()
        self.g_loss = g_loss()
        self.b_loss = b_loss()
        self.ssim_loss = ms_ssim_loss()
        # self.l_loss = L_grad()

        #only F-norm
        self.f_loss = F_loss()

    def forward(self, img, r_img):

        # L_grad = self.g_loss(img, r_img).mean()
        # L_bright = self.b_loss(img, r_img).mean()
        L_SSIM = self.ssim_loss(img, r_img).mean()
        # loss1 = 80 * L_grad + L_bright + 100*L_SSIM #相似度更高

        loss2 = self.f_loss(img, r_img).mean()

        return loss2, L_SSIM


class L_G2(nn.Module):
    def __init__(self):
        super(L_G2, self).__init__()

        self.L_con = L_content()


    def forward(self, img, r_img):

        l_con, l_ssim = self.L_con(img, r_img)
        return l_con, l_ssim


#----------------------------------------------------训练策略--------------------------------------------------#
def train_G2(model,L_G2, vis, ir, iter_max, opt):
    for i in model.parameters():
        i.requires_grad = True

    step1 = 0
    while step1 < iter_max:
        r_vis = model(vis)

        loss_g, l_ssim = L_G2(vis, r_vis)

        step1 += 1
        opt.zero_grad()
        loss_g.backward()
        opt.step()

    step2 = 0
    while step2 < iter_max:
        r_ir = model(ir)

        loss_g, l_ssim = L_G2(ir, r_ir)

        step2 += 1
        opt.zero_grad()
        loss_g.backward()
        opt.step()

    return model, l_ssim, loss_g

#----------------------------------------------------主函数--------------------------------------------------#
def main2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("current device：", torch.cuda.get_device_name(device))

    Datasets = pet_mri_datasets
    path = '/home/admin/xkc/Medical/PET-MRI/test/MRI'
    log_path = 'logAE/pet_mri'
    BATCH_SIZE = 16

    iter_num = 3
    EPOCH = 10
    lr = 0.0002

    num_imgs = Datasets.__len__()
    batches = int(num_imgs // BATCH_SIZE)

    print('Train images number %d, Batches: %d.\n' % (num_imgs, batches))

    data_loader = DataLoader(Datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)  # batch_size=24,batch =954

    torch.autograd.set_detect_anomaly(True)

    model = AE().to(device)
    Loss_G = L_G2().to(device)

    opt1 = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999))
    exp_lr_scheduler1 = lr_scheduler.ExponentialLR(opt1, gamma=0.9)


    for epoch in range(EPOCH):
        print("Epoch:", epoch + 1)
        os.makedirs(log_path + '/' + str(epoch + 1), exist_ok=True)  # log/log2/epoch/
        step = 0
        for vis_batch, ir_batch in data_loader:  # dataset2 batch(24,2,84,84)
            step += 1

            vis = vis_batch.to(device)
            ir = ir_batch.to(device)

            model, l_ssim, loss_g = train_G2(model, Loss_G, vis, ir, iter_num, opt1)

            if step % 1 == 0:
                print(
                    f"Epoch:[{epoch + 1}/{EPOCH}],\tBatch[{step}/{len(data_loader)}],\t\tAE_loss:{loss_g.item():.4f},\tssim_loss:{l_ssim.item():.4f}")

            if step % 10 == 0:

                os.makedirs(log_path + '/' + str(epoch + 1) + '/' + str(step), exist_ok=True)
                train_save_path = log_path + '/' + str(epoch + 1) + '/' + str(step) + '/'
                torch.save(model.state_dict(), train_save_path + str(step) + '.pth')

                with torch.no_grad():
                    for img_name in os.listdir(path):
                        img_path = os.path.join(path, img_name)

                        img = plt.imread(img_path) / 255.0
                        img1 = torch.Tensor(img)
                        img2 = torch.unsqueeze(img1, 0)
                        img_tensor = torch.unsqueeze(img2, 0)
                        img_tensor = img_tensor.to(device)
                        rebuild_img = model(img_tensor)

                        img_filename = f'{train_save_path}{img_name}'
                        save_image(rebuild_img, img_filename)

        exp_lr_scheduler1.step()



# if __name__=='__main__':
#     main2()