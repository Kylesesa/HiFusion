import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Generator import *
from Discriminator import *
from torchstat import stat


class Model(nn.Module):
    def __init__(self,device):
        super(Model, self).__init__()

        self.G = Generator().to(device)
        # ————————cross-network——————————
        # self.G = SeAFusion().to(device)
        # self.G = DDcGAN().to(device)
        # self.G = U2Fusion().to(device)
        # self.G = TarDAL().to(device)
        #————————————————————————————————
        self.Dv = Discriminator().to(device)
        self.Di = Discriminator().to(device)




    def forward(self,vis,ir, is_train):
        fake_img = self.G(vis, ir)
        #print('model-1', torch.cuda.memory_allocated() / 1024 / 1024)   #增至18923->18931但是在训练D时，只增至28.718
        if is_train:
            score_v = self.Dv(vis)#print('model-2', torch.cuda.memory_allocated() / 1024 / 1024)#18923  D:4065
            score_i = self.Di(ir)#print('model-3', torch.cuda.memory_allocated() / 1024 / 1024)#18923 D:4065
            score_Gv = self.Dv(fake_img) #print('model-4', torch.cuda.memory_allocated() / 1024 / 1024)#增至22956   D:8100
            score_Gi = self.Di(fake_img)#print('model-5', torch.cuda.memory_allocated() / 1024 / 1024)#增至26990->26999        #D:8100/8101

            return fake_img,score_v,score_i,score_Gv,score_Gi
        else:
            return fake_img
# model = Model('cuda')
# g=model.G
# print(g)