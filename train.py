import os
import torch
import torch.nn as nn
from Loss import *
from Model import *
from datasets import *
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils import *

def train_G(model,L_G, vis, ir, iter_max, opt, en):
    for i in model.G.parameters():
        i.requires_grad = True
    for i in model.Dv.parameters():
        i.requires_grad = False
    for i in model.Di.parameters():
        i.requires_grad = False
    for i in en.parameters():
        i.requires_grad = False

    step = 0
    while step < iter_max:

        fake_img, score_v, score_i, score_Gv, score_Gi = model(vis, ir, True)

        loss_g, l_ssim = L_G(vis, ir, fake_img, score_Gv, score_Gi, en)

        step += 1
        opt.zero_grad()
        loss_g.backward()
        opt.step()

    return model, l_ssim,loss_g

def train_D(model, L_Dv, L_Di, vis, ir, iter_max, opt1, opt2,en):
    #print('Dv1', torch.cuda.memory_allocated() / 1024 / 1024)       #19.718
    for i in model.G.parameters():
        i.requires_grad = False
    for i in model.Dv.parameters():
        i.requires_grad = True
    for i in model.Di.parameters():
        i.requires_grad = False
    for i in en.parameters():
        i.requires_grad = False

    step1 = 0
    while step1 < iter_max:

        fake_img, score_v, _, score_Gv, _ = model(vis, ir, True)   #print('Dv4-1', torch.cuda.memory_allocated() / 1024 / 1024)     #8091
        loss_v = L_Dv(score_v, score_Gv)                    #print('Dv4-2', torch.cuda.memory_allocated() / 1024 / 1024)     #8091
        gp_v = gradient_penalty(model.Dv, vis, fake_img,device='cuda')
        loss_v = loss_v + 10 * gp_v

        step1 += 1
        opt1.zero_grad()
        loss_v.backward()
        opt1.step()

    for i in model.G.parameters():
        i.requires_grad = False
    for i in model.Dv.parameters():
        i.requires_grad = False
    for i in model.Di.parameters():
        i.requires_grad = True
    for i in en.parameters():
        i.requires_grad = False

    step2 = 0
    while step2 < iter_max:

        fake_img, _, score_i, _, score_Gi = model(vis, ir, True)   #print('Di2', torch.cuda.memory_allocated() / 1024 / 1024)       #8091
        loss_i = L_Di(score_i, score_Gi)                    #print('Di3', torch.cuda.memory_allocated() / 1024 / 1024)
        gp_i = gradient_penalty(model.Di, ir, fake_img, device='cuda')
        loss_i = loss_i + 10 * gp_i

        step2 += 1
        opt2.zero_grad()
        loss_i.backward()
        opt2.step()

    return model, loss_i, loss_v
