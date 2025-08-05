import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from datasets import *
from Loss import *
from Model import *
from train import *
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import time
from generate import *
import torch.distributed as dist
from AE3 import AE


# os.environ['CUDA_VISIBLE_DEVICES'] = '2,5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("current device：",torch.cuda.get_device_name(device))

# local_rank = int(os.environ["LOCAL_RANK"])
# dist.init_process_group(backend="gloo|nccl")

# file_path = 'Training_Dataset.h5'
test_path = '/home/admin/xkc/RCMCGAN_pytorch/TNO_bmp'
test_path2 = '/home/admin/xkc/RCMCGAN_pytorch/VIFB'
test_path3 = '/home/admin/xkc/RCMCGAN_pytorch/MSRS'
# test_path_medical = '/home/admin/xkc/Medical/PET-MRI/test'
# save_path= 'result/result2/'

log_path = '/home/admin/xkc/RCMCGAN_pytorch/Hyperpara2/Lpixel+Ldetail'
metric_path = log_path+'/metrics_log.xlsx'

#BATCH_SIZE = 224
# BATCH_SIZE = 24
BATCH_SIZE =8

EPOCH = 3
lr = 0.0002

EPSILON = 1e-5
DECAY_RATE = 0.9
eps = 1e-8

Datasets = datasets

num_imgs = Datasets.__len__()
batches = int(num_imgs // BATCH_SIZE)
mod = num_imgs % BATCH_SIZE
print('Train images number %d, Batches: %d.\n' % (num_imgs, batches))

# train_sampler = torch.utils.data.distributed.DistributedSampler(Datasets)
data_loader = DataLoader(Datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4) #batch_size=24,batch =954
ae = AE()

ae.load_state_dict(torch.load('/home/admin/xkc/RCMCGAN_pytorch/logAE/log5/10/400/400.pth'))
is_train = True

def main():
    if is_train:

        iter_num = 3
        torch.autograd.set_detect_anomaly(True)

        model = Model(device) # ----------------初始化模型-----------------# #print(f'此时初始化模型：,{torch.cuda.memory_allocated()*1000/1024/1024/1024}')
        Loss_G = L_G().cuda()
        Loss_Dv = L_Dv().cuda()
        Loss_Di = L_Di().cuda()#print(f'此时初始化loss：,{torch.cuda.memory_allocated() *1000/1024/ 1024 / 1024}')

        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        #若要断点续训，就用这两行代码
        # checkpoint_path = '/home/admin/xkc/RCMCGAN_pytorch/log/log_d9_2/1/100/100.path'
        # model.load_state_dict(torch.load(checkpoint_path))

        opt1 = torch.optim.Adam(model.G.parameters(), lr,betas=(0.9,0.999))
        # opt1 = torch.optim.RMSprop(model.G.parameters(), lr, weight_decay=0.9)#ddcgan,u2fusion
        opt2 = torch.optim.Adam(model.Dv.parameters(), lr,betas=(0.9,0.999))
        opt3 = torch.optim.Adam(model.Di.parameters(), lr,betas=(0.9,0.999))

        exp_lr_scheduler1 = lr_scheduler.ExponentialLR(opt1, gamma=0.9)
        exp_lr_scheduler2 = lr_scheduler.ExponentialLR(opt2, gamma=0.9)
        exp_lr_scheduler3 = lr_scheduler.ExponentialLR(opt3, gamma=0.9)#print(f'此时初始化优化器：,{torch.cuda.memory_allocated() / 1024 / 1024}')

        G_loss = []
        Di_loss = []
        Dv_loss = []
        best_metrics = np.zeros(10)

        for epoch in range(EPOCH):
            print("Epoch:", epoch + 1)
            # model,loss_G = train_D(model,l_max,Loss_G,Loss_Dv,Loss_Di,)
            # model= train_G(model,l_max,l_min,Loss_G,Loss_Dv,Loss_Di,d)
            os.makedirs(log_path + '/' + str(epoch + 1), exist_ok=True)  # log/log2/epoch/
            step = 0
            for vis_batch,ir_batch in data_loader:  # dataset2 batch(24,2,84,84)
            # for batch in data_loader:  # dataset1 batch(24,2,84,84)                   #用来加载.h5文件的data
                step += 1
                # print(step)
                # vis = batch[:, 0, :, :]
                # vis = torch.unsqueeze(vis, dim=1).to(device)  # (24,1,84,84)
                # ir = batch[:, 1, :, :]
                # ir = torch.unsqueeze(ir, dim=1).to(device)  # (24,1,84,84)            #这四句使用.h5文件时要加上，因为数据类型不一样

                vis = vis_batch.to(device)
                ir = ir_batch.to(device)

                # fake_img, score_v, score_i, score_Gv, score_Gi = model(vis,ir,is_train)  # ----------------每个batch，先过一边当前的模型，得到初始的生成图像，以及鉴别器对真假图像的打分-----------------#
                # loss_g = Loss_G(vis, ir, fake_img, score_Gv,score_Gi)  # ----------------得到当前batch的生成器总loss-----------------#
                # loss_adv = Loss_adv(score_Gv, score_Gi)
                # loss_v = Loss_Dv(score_v, score_Gv)
                # loss_i = Loss_Di(score_i, score_Gi)

                if step % 2 == 0:
                    model, loss_i, loss_v = train_D(model, Loss_Dv, Loss_Di, vis, ir, iter_num*2, opt2, opt3, ae.en)

                else:
                    model, l_ssim, loss_g = train_G(model, Loss_G, vis, ir, iter_num, opt1, ae.en)
                if step % 10 == 0:
                    # lr1 = exp_lr_scheduler1.get_last_lr()[0]
                    # lr2 = exp_lr_scheduler2.get_last_lr()[0]
                    print(
                        f"Epoch:[{epoch + 1}/{EPOCH}],\tBatch[{step}/{len(data_loader)}],\tG_loss:{loss_g.item():.4f},\tDi_loss:{loss_i.item():.4f},\tDv_loss:{loss_v.item():.4f},\tssim_loss:{l_ssim.item():.4f}")
                    G_loss.append(loss_g.item())
                    Di_loss.append(loss_i.item())
                    Dv_loss.append(loss_v.item())

                if step % 100 == 0:
                    #在保存log数据的同时，我也设计了图像测试程序，可以保存每个log的测试图像
                    os.makedirs(log_path + '/' + str(epoch + 1)+'/'+str(step), exist_ok=True)
                    train_save_path = log_path + '/' + str(epoch + 1)+'/'+str(step)+'/'
                    torch.save(model.state_dict(), train_save_path+ str(step)+'.path')

                    with open(f"{log_path}/loss_g.txt", 'w') as file:
                        for loss in G_loss:
                            file.write(str(loss) + '\n')
                    with open(f"{log_path}/loss_dv.txt", 'w') as file:
                        for loss in Dv_loss:
                            file.write(str(loss) + '\n')
                    with open(f"{log_path}/loss_di.txt", 'w') as file:
                        for loss in Di_loss:
                            file.write(str(loss) + '\n')

                    with torch.no_grad():
                        print('start generate...')
                        generate_VIFB(train_save_path, test_path3, train_save_path, step, model)

                        print('start evaluate...')
                        metrics = get_metrics(test_path3, train_save_path)
                        print("=" * 90)
                        print("\t EN\t SD\t SF\t VIFF\t AG\t SCD\t MI\t Qabf\t SSIM\t CC\t")
                        print("best:\t"
                              + str(np.round(best_metrics[0], 2)) + '\t'
                              + str(np.round(best_metrics[1], 2)) + '\t'
                              + str(np.round(best_metrics[2], 2)) + '\t'
                              + str(np.round(best_metrics[3], 2)) + '\t'
                              + str(np.round(best_metrics[4], 2)) + '\t'
                              + str(np.round(best_metrics[5], 2)) + '\t'
                              + str(np.round(best_metrics[6], 2)) + '\t'
                              + str(np.round(best_metrics[7], 2)) + '\t'
                              + str(np.round(best_metrics[8], 2)) + '\t'
                              + str(np.round(best_metrics[9], 2))
                              )
                        print("This:\t"
                              + str(np.round(metrics[0], 2)) + '\t'
                              + str(np.round(metrics[1], 2)) + '\t'
                              + str(np.round(metrics[2], 2)) + '\t'
                              + str(np.round(metrics[3], 2)) + '\t'
                              + str(np.round(metrics[4], 2)) + '\t'
                              + str(np.round(metrics[5], 2)) + '\t'
                              + str(np.round(metrics[6], 2)) + '\t'
                              + str(np.round(metrics[7], 2)) + '\t'
                              + str(np.round(metrics[8], 2)) + '\t'
                              + str(np.round(metrics[9], 2))
                              )
                        print("=" * 90)
                        save_metrics_to_excel(metrics, metric_path)
                        better_than_best = metrics > best_metrics
                        count_better = np.sum(better_than_best)

                        # 判断是否保存模型
                        if count_better >= 5:
                            os.makedirs(log_path+ '/good_result', exist_ok=True)
                            torch.save(model.state_dict(), log_path+ '/good_result/' +str(epoch+1)+'_'+str(step)+'.path')
                        if count_better >= 7:
                            os.makedirs(log_path+ '/best_result', exist_ok=True)
                            best_metrics = metrics
                            torch.save(model.state_dict(), log_path+ '/best_result/' +str(epoch+1)+'_'+str(step)+'.path')


            exp_lr_scheduler1.step()
            exp_lr_scheduler2.step()
            exp_lr_scheduler3.step()



if __name__=='__main__':
    main()
