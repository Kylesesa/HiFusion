import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch
import os
import cv2
import time
from utils import *
from PIL import Image
from Model import *
from AE3 import *
from torchvision.transforms.functional import to_pil_image
from torchstat import stat
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate(log_path, test_path, save_path, step, model):
    '''
    
    :param log_path: log的保存路径
    :param test_path: 测试图像放的位置
    :param step: 需要知道现在训练到多少step了，这样才可以调用这个step的log用于model的融合
    :param model: 调用训练的模型，用来生成图像
    :return: None
    '''
    vi_dir = test_path + '/' + 'vi'
    ir_dir = test_path + '/' + 'ir'
    index = 1
    T = 0

    for name in os.listdir(vi_dir):

        vis_path = os.path.join(vi_dir, name)
        ir_path = os.path.join(ir_dir, name)

        ir_img = plt.imread(ir_path) / 255.0
        vis_img = plt.imread(vis_path) / 255.0
        ir1 = torch.Tensor(ir_img)  # (h,w)
        vis1 = torch.Tensor(vis_img)  # (h,w)

        ir2 = torch.unsqueeze(ir1, 0)
        ir = torch.unsqueeze(ir2, 0)  # (1,1,h,w)
        vis2 = torch.unsqueeze(vis1, 0)
        vis = torch.unsqueeze(vis2, 0)  # (1,1,h,w)

        vis = vis.to(device)
        ir = ir.to(device)

        model.load_state_dict(torch.load(log_path+'/'+ str(step) + '.path'))
        start = time.time()
        img = model(vis, ir, is_train=False)
        end = time.time()

        img_filename = f'{save_path}/{name}'
        save_image(img, img_filename)
        print(f'[{index}]{name} has been finished,[{end - start}]')
        T = T + (end - start)
        index += 1
    print(f'Finished,average time=[{(T / (index - 1))}]')

#<---------------------------------------生成VIFB测试图像-------------------------------------->
def generate_VIFB(log_path, test_path, save_path, step, model):
    '''

    :param log_path: log的保存路径
    :param test_path: 测试图像放的位置
    :param step: 需要知道现在训练到多少step了，这样才可以调用这个step的log用于model的融合
    :param model: 调用训练的模型，用来生成图像
    :return: None
    '''
    os.makedirs(save_path,exist_ok=True)
    vi_dir = test_path + '/' + 'vi'
    ir_dir = test_path + '/' + 'ir'
    index=1
    T = 0

    for name in os.listdir(vi_dir):

        vi_path = os.path.join(vi_dir, name)
        ir_path = os.path.join(ir_dir, name)

        ir_img = Image.open(ir_path).convert('L')

        Y, Cb, Cr, tag = RGB2Y(vi_path)

        ir = loader(ir_img).unsqueeze(0)# (1,1,h,w)
        Y = loader(Y).unsqueeze(0)
        Cb = loader(Cb).unsqueeze(0)
        Cr = loader(Cr).unsqueeze(0)

        Y = Y.to(device)
        ir = ir.to(device)
        Cb = Cb.to(device)
        Cr = Cr.to(device)

        model.load_state_dict(torch.load(log_path + '/' + str(step) + '.path'),strict=False)

        start = time.time()
        gray_img = model(Y, ir, is_train=False)
        # gray_img = (gray_img - torch.min(gray_img)) / (torch.max(gray_img) - torch.min(gray_img))

        if tag == True:
            img = YCbCr2RGB(gray_img, Cb, Cr)
        else:
            img = gray_img
        end = time.time()

        img_filename = f'{save_path}/{name}'
        save_image(img, img_filename)
        print(f'[{index}]{name} has been finished,[{end-start}]')
        T=T+(end-start)
        index+=1
    print(f'Finished,average time=[{(T/(index-1))}]')

# log_path = '/home/admin/xkc/RCMCGAN_pytorch/comparision/seafusion/1/2900'
# test_path = '/home/admin/xkc/OtherMethods/ATGANFusion-master/prepare_Dataset/MSRS'
# save_path = '/home/admin/xkc/RCMCGAN_pytorch/comparision/seafusion/result_MSRS_1_2900'
# model = Model(device)
# generate_VIFB(log_path, test_path, save_path, 2900, model)

#<---------------------------------------生成测试图像-------------------------------------->
def generate3(log_path, test_path, save_path, step, model):
    '''
    :param log_path: log的保存路径
    :param test_path: 测试图像放的位置
    :param step: 需要知道现在训练到多少step了，这样才可以调用这个step的log用于model的融合
    :param model: 调用训练的模型，用来生成图像
    :return: None
    '''
    T = 0
    for i in range(62):
        index = i + 1

        ir_path = test_path + '/' + 'IR' + str(index) + '.bmp'
        vis_path = test_path + '/' + 'VIS' + str(index) + '.bmp'

        ir_img = plt.imread(ir_path) / 255.0
        vis_img = plt.imread(vis_path) / 255.0
        ir1 = torch.Tensor(ir_img)  # (h,w)
        vis1 = torch.Tensor(vis_img)  # (h,w)

        ir2 = torch.unsqueeze(ir1, 0)
        ir = torch.unsqueeze(ir2, 0)  # (1,1,h,w)
        vis2 = torch.unsqueeze(vis1, 0)
        vis = torch.unsqueeze(vis2, 0)  # (1,1,h,w)

        vis = vis.to(device)
        ir = ir.to(device)

        model.load_state_dict(torch.load(log_path + '/' + str(step) + '.path'))
        start = time.time()
        img = model(vis, ir, is_train=False)
        end = time.time()

        img_filename = f'{save_path}{index}.png'
        save_image(img, img_filename)
        # print(f'[{index}] has been finished,[{end - start}]')
        T = T + (end - start)
        index += 1
    print(f'Finished,average time=[{(T / 62)}]')

# log_path = '/home/admin/xkc/RCMCGAN_pytorch/Ablation/woLDI/1/500'
# test_path = '/home/admin/xkc/RCMCGAN_pytorch/TNO_bmp'
# save_path = '/home/admin/xkc/aaa/'
# model = Model(device)
# generate3(log_path, test_path, save_path,500, model)

def generate_feature(log_path, save_path, step):
    '''
    输出中间特征图
    :param log_path:
    :param save_path:
    :param step:
    :return:
    '''
    os.makedirs(save_path,exist_ok=True)

    # ir_path = '/home/admin/xkc/RCMCGAN_pytorch/TNO_bmp/IR2.bmp'
    # vis_path = '/home/admin/xkc/RCMCGAN_pytorch/TNO_bmp/VIS2.bmp'
    ir_path = '/home/admin/xkc/RCMCGAN_pytorch/VIFB_Y/IR/kettle.jpg'
    vis_path = '/home/admin/xkc/RCMCGAN_pytorch/VIFB_Y/VI/kettle.jpg'

    ir_img = plt.imread(ir_path) / 255.0
    vis_img = plt.imread(vis_path) / 255.0
    ir1 = torch.Tensor(ir_img)  # (h,w)
    vis1 = torch.Tensor(vis_img)  # (h,w)

    ir2 = torch.unsqueeze(ir1, 0)
    ir = torch.unsqueeze(ir2, 0)  # (1,1,h,w)
    vis2 = torch.unsqueeze(vis1, 0)
    vis = torch.unsqueeze(vis2, 0)  # (1,1,h,w)

    vis = vis.to(device)
    ir = ir.to(device)

    model = Model(device)
    model.load_state_dict(torch.load(log_path + '/' + str(step) + '.path'))
    _, features = model(vis, ir, is_train=False)
    features = features.squeeze(0)
    print(features.shape)

    c,h,w = features.shape
    for index in range(c):
        feature = features[index,:,:]
        feature_name = f'{save_path}{index}.png'
        save_image(feature, feature_name)
        print(index ,' finished')

# save_path = '/home/admin/xkc/RCMCGAN_pytorch/features/log_b13_8_200/'
# log_path = '/home/admin/xkc/RCMCGAN_pytorch/log/log_b13/8/200'
#
# generate_feature(log_path, save_path, step=200)

def generate_infmap(log_path, save_path, step):
    '''
    输出中间特征图
    :param log_path:
    :param save_path:
    :param step:
    :return:
    '''
    os.makedirs(save_path,exist_ok=True)

    vis_path = '/home/admin/xkc/RCMCGAN_pytorch/数据集/MSRS-main/test/vi/01502D.png'

    vis_img = Image.open(vis_path).convert('L')
    vis = loader(vis_img).unsqueeze(0)


    vis = vis.to(device)


    model = AE()

    model.load_state_dict(torch.load(log_path + '/' + str(step) + '.pth'))

    with torch.no_grad():
        features = model(vis) #选择需要生成的图片的位置

    activation_map = features.squeeze(0)
    c,h,w = activation_map.shape
    for index in range(c):
        feature = activation_map[index,:,:]

        feature_name = f'{save_path}/{index}.png'
        save_image(feature, feature_name)
        print(index ,' finished')

#
# save_path = '/home/admin/xkc/RCMCGAN_pytorch/'
# log_path = '/home/admin/xkc/RCMCGAN_pytorch/logAE/log5/10/400'
#
# generate_infmap(log_path, save_path, step=400)