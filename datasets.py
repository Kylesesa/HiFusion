from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import h5py
import numpy as np
from PIL import Image
from utils import *
import matplotlib.pyplot as plt

loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()



datapath1 = 'vis_ir_dataset64.h5'
# datapath1 = 'Training_Dataset.h5'

datapath2 = '/home/admin/xkc/RCMCGAN_pytorch/数据集/MSRS-main/train/'
datapath2_vi = f'{datapath2}vis_gray'
datapath2_ir = f'{datapath2}Infrared2'

datapath3 = '/home/admin/xkc/RCMCGAN_pytorch/RoadScene/'
datapath3_vi = f'{datapath3}crop_HR_visible'
datapath3_ir = f'{datapath3}ir'

datapath4 = '/home/admin/xkc/RCMCGAN_pytorch/数据集/TNO_cropped/'
datapath4_vi = f'{datapath4}vi'
datapath4_ir = f'{datapath4}ir'

datapath = '/home/admin/xkc/RCMCGAN_pytorch/数据集/'
datapath_vi = f'{datapath}VIS'
datapath_ir = f'{datapath}IR'

datapath_pet='/home/admin/xkc/Medical/PET-MRI/train/PET_Y'
datapath_mri='/home/admin/xkc/Medical/PET-MRI/train/MRI'

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyDataset(Dataset):
    '''
    用于加载TNO.h5文件
    '''

    def __init__(self, file_path):
        self.file_path = file_path

        with h5py.File(file_path, 'r') as hf:
            self.data = hf['data'][:]  # (22912,2,84,84)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        x = torch.tensor(x)
        # print('1:', x.shape)
        # vis = x[0,:,:]
        # ir = x[1,:,:]
        return x

mydatasets = MyDataset(datapath1)

class MSRSDataset(Dataset):
    '''
    用于加载MSRS数据集
    '''

    def __init__(self, visible_root, infrared_root):
        self.visible_root = visible_root
        self.infrared_root = infrared_root
        self.visible_files = os.listdir(visible_root)
        self.infrared_files = os.listdir(infrared_root)

    def __len__(self):
        return min(len(self.visible_files), len(self.infrared_files))

    def __getitem__(self, index):
        visible_img = Image.open(os.path.join(self.visible_root, self.visible_files[index]))
        infrared_img = Image.open(os.path.join(self.infrared_root, self.infrared_files[index]))
        visible_img = loader(visible_img).unsqueeze(0)  # (1,3,480,640)
        infrared_img = loader(infrared_img).unsqueeze(0)  # (1,1,480,640)

        # vis, Cr, Cb = RGB2YCrCb(visible_img)  # (1,1,480,640) #用了转换后的灰度图，这一步就不用
        # ir = infrared_img  # (1,1,480,640)
        vis = visible_img.squeeze(0)  # (1,480,640)
        ir = infrared_img.squeeze(0)  # (1,480,640)因为在dataloader的时候，batch维度会自动补上，所以把这一维度去掉，否则会多一个维度
        # x = torch.cat((vis, ir), dim=0)
        return vis,ir
msrsdatasets = MSRSDataset(datapath2_vi, datapath2_ir)

class RoadSceneDataset(Dataset):
    '''
    用于加载RoadScene数据集
    '''

    def __init__(self, visible_root, infrared_root):
        self.visible_root = visible_root
        self.infrared_root = infrared_root
        self.visible_files = os.listdir(visible_root)
        self.infrared_files = os.listdir(infrared_root)

    def __len__(self):
        return min(len(self.visible_files), len(self.infrared_files))

    def __getitem__(self, index):
        visible_img = Image.open(os.path.join(self.visible_root, self.visible_files[index]))
        infrared_img = Image.open(os.path.join(self.infrared_root, self.infrared_files[index]))
        visible_img = loader(visible_img).unsqueeze(0)  # (1,3,480,640)
        infrared_img = loader(infrared_img).unsqueeze(0)  # (1,1,480,640)

        vis, Cr, Cb = RGB2YCrCb(visible_img)  # (1,1,480,640)
        ir = infrared_img  # (1,1,480,640)
        vis = vis.squeeze(0)  # (1,480,640)
        ir = ir.squeeze(0)  # (1,480,640)因为在dataloader的时候，batch维度会自动补上，所以把这一维度去掉，否则会多一个维度
        # x = torch.cat((vis, ir), dim=0)
        return vis,ir


# datasets1 = MyDataset(datapath1)
# datasets2 = MSRSDataset(datapath2_vi, datapath2_ir)
# datasets3 = RoadSceneDataset(datapath3_vi, datapath3_ir)


class CombinedDataset(Dataset):
    '''
    用于加载自制数据集'Road+MSRS'
    '''
    def __init__(self, visible_root, infrared_root):
        self.visible_root = visible_root
        self.infrared_root = infrared_root
        self.visible_files = os.listdir(visible_root)
        self.infrared_files = os.listdir(infrared_root)

    def __len__(self):
        return min(len(self.visible_files), len(self.infrared_files))

    def __getitem__(self, index):
        visible_img = Image.open(os.path.join(self.visible_root, self.visible_files[index]))
        infrared_img = Image.open(os.path.join(self.infrared_root, self.infrared_files[index]))
        visible_img = loader(visible_img).unsqueeze(0)  # (1,1,224,224)
        infrared_img = loader(infrared_img).unsqueeze(0)  # (1,1,224,224)

        vis = visible_img.squeeze(0)  # (1,224,224)
        ir = infrared_img.squeeze(0)  # (1,224,224)因为在dataloader的时候，batch维度会自动补上，所以把这一维度去掉，否则会多一个维度
        # x = torch.cat((vis, ir), dim=0)
        # print(index)
        return vis,ir
datasets = CombinedDataset(datapath_vi, datapath_ir)

class PET_MRI_Dataset(Dataset):
    def __init__(self, pet_root, mri_root):
        self.pet_root = pet_root
        self.mri_root = mri_root
        self.pet_files = os.listdir(pet_root)
        self.mri_files = os.listdir(mri_root)

    def __len__(self):
        return min(len(self.pet_files), len(self.mri_files))

    def __getitem__(self, index):
        pet_img = Image.open(os.path.join(self.pet_root, self.pet_files[index]))
        mri_img = Image.open(os.path.join(self.mri_root, self.mri_files[index]))
        pet_img = loader(pet_img).unsqueeze(0)  # (1,1,224,224)
        mri_img = loader(mri_img).unsqueeze(0)  # (1,1,224,224)

        pet = pet_img.squeeze(0)  # (1,224,224)
        mri = mri_img.squeeze(0)  # (1,224,224)因为在dataloader的时候，batch维度会自动补上，所以把这一维度去掉，否则会多一个维度
        # x = torch.cat((vis, ir), dim=0)
        # print(index)
        return pet,mri
# pet_mri_dataset = PET_MRI_Dataset(datapath_pet, datapath_mri)

class TNODataset(Dataset):
    '''
    用于加载自制数据集'TNO_cropped'
    '''
    def __init__(self, visible_root, infrared_root):
        self.visible_root = visible_root
        self.infrared_root = infrared_root
        self.visible_files = os.listdir(visible_root)
        self.infrared_files = os.listdir(infrared_root)

    def __len__(self):
        return min(len(self.visible_files), len(self.infrared_files))

    def __getitem__(self, index):
        visible_img = Image.open(os.path.join(self.visible_root, self.visible_files[index]))
        infrared_img = Image.open(os.path.join(self.infrared_root, self.infrared_files[index]))
        visible_img = loader(visible_img).unsqueeze(0)  # (1,1,224,224)
        infrared_img = loader(infrared_img).unsqueeze(0)  # (1,1,224,224)


        vis = visible_img.squeeze(0)  # (1,224,224)
        ir = infrared_img.squeeze(0)  # (1,224,224)因为在dataloader的时候，batch维度会自动补上，所以把这一维度去掉，否则会多一个维度
        # x = torch.cat((vis, ir), dim=0)
        return vis,ir
# datasets_tno = TNODataset(datapath4_vi, datapath4_ir)

# for i in range(len(datasets)):
#     element = datasets[i]
#     print(element.shape)

def image_save(tensor, name):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(name)


def image_loader(image_name, gpu):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    image = image.to('cuda:' + str(gpu), torch.float32)

    if image.shape[1] > 1:
        image = image[:, 0, :, :]
        image = torch.reshape(image, (1, 1, shape, shape))
        return image


def load_train_data(path, batch, batch_size, gpu):
    dirname = os.listdir(path)
    imgname = []
    for i in dirname:
        img = os.listdir(path + i)
        img = [path + i + '/' + img[0], path + i + '/' + img[1]]
        imgname.append(img)

    for i in range(batch * batch_size, min(len(imgname), (batch + 1) * batch_size)):
        train_data1 = torch.empty(1, 1, 256, 256)
        train_data2 = torch.empty(1, 1, 64, 64)

        if i == batch * batch_size:
            train_data1 = image_loader(imgname[i][0], gpu, 256)
            train_data2 = image_loader(imgname[i][1], gpu, 64)
        else:
            data1 = image_loader(imgname[i][0], gpu, 256)
            data2 = image_loader(imgname[i][1], gpu, 64)
            train_data1 = torch.cat((train_data1, data1), 0)
            train_data2 = torch.cat((train_data2, data2), 0)
        return train_data1, train_data2, len(imgname)


def getdata(datapath, device):
    img = Image.open(datapath)
    img = loader(img).unsqueeze(0)
    image = img.to(device)

    if image.shape[1] > 1:
        Y, Cr, Cb = RGB2YCrCb(image)

        return Y, Cr, Cb
