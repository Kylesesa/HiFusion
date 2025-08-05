import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

def sliding_window_crop(image, crop_size, x_slide, y_slide):
    img_H, img_W = image.shape[:2]
    crop_H, crop_W = crop_size
    step_H = y_slide
    step_W = x_slide

    crops = []
    for y in range(0, img_H - crop_H+1, step_H):
        for x in range(0, img_W - crop_W+1, step_W):
            crop = image[y:y+crop_H, x:x+crop_W]
            crops.append(crop)

    return crops

# rgb_image = Image.open('/home/admin/xkc/RCMCGAN_pytorch/TNO_bmp/VIS10.bmp')
# transform = transforms.Compose([transforms.Grayscale(num_output_channels=1)  # 将RGB图像转换为单通道的灰度图像
# ])
# gray_image = transform(rgb_image)
#
# #显示灰度图像
# gray_image.save('/home/admin/xkc/RCMCGAN_pytorch/TNO_bmp/VIS10.bmp')

def RGB2Y(path, save_path):
    for filename in os.listdir(path):
        if not filename.startswith('.'):
            img_path = os.path.join(path,filename)
            img_save_path = os.path.join(save_path,filename)
            os.makedirs(save_path,exist_ok=True)
            rgb_img = Image.open(img_path)

            ycrcb_image = rgb_img.convert("YCbCr")

            y_channel = ycrcb_image.split()[0]

            y_channel.save(img_save_path)

            print(f'{filename} has been finished')
# RGB2Y(path='/home/admin/xkc/Medical/PET-MRI/train/PET',save_path='/home/admin/xkc/Medical/PET-MRI/train/PET_Y')

def RGB2Gray(path, save_path):
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            img_path = os.path.join(path,filename)
            img_save_path = os.path.join(save_path,filename)
            os.makedirs(save_path,exist_ok=True)
            rgb_img = Image.open(img_path)

            if rgb_img.mode !='L':
                rgb_img = rgb_img.convert('L')
                rgb_img.save(img_save_path)

            print(f'{filename} has been finished')
# RGB2Gray(path='/home/admin/xkc/RCMCGAN_pytorch/log/log_c4/5/1000',save_path='/home/admin/xkc/RCMCGAN_pytorch/log/log_c4/5/1000')
