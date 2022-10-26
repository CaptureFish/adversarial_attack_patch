import numpy as np
import os
import sys
from sqlalchemy import false
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import time
from tqdm import tqdm
from torch import autograd
from ensemble_tool.utils import *
from ensemble_tool.model import  TotalVariation
import random

from PyTorchYOLOv3.attack_detector import MyDetectorYolov3
# from pytorch_pretrained_detection import FasterrcnnResnet50, MaskrcnnResnet50
from pytorchYOLOv4.demo import DetectorYolov4
from load_data import InriaDataset, PatchTransformer, PatchApplier
from pathlib import Path
from ipdb import set_trace as st
import argparse
import sys


def parser_opt(known=False):
    Gparser = argparse.ArgumentParser(description='Advpatch Training')
    Gparser.add_argument('--model', default='yolov3', type=str, help='options : yolov2, yolov3, yolov4, fasterrcnn')
    Gparser.add_argument('-generator',default='gray',type=str, help='options : biggan,stylegan')
    return Gparser.parse_known_args()[0] if known else Gparser.parse_args()

def run(opt):
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda or cpu
    train_image_size   = 416
    train_batch_size      = 8
    train_loader = torch.utils.data.DataLoader(InriaDataset(img_dir='/data1/yjt/mydatasets/images/val/', 
                                                            lab_dir='/data1/yjt/mydatasets/labels/val/', 
                                                            max_lab=50,
                                                            imgsize=train_image_size,
                                                            shuffle=True),
                                                batch_size=train_batch_size,
                                                shuffle=True,
                                                num_workers=10)
    train_loader = DeviceDataLoader(train_loader, device)
    # 检测器
    detector=chooseDector(opt)
    # 生成patch 
    adv_patch_cpu = generate_patch(opt)
    adv_patch_cpu.requires_grad_(True)
    # patch 应用器 增强器 和 TV计算器
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()
    total_variation = TotalVariation().cuda()
    # 优化器
    learning_rate = 0.005
    # 只要这里指定优化的参数就够了 
    optimizer = torch.optim.Adam([adv_patch_cpu], lr=learning_rate, betas=(0.5, 0.999), amsgrad=True)
    optimizer_sc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)

    # 这一部分保留 暂时不动
    # global_dir = increment_path(Path('./exp') / 'exp', exist_ok=False) # 'checkpoint'
    # global_dir = Path(global_dir)
    # checkpoint_dir = global_dir / 'checkpoint'
    # checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # sample_dir = global_dir / 'generated'
    # sample_dir.mkdir(parents=True, exist_ok=True)
    # print(f"\n##### The results are saved at {global_dir}. #######\n")
    # 开始训练
    iteration_total = len(train_loader)
    torch.cuda.empty_cache()
    start_epoch=1
    n_epochs=3
    for epoch in range(start_epoch, n_epochs+1):
        for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',total=iteration_total):
            with autograd.detect_anomaly():
                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()
                adv_patch = adv_patch_cpu.cuda()
                adv_batch_t = patch_transformer(adv_patch, lab_batch, train_image_size, do_rotate=True, rand_loc=False)[0]
                p_img_batch = patch_applier(img_batch, adv_batch_t)
                # print(p_img_batch.shape)
                # img = p_img_batch[1, :, :,]
                # img = transforms.ToPILImage()(img.detach().cpu().tanspose())
                # img.save('./patch.png',img)
                # 进行攻击
                loss_det = detector.attack(input_imgs=p_img_batch, attack_id=[0,1,2], total_cls=80,object_thres=0.1,clear_imgs=img_batch,img_size=416)
                loss = loss_det
                loss.backward()
                optimizer.step() 
                optimizer_sc.step(loss)
                optimizer.zero_grad()
                adv_patch_cpu.data.clamp_(0,1) #保证图像范围 暂不添加其他的损失函数
                # print(loss.item())  损失函数用tensorboard 写 
                del adv_batch_t,loss_det, p_img_batch, loss
                torch.cuda.empty_cache()
        # epoch结束 
        print('\n')
        patch_save = np.clip(np.transpose(adv_patch_cpu.detach().numpy(), (1, 2, 0)), 0, 1)
        patch_save = Image.fromarray(np.uint8(255*patch_save))
        patch_save.save("training_patches/" + str(epoch) + " patch.png")
        # 查看损失？yolo detect map的结果，直接通过执行来查看，损失函数的下降如何判断
        
def chooseDector(opt):
    """
    每个检测器存在一个attack攻击方法 用于返回 攻击之后的损失函数 
    """
    if opt.model=='yolov3':
        cfgfile = '/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv3/config/yolov3.cfg'
        weightfile='/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv3/weights/yolov3.weights'
        detector=MyDetectorYolov3(cfgfile,weightfile)
        return detector 
    elif opt.model=='yolov5':
        pass # todo 
    elif opt.model=='yolov4':
        pass #todo 
    elif opt.model=='yolov7':
        pass #todo 
    elif opt.model=='fastercnn':
        pass 
    elif opt.model=='ssd':
        pass 

def chooseTransformation():
    enable_rotation       = False
    enable_randomLocation = False
    enable_crease         = False
    enable_projection     = False
    enable_rectOccluding  = False
    enable_blurred = False 

def generate_patch(opt):
    if opt.generator == 'gray':
        adv_patch_cpu = torch.full((3,50,50), 0.5)
        return adv_patch_cpu
    elif opt.generator == 'random':
        adv_patch_cpu = torch.rand((3,50,50))
        return adv_patch_cpu
    elif opt.generator == 'biggan':
        return None 

if __name__ =='__main__':
    opt=parser_opt(known=True)
    run(opt)


