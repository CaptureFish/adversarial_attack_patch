from __future__ import division
import sys
import torch
sys.path.append('PyTorchYOLOv3/')
from models import *
from utils.utils import *
from utils.datasets import *

class MyDetectorYolov3():
    def __init__(self,cfgfile=None,weightfile=None):
        if cfgfile==None or weightfile==None:
            print('need configfile or weightfile')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=Darknet(cfgfile).to(self.device)
        self.model.load_darknet_weights(weightfile)
        print('Loading Yolov3 weights from %s... Done!' % (weightfile))

    def attack(self,input_imgs,attack_id=[0],total_cls=85,object_thres=0.1,clear_imgs=None,compare_imgs=None,img_size=None):
        for index in attack_id:
            index+=5
        input_imgs = F.interpolate(input_imgs, size=img_size).to(self.device)
        self.model.eval()
        detections = self.model(input_imgs) #v3tiny:torch.Size([B, 2535, 85]), v3:torch.Size([B, 10647, 85])
        batch=detections.shape[0]
        boxes =detections.shape[1]
        assert total_cls+5==detections.shape[2]
        detections = detections.view(batch*boxes,(total_cls+5))#([B* 2535, 85]), v3:torch.Size([B*10647, 85])
        detections =  detections[detections[:,4] >= object_thres] # [x ,85] 
        objectness = detections[:,4:5] # [x,1]
        
        classness = detections[:,5:] #[x,80]
        classness = torch.nn.Softmax(dim=1)(classness) #[x,80]
        attack_classness =detections[:,attack_id] #[x,3] assuming tank airplane and armored vehicles 
        confs = torch.mul(attack_classness,objectness) #[x,3]
        # 至少存在一种吗 也不一定 或者不用topk 直接全选
        if confs.shape[0]>=3:
            confs, _= confs.topk(3,dim=0) 
        elif confs.shape[0]>=1:
            confs, _= confs.topk(1,dim=0) 

        if not clear_imgs == None:
            pass 
        if not compare_imgs ==None:
            pass 
        return torch.mean(confs)
