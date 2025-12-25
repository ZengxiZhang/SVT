import os
import sys
import cv2
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import numpy as np
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
NPY_EXTENSIONS = ['.npy']

import time
def resize2tensor(x: str, reshape: bool = False, size: tuple = (), channel: int = 1):
    x = Image.open(x)
    if channel == 3:
        if reshape:
            x = x.resize(size).convert('RGB')
        tensor = transforms.Compose([transforms.ToTensor()])
    else:
        if reshape:
            x = x.resize(size)
        tensor = transforms.Compose([transforms.ToTensor()])
    x = tensor(x)
    return x


def is_image(path):
    return any(path.endswith(t) for t in IMG_EXTENSIONS)
def is_npy(path):
    return any(path.endswith(t) for t in NPY_EXTENSIONS)



class Image_stitch(data.Dataset):
    #"数据集"

    # def __init__(self, ir1_path: str, ir2_path: str, vis1_path: str, vis2_path: str, gt_path: str,mode='mix'):
    def __init__(self, ir1_path: str, vis1_path: str, vis2_path: str,homo_gt_path:str,tps_gt_path:str):
        super(Image_stitch, self).__init__()

       

        # self.ir1_path, self.ir2_path ,self.vis1_path, self.vis2_path, self.gt_path= ir1_path, ir2_path, vis1_path, vis2_path, gt_path
        self.ir1_path, self.vis1_path, self.vis2_path, self.homo_gt_path, self.tps_gt_path = ir1_path, vis1_path, vis2_path, homo_gt_path, tps_gt_path
        self.ir1s = sorted([x for x in os.listdir(self.ir1_path) if is_image(x)])
        self.vis1s = sorted([x for x in os.listdir(self.vis1_path) if is_image(x)])
        self.vis2s = sorted([x for x in os.listdir(self.vis2_path) if is_image(x)])
        self.homos = sorted([x for x in os.listdir(self.homo_gt_path) if is_npy(x)])
        self.tpss = sorted([x for x in os.listdir(self.tps_gt_path) if is_npy(x)])

        try:
            if len(self.ir1s) != len(self.vis1s) and len(self.vis1s) != len(self.vis2s):
                sys.exit(0)
            # for i in range(len(self.ir1s)):
            #     if self.ir1s[i] != self.ir2s[i]:
            #         sys.exit(0)
            # for i in range(len(self.vis1s)):
            #     if self.vis1s[i] != self.vis2s[i]:
            #         sys.exit(0)
        except:
            print("[Src Image] and [Sal Image] don't match.")

    def __getitem__(self, index):
        name=self.ir1s[index]
        gt_name=self.homos[index]
        ir1 = cv2.imread(os.path.join(self.ir1_path, name))
        vis1 = cv2.imread(os.path.join(self.vis1_path, name))
        vis2 = cv2.imread(os.path.join(self.vis2_path, name))
        gt = np.zeros((2,2,2))
        shift=np.load(os.path.join(self.homo_gt_path, gt_name))
        if shift.shape[0]==4:
            shift = shift.reshape(8,1)
        gt[:, 0, 0] = shift[:2, 0]
        gt[:, 0, 1] = shift[2:4, 0]
        gt[:, 1, 0] = shift[4:6, 0]
        gt[:, 1, 1] = shift[6:8, 0]
        tps=np.load(os.path.join(self.tps_gt_path, gt_name))
        # print(ir1.shape)
        # tps = cv2.resize(tps, (128, 128))
        tps = np.transpose(tps, (2,0,1))
        # gt_fus = cv2.imread(os.path.join(self.gt_fus_path, name))
        # irlc1 = cv2.imread(os.path.join(self.irlc_path, name))
        # vislc1 = cv2.imread(os.path.join(self.vislc_path, name))
        # shift=np.reshape(shift,(4,2))
        ir1 = cv2.resize(ir1,(256,256))
        vis1 = cv2.resize(vis1,(256,256))
        vis2 = cv2.resize(vis2,(256,256))
        height = ir1.shape[0] 
        width = ir1.shape[1]  
        size = np.array([width, height], dtype=np.float32)
        size=np.expand_dims(size, 1)
        

        ir1= (cv2.cvtColor(ir1, cv2.COLOR_BGR2GRAY)/255.)[...,None]
        vis1= (cv2.cvtColor(vis1, cv2.COLOR_BGR2GRAY)/255.)[...,None]
        vis2= (cv2.cvtColor(vis2, cv2.COLOR_BGR2GRAY)/255.)[...,None]
        

        return ir1,vis1,vis2,gt,tps#,shift#,ir1gray,vis1gray, irlc1, vislc1, shift#, name

    def __len__(self):
        return len(self.vis1s)




