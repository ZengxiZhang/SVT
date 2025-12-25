import numpy as np
import torch
import os
import argparse
import time

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from dataset import Image_style_test
import utils
from network import ImageTransformNet
from vgg import Vgg16
import cv2
# Global Variables
net_path='models/1250.pkl'
dataset='/data/zhangzengxi/RoadScene-rgb/'
dtype = torch.cuda.FloatTensor
image_transformer = ImageTransformNet().type(dtype)
if net_path is not None:
    image_transformer.load_state_dict(torch.load(net_path,map_location='cpu'))

test_loader = Image_style_test(vis_path=os.path.join(dataset,'VIS'))
test_loader = DataLoader(test_loader, batch_size = 1)



# image_transformer.eval()
image_transformer.eval()
for batch_num, (x, name) in enumerate(test_loader):
    img_batch_read = len(x)
    name=name[0]
    x=x.cuda()
    x = Variable(x).type(dtype)
    y_hat = image_transformer(x)
    
    y=y_hat[0,0].cpu().detach().numpy()*255
    print(name)
    print(os.path.join('output',name))
    cv2.imwrite(os.path.join('output',name),y)