import os
from models import disjoint_augment_image_pair#,H_estimator
# from H_model_mini import H_estimator
from H_model import H_estimator,Encoder,Decoder
# from H_model_detone import H_estimator
import torch.nn as nn
import numpy as np
import torch
import cv2
from loss import cal_lp_loss, inter_grid_loss, intra_grid_loss
from dataset import Image_stitch
import time
import genotypes 
# from visdom import Visdom
from transfer.network import ImageTransformNet
from ssim  import SSIM
# viz = Visdom(env='stitch-pytorch')  # 启用可视化工具
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# def environment_check():
#     if torch.cuda.is_available():
#         os.system('gpustat')
#         i = int(input("choose devise:"))

#         if i != -1:
#             torch.cuda.set_device(device = i)
#             return i
#     print("cuda: False")
#     return 'cpu'
# device = environment_check()
device = 'cuda:0'
if device != 'cpu':
    use_cuda = True
else:
    use_cuda = False
vis_batch = 200# for checking loss during training 

dataset_mode = 'roadscene' 
# learning_rate = 0.0001
height, width = 128, 128


data_root = '../tps_registration_roadscene/test'

netR_path = 'snapshot/100_R.pkl'
netT_path = 'transfer/models_roadscene/1310.pkl'
netE_path = 'snapshot/100_E.pkl'
netD_path = 'snapshot/100_D.pkl'


gene_mix = [  ('dil_conv_5x5', 0), ('conv_5x5', 1), ('conv_3x3', 2),  
			  ('conv_3x3', 0), ('conv_5x5', 1), ('dil_conv_3x3', 2),
			  ('conv_5x5', 0), ('conv_5x5', 1), ('conv_3x3', 2), 
			  ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2)]

data=Image_stitch(ir1_path=os.path.join(data_root,'ir_input1'),\
                  vis1_path=os.path.join(data_root,'vis_input1'),\
                  vis2_path=os.path.join(data_root,'vis_input2'),
                  homo_gt_path=os.path.join(data_root,'homo_shift'),
                  tps_gt_path=os.path.join(data_root,'elastic_disp'))
dataloader = torch.utils.data.DataLoader(data, batch_size=1,shuffle=False,num_workers=0,pin_memory=True)


image_transformer = ImageTransformNet().type(torch.cuda.FloatTensor)
netE = Encoder(batch_size=batch_size,device=device,is_training=False,geno_mix=gene_mix)
netR = H_estimator(batch_size=batch_size,device=device,is_training=True,geno_mix=gene_mix)
# netD = Decoder(batch_size=batch_size,device=device,is_training=False,geno_mix=gene_mix)
if netR_path is not None:
    netR.load_state_dict(torch.load(netR_path, map_location='cpu'))
if netE_path is not None:
    netE.load_state_dict(torch.load(netE_path, map_location='cpu'))
if netT_path is not None:
    image_transformer.load_state_dict(torch.load(netT_path, map_location='cpu'))

if use_cuda:
    l1loss = l1loss.to(device)
    l2loss = l2loss.to(device)
    netE = netE.to(device)
    # netD = netD.to(device)
    netR = netR.to(device)
    ssim=ssim.to(device)
    # netH = netH.to(device)

save_folder = 'snapshot'
# define dataset
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
loss_all_batch = 0

netR.eval()
netE.eval()
image_transformer.eval()
for epoch in range(0,num_epochs+1):
    for i,(ir_input1,vis_input1,vis_input2,homo,tps) in enumerate(dataloader):
        ir_input1=ir_input1.float()
        vis_input1=vis_input1.float()
        vis_input2=vis_input2.float()
        homo=homo.float()

        if use_cuda:
            ir_input1=ir_input1.to(device)
            # train_ir_inputs_aug=train_ir_inputs_aug.to(device)
            vis_input1 = vis_input1.to(device)
            vis_input2 = vis_input2.to(device)
            homo = homo.to(device)
            tps = tps.to(device)
        ir_input2 = image_transformer(vis_input2.permute(0,3,1,2)).permute(0,2,3,1)
        train_ir_inputs_aug = disjoint_augment_image_pair(ir_input1,ir_input2)
        ir_input1_aug = train_ir_inputs_aug[...,0][...,None].permute(0,3,1,2)
        ir_input2_aug = train_ir_inputs_aug[...,1][...,None].permute(0,3,1,2) 
        ir_input1 = torch.nn.functional.interpolate(ir_input1.permute(0,3,1,2), [256,256])
        ir_input2 = torch.nn.functional.interpolate(ir_input2.permute(0,3,1,2), [256,256])   
        vis_input1 = torch.nn.functional.interpolate(vis_input1.permute(0,3,1,2), [256,256])
        vis_input2 = torch.nn.functional.interpolate(vis_input2.permute(0,3,1,2), [256,256])   

        train_ir_inputs = torch.cat((ir_input1, ir_input2), 1)

        # ir_en1, ir_en2, vis_en1, vis_en2 = netE(ir_input1,ir_input2, vis_input1,vis_input2)
        ir_en1, ir_en2 = netE(ir_input1_aug,ir_input2_aug)        # off1, off2, off3, ir_warp1, ir_warp2, vis_warp1, vis_warp2= netR(ir_en1,ir_en2,vis_en1,vis_en2,ir_input1,ir_input2,vis_input1,vis_input2,gt)
        off1, off2, off3, ir_warp1, vis_warp1, vis_warp2, tps_gt = netR(ir_en1,ir_en2,ir_input1,vis_input1,vis_input2,homo=homo,tps=tps)
        output = vis_warp2[-1]

 

