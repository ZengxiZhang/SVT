import torch
import numpy as np
import cv2
import time
import torch.nn as nn
def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output * mask
def rgb2ycrcb(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    # cr = (r - y) * 0.713 + 128
    # cb = (b - y) * 0.564 + 128
    cr = ((r - y)*255 * 0.713 + 128)/255
    cb = ((b - y)*255 * 0.564 + 128)/255
    ycrcb = torch.zeros_like(rgb)
    ycrcb[:, 0], ycrcb[:, 1], ycrcb[:, 2] = y, cr, cb
    return ycrcb
def ycrcbtorgb(ycrcb_tensor):
    # 转换为Tensor并分离通道
    # print(ycrcb_tensor.shape)
    ycrcb_tensor=ycrcb_tensor*255
    y_channel, cr_channel, cb_channel = torch.split(ycrcb_tensor, 1, dim=1)
    
    # 进行YCrCb到RGB的转换
    r_channel = y_channel + 1.402 * (cr_channel - 128)
    g_channel = y_channel - 0.344136 * (cb_channel - 128) - 0.714136 * (cr_channel - 128)
    b_channel = y_channel + 1.772 * (cb_channel - 128)
    
    # 合并通道并转换为RGB图像
    rgb_tensor = torch.cat([r_channel, g_channel, b_channel], dim=1)
    rgb_tensor = torch.clamp(rgb_tensor, 0, 255)  # 确保像素值在0到255之间
    # rgb_image = rgb_tensor.squeeze(0).permute(1, 2, 0).numpy().astype('uint8')
    rgb_tensor=rgb_tensor/255.
    return rgb_tensor
#第一个是h4p
def DLT_solve(src_p, off_set):
    # print("haha")
    # print(src_p.shape)
    # print(src_p)
    # # print(src_p.device)
    # # print(off_set.shape)
    # print(src_p.shape)
    # print(off_set)
    # src_p: shape=(bs, n, 4, 2)
    # off_set: shape=(bs, n, 4, 2)
    # can be used to compute mesh points (multi-H)
    bs, _ = src_p.shape
    # print(len(src_p[0]))
    #print(np.sqrt(len(src_p[0])))#len(src_p[0])=8,sqrt是开方（float）
    divide = int(np.sqrt(len(src_p[0])/2)-1)# divide=1 
    # print(divide)
    row_num = (divide+1)*2# row_num = 4，可能是看几边形吧

    for i in range(divide):
        for j in range(divide):
            # print(src_p)
            h4p = src_p[:,[ 2*j + row_num*i, 2*j + row_num*i + 1, 
                    2*(j+1) + row_num*i, 2*(j+1) + row_num*i + 1, 
                    2*(j+1) + row_num*i + row_num, 2*(j+1) + row_num*i + row_num + 1,
                    2*j + row_num*i + row_num, 2*j + row_num*i + row_num+1]].reshape(bs, 1, 4, 2)  
            # print(h4p)
            
            pred_h4p = off_set[:,[2*j+row_num*i, 2*j+row_num*i+1, 
                    2*(j+1)+row_num*i, 2*(j+1)+row_num*i+1, 
                    2*(j+1)+row_num*i+row_num, 2*(j+1)+row_num*i+row_num+1,
                    2*j+row_num*i+row_num, 2*j+row_num*i+row_num+1]].reshape(bs, 1, 4, 2)

            if i+j==0:
                src_ps = h4p
                off_sets = pred_h4p
            else:
                src_ps = torch.cat((src_ps, h4p), axis = 1)    
                off_sets = torch.cat((off_sets, pred_h4p), axis = 1)

    bs, n, h, w = src_ps.shape

    N = bs*n #1*1=1

    src_ps = src_ps.reshape(N, h, w)#(1,4,2)
    off_sets = off_sets.reshape(N, h, w)#(1,4,2)

    dst_p = src_ps + off_sets# 直接加偏移量,新的图像四边形
    # print(dst_p)
    ones = torch.ones(N, 4, 1) #(1,4,1)
    if torch.cuda.is_available():
        ones = ones.cuda()
    xy1 = torch.cat((src_ps, ones), 2)#(1,4,3) 
    # print(xy1.shape)
    zeros = torch.zeros_like(xy1)#(1,4,3)
    if torch.cuda.is_available():
        zeros = zeros.cuda()

    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)#(1,4,6)
    # print(xyu.shape)
    M1 = torch.cat((xyu, xyd), 2).reshape(N, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1), 
        src_ps.reshape(-1, 1, 2),
    ).reshape(N, -1, 2)

    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(N, -1, 1)

    Ainv = torch.inverse(A)
    h8 = torch.matmul(Ainv, b).reshape(N, 8)
 
    H = torch.cat((h8, ones[:,0,:]), 1).reshape(N, 3, 3)
    H = H.reshape(bs, n, 3, 3)
    
    # print(H.shape)
    return H

def coords_grid(batch, ht, wd, normalize=False):
    if normalize:  # [-1, 1]
        coords = torch.meshgrid(2 * torch.arange(ht) / (ht - 1) - 1,
                                2 * torch.arange(wd) / (wd - 1) - 1)
    else:
        coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)  # [B, 2, H, W]
