import numpy as np
# from tensorDLT import solve_DLT
# from tensorDLT import tensor_DLT
# from tensorDLT_function import solve_DLT
# from tf_spatial_transform import transform
import torchgeometry as tgm
# from spatial_transform import Transform
# import utils.torch_homo_transform as torch_homo_transform
import utils_tps.torch_tps_transform as torch_tps_transform
from output_spatial_transform import Transform_output
import torch.nn as nn
# from output_py_spatial_transform import Transform_output
import torch.nn.functional as F
import torch
import cv2
from layers import predict_flow
from kornia.geometry.transform import warp_points_tps
from kornia.utils import create_meshgrid
from output_tensorDLT import output_solve_DLT
import torch.nn as nn
from operations import *
from attention import Attention1D
from position import PositionEmbeddingSine
from correlation import Correlation1D
# from torchvision.transforms import Resize
import grid_res
import kornia
from utils import coords_grid,warp
import time
from genotypes import *
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W
def get_rigid_mesh(batch_size, height, width):

    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

class CNN_32(nn.Module):
    def __init__(self, input_dim=256):
        super(CNN_32, self).__init__()
        
        outputdim = input_dim
        self.layer1 = nn.Sequential(nn.Conv2d(128, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim_final = outputdim
        # global motion
        self.layer10 = nn.Sequential(nn.Conv2d(input_dim, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer10(x)
        return x
class CNN_64(nn.Module):
    def __init__(self, input_dim=256):
        super(CNN_64, self).__init__()

        outputdim = input_dim
        self.layer1 = nn.Sequential(nn.Conv2d(128, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = input_dim
        outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = input_dim
        outputdim = input_dim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = input_dim
        outputdim = input_dim
        self.layer5 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        outputdim_final = outputdim
        # global motion
        self.layer10 = nn.Sequential(nn.Conv2d(outputdim_final, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer10(x)
        return x

class GMA_update(nn.Module):
    # def __init__(self, args, sz):
    def __init__(self, sz):
        super().__init__()
        # self.args = args
        # if sz==32:
        #     self.cnn = CNN_32(80)
        if sz==64:
            self.cnn = CNN_64(64)
        # if sz==16:
            # self.cnn = CNN_16(80)
        # if sz==32:
        #     self.cnn = CNN_32(64)
        if sz==32:
            self.cnn = CNN_32(80)
            
    def forward(self, corr_flow):   
        # print(corr_flow.shape)   
        delta_flow = self.cnn(corr_flow)
        # print(delta_flow.shape)
        # time.sleep(1000)   
        return delta_flow

class Get_Flow(nn.Module):
    def __init__(self, sz):
        super().__init__()
        self.sz = sz

    def forward(self, four_point, a):
        four_point = four_point/ torch.Tensor([a]).cuda()

        four_point_org = torch.zeros((2, 2, 2)).cuda()

        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([self.sz[3]-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, self.sz[2]-1])
        four_point_org[:, 1, 1] = torch.Tensor([self.sz[3]-1, self.sz[2]-1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(self.sz[0], 1, 1, 1)

        # four_point_new = four_point_org + four_point
        four_point_new = four_point_org + four_point
        # print(four_point_org)
        # print(four_point_new)
        # time.sleep(1000) 
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
        # H = tgm.get_perspective_transform(four_point_org, four_point_new)
        H = tgm.get_perspective_transform(four_point_new, four_point_org)
        gridy, gridx = torch.meshgrid(torch.linspace(0, self.sz[3]-1, steps=self.sz[3]), torch.linspace(0, self.sz[2]-1, steps=self.sz[2]))
        points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, self.sz[3] * self.sz[2]))),
                           dim=0).unsqueeze(0).repeat(self.sz[0], 1, 1).to(four_point.device)
        points_new = H.bmm(points)
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1),
                          points_new[:, 1, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1)), dim=1)
        return flow


# normalize mesh from -1 ~ 1
def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2

    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2
def L1_norm(narry_a, narry_b):
    # caculate L1-norm
    temp_abs_a = torch.abs(narry_a)
    temp_abs_b = torch.abs(narry_b)
    l1_a = torch.sum(temp_abs_a,1)
    l1_b = torch.sum(temp_abs_b,1)
    mask_value = l1_a + l1_b
    array_MASK_a = torch.unsqueeze(l1_a/mask_value,1)
    array_MASK_b =  torch.unsqueeze(l1_b/mask_value,1)
    resule_torch = array_MASK_a*narry_a + array_MASK_b * narry_b
    return resule_torch

class Cell(nn.Module):
    def __init__(self, genotype, C=32, reduction=False, reduction_prev=False,mode='vis'):
        super(Cell, self).__init__()
        # print(C_prev_prev, C_prev, C)

        if mode == 'ir':
            op_names, indices = zip(*genotype.ir)
        elif mode == 'vis':
            op_names, indices = zip(*genotype.vis)
        elif mode == 'decoder':
            op_names, indices = zip(*genotype.decoder)
        concat = [0,2,4,6]
        self._compile(C,C, op_names, indices, concat, reduction)

    def _compile(self,C_in,C_out, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            # print(name)
            op = OPS[name](C_in,C_out, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0):
        op = self._ops[0]
        states = [s0]
        for i in range(7):
            states += [self._ops[i](states[self._indices[i]])]
        
        return torch.cat([states[i] for i in self._concat], dim=1)
        # for i in range(self._steps):
        #     print(self._indices)
        #     time.sleep(100)
        #     h1 = states[self._indices[2*i+1]]
        #     h2 = states[self._indices[2*i+1+1]]
        #     op1 = self._ops[2*i+1]
        #     op2 = self._ops[2*i+1+1]
        #     h1 = op1(h1)
        #     h2 = op2(h2)
        #     s = h1 + h2
        #     states += [s]
        # return torch.cat([states[i] for i in self._concat], dim=1)


class feature_extractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.resize=Resize((128,128))
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.conv1=torch.nn.Sequential( nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True),
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True)
                                        )
        # self.feature.append(conv1)
        self.maxpool1 = torch.nn.MaxPool2d((2,2), stride=2)
        self.conv2=torch.nn.Sequential( nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True),
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True))
        self.maxpool2 = torch.nn.MaxPool2d((2,2), stride=2)
        self.conv3=torch.nn.Sequential( nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(True),
                                        nn.BatchNorm2d(128),
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                        nn.ReLU(True))
        self.maxpool3 = torch.nn.MaxPool2d((2,2), stride=2)
        self.conv4 = torch.nn.Sequential( nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(True),
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(True))

        # end
    def forward(self, input):
        # input=self.resize(input)
        conv1 = self.conv1(input)
        conv1m = self.maxpool1(conv1)
        conv2 = self.conv2(conv1m)
        conv2m = self.maxpool1(conv2)
        conv3 = self.conv3(conv2m)
        conv3m = self.maxpool1(conv3)
        conv4 = self.conv4(conv3m)
        return conv1,conv2,conv3,conv4


def cost_volume(c1, warp, search_range,pad,lrelu):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = pad(warp)
    _,c ,h, w = c1.shape
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice=padded_lvl[:, :,y:y+h, x:x+w]
            cost = torch.mean(c1 * slice, axis=1, keepdims=True)
            cost_vol.append(cost)
    cost_vol = torch.cat(cost_vol, axis=1)
    cost_vol = lrelu(cost_vol)
    return cost_vol
def vertical_cost_volume(c1, warp, search_range,pad,lrelu):
    """Build vertical cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    print(warp.shape)
    padded_lvl = pad(warp)
    print(padded_lvl.shape)
    time.sleep(1000)
    _,c ,h, w = c1.shape
    max_offset = search_range * 2 + 1
    cost_vol = []
    for y in range(0, max_offset):
        slice=padded_lvl[:, :,y:y+h, :]
        cost = torch.mean(c1 * slice, axis=1, keepdims=True)
        cost_vol.append(cost)
    cost_vol = torch.cat(cost_vol, axis=1)
    cost_vol = lrelu(cost_vol)
    return cost_vol
def horizontal_cost_volume(c1, warp, search_range,pad,lrelu):
    """Build horizontal cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """

    padded_lvl = pad(warp)
    _,c ,h, w = c1.shape
    max_offset = search_range * 2 + 1
    cost_vol = []
    for x in range(0, max_offset):
        slice = padded_lvl[:, :, :, x:x+w]
        cost = torch.mean(c1 * slice, axis=1, keepdims=True)
        cost_vol.append(cost)
    cost_vol = torch.cat(cost_vol, axis=1)
    cost_vol = lrelu(cost_vol)
    return cost_vol
		
class RegressionNet_Homo(torch.nn.Module):
    def __init__(self,is_training, search_range, warped=False):
        super().__init__()
        self.search_range=search_range
        self.warped=warped
        self.keep_prob = 0.5 if is_training==True else 1.0
        self.feature = []
        if self.search_range==32:
            self.wh=32
            self.in_channel=128
            self.stride=[1,1,2]
            self.out=[128,128,128]
            self.fully=1024
            self.shift=8
        elif self.search_range==16:
            self.wh=64
            self.in_channel=128
            self.stride=[1,1,2]
            self.out=[128,128,128]
            self.fully=512
            self.shift=8
        elif self.search_range==8:
            self.wh=128
            self.in_channel=128
            self.stride=[1,2,2]
            self.out=[128,128,128]
            self.fully=256
            self.shift=5*2
        
        # self.pad=nn.ConstantPad2d(value = 0,padding = [search_range, search_range, search_range, search_range])
        self.vertical_pad=nn.ConstantPad2d(value = 0,padding = [0, 0, search_range, search_range])
        self.horizontal_pad=nn.ConstantPad2d(value = 0,padding = [search_range, search_range, 0, 0])
        if search_range==8:
            self.pad=nn.ConstantPad2d(value = 0,padding = [search_range, search_range, search_range, search_range])
        self.lrelu=nn.LeakyReLU(inplace = True)
        # self.conv1 =torch.nn.Sequential( nn.Conv2d(in_channels=((2*self.search_range)+1)*2, out_channels=self.out[0], kernel_size=3, stride=self.stride[0],padding=1),
        #                                 nn.BatchNorm2d(self.out[0]),
        #                                 nn.ReLU(True))
        # if search_range!=8:
        #     self.conv1 =torch.nn.Sequential( nn.Conv2d(in_channels = ((2*self.search_range)+1)*2, out_channels=self.out[0], kernel_size=3, stride=self.stride[0],padding=1),
        #                                     nn.BatchNorm2d(self.out[0]),
        #                                     nn.ReLU(True))
        # else:
        self.conv1 =torch.nn.Sequential( nn.Conv2d(in_channels = ((2*self.search_range)+1)*2, out_channels=self.out[0], kernel_size=3, stride=self.stride[0],padding=1),
                                        nn.BatchNorm2d(self.out[0]),
                                        nn.ReLU(True))
        self.conv2 = torch.nn.Sequential( nn.Conv2d(in_channels = self.out[0], out_channels=self.out[1], kernel_size=3, stride=self.stride[1],padding=1),
                                        nn.BatchNorm2d(self.out[1]),
                                        nn.ReLU(True))
        self.conv3 = torch.nn.Sequential( nn.Conv2d(in_channels = self.out[1], out_channels=self.out[2], kernel_size=3, stride=self.stride[2],padding=1),
                                        nn.BatchNorm2d(self.out[2]),
                                        nn.ReLU(True))
        self.getoffset = torch.nn.Sequential(torch.nn.Linear(in_features = int((self.out[2]*self.wh*self.wh)/(self.stride[0]*self.stride[1]*self.stride[2])**2), out_features = self.fully),
                                            nn.ReLU(True),
                                            nn.Dropout(p = self.keep_prob),
                                            torch.nn.Linear(in_features = self.fully, out_features = self.shift))
        self.horizontal_attention = Attention1D(self.in_channel,
                                  y_attention = False,
                                  double_cross_attn = True,
                                  )
        self.vertical_attention = Attention1D(self.in_channel,
                                  y_attention=True,
                                  double_cross_attn=True,
                                  )
    def initialize_flow(self, feature, downsample_factor=1):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        n, c, h, w = feature.shape
        # downsample_factor = self.downsample_factor if downsample is None else downsample
        coords0 = coords_grid(n, h , w ).to(feature.device)
        coords1 = coords_grid(n, h , w ).to(feature.device)
        # coords0 = coords_grid(n, h, w // downsample_factor).to(img.device)
        # coords1 = coords_grid(n, h , w // downsample_factor).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    def forward(self, feature1, feature2):
        pos_channels = feature1.shape[1] // 2
        feature1 = nn.functional.normalize(feature1, dim=1, p=2)
        pos_enc = PositionEmbeddingSine(pos_channels)
        position = pos_enc(feature1)  # [B, C, H, W]
        if not self.warped:
            feature2 = nn.functional.normalize(feature2,dim=1, p=2)

        feature2_x, attn_x = self.horizontal_attention(feature1, feature2, position)
        verti_correlation = Correlation1D(feature1, feature2_x,
                                  radius=self.search_range,
                                  x_correlation=False,
                                  )

        feature2_y, attn_y = self.vertical_attention(feature1, feature2, position)
        hori_correlation = Correlation1D(feature1, feature2_y,
                                  radius=self.search_range,
                                  x_correlation=True,
                                  )

        # coords0, coords1 = self.initialize_flow(feature1,256//feature1.shape[-1])  # 1/8 resolution or 1/4
        coords0, coords1 = self.initialize_flow(feature1)  # 1/8 resolution or 1/4
        oords1 = coords1.detach()  # stop gradient
        corr_x = hori_correlation(coords1)
        corr_y = verti_correlation(coords1)
        correlation = torch.cat((corr_x, corr_y), dim=1)  # [B, 2(2R+1), H, W]
        # correlation = torch.cat((hori_correlation, verti_correlation),1)
        conv1 = self.conv1(correlation)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        conv3_flatten = conv3.contiguous().view(conv3.shape[0],-1)
        offset=self.getoffset(conv3_flatten)
        return offset


class RegressionNet_TPS(torch.nn.Module):
    def __init__(self,is_training, search_range, warped=False):
        super().__init__()
        self.search_range=search_range
        self.warped=warped
        self.keep_prob = 0.5 if is_training==True else 1.0
        self.feature = []
        if self.search_range==32:
            self.wh=32
            self.in_channel=128
            self.stride=[1,1,2]
            self.out=[128,128,128]
            self.fully=1024
            self.shift=8
        elif self.search_range==16:
            self.wh=64
            self.in_channel=128
            self.stride=[1,1,2]
            self.out=[128,128,128]
            self.fully=512
            self.shift=8
        elif self.search_range==8:
            self.wh=128
            self.in_channel=128
            # self.stride=[1,2,2]
            self.stride=[1,1,1]
            self.out=[128,128,128]
            self.fully=256
            self.shift=5*2
        
        # self.pad=nn.ConstantPad2d(value = 0,padding = [search_range, search_range, search_range, search_range])
        self.vertical_pad=nn.ConstantPad2d(value = 0,padding = [0, 0, search_range, search_range])
        self.horizontal_pad=nn.ConstantPad2d(value = 0,padding = [search_range, search_range, 0, 0])
        if search_range==8:
            self.pad=nn.ConstantPad2d(value = 0,padding = [search_range, search_range, search_range, search_range])
        self.lrelu=nn.LeakyReLU(inplace = True)

        self.conv1 =torch.nn.Sequential( nn.Conv2d(in_channels = ((2*self.search_range)+1)*2, out_channels=self.out[0], kernel_size=3, stride=self.stride[0],padding=1),
                                        nn.BatchNorm2d(self.out[0]),
                                        nn.ReLU(True))
        self.conv2 = torch.nn.Sequential( nn.Conv2d(in_channels = self.out[0], out_channels=self.out[1], kernel_size=3, stride=self.stride[1],padding=1),
                                        nn.BatchNorm2d(self.out[1]),
                                        nn.ReLU(True))
        self.conv3 = torch.nn.Sequential( nn.Conv2d(in_channels = self.out[1], out_channels=self.out[2], kernel_size=3, stride=self.stride[2],padding=1),
                                        nn.BatchNorm2d(self.out[2]),
                                        nn.ReLU(True))
        self.getoffset = torch.nn.Sequential(torch.nn.Linear(in_features = int((self.out[2]*self.wh*self.wh)/(self.stride[0]*self.stride[1]*self.stride[2])**2), out_features = self.fully),
                                            nn.ReLU(True),
                                            nn.Dropout(p = self.keep_prob),
                                            torch.nn.Linear(in_features = self.fully, out_features = self.shift))
        self.horizontal_attention = Attention1D(self.in_channel,
                                  y_attention = False,
                                  double_cross_attn = True,
                                  )
        self.vertical_attention = Attention1D(self.in_channel,
                                  y_attention=True,
                                  double_cross_attn=True,
                                  )
        self.predict_flow = predict_flow(128,32, 2)
    def initialize_flow(self, feature, downsample_factor=1):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        n, c, h, w = feature.shape
        # downsample_factor = self.downsample_factor if downsample is None else downsample
        coords0 = coords_grid(n, h , w ).to(feature.device)
        coords1 = coords_grid(n, h , w ).to(feature.device)
        # coords0 = coords_grid(n, h, w // downsample_factor).to(img.device)
        # coords1 = coords_grid(n, h , w // downsample_factor).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    def forward(self, feature1, feature2):
        pos_channels = feature1.shape[1] // 2
        feature1 = nn.functional.normalize(feature1, dim=1, p=2)
        pos_enc = PositionEmbeddingSine(pos_channels)
        position = pos_enc(feature1)  # [B, C, H, W]
        if not self.warped:
            feature2 = nn.functional.normalize(feature2,dim=1, p=2)

        feature2_x, attn_x = self.horizontal_attention(feature1, feature2, position)
        verti_correlation = Correlation1D(feature1, feature2_x,
                                  radius=self.search_range,
                                  x_correlation=False,
                                  )

        feature2_y, attn_y = self.vertical_attention(feature1, feature2, position)
        hori_correlation = Correlation1D(feature1, feature2_y,
                                  radius=self.search_range,
                                  x_correlation=True,
                                  )

        # coords0, coords1 = self.initialize_flow(feature1,256//feature1.shape[-1])  # 1/8 resolution or 1/4
        coords0, coords1 = self.initialize_flow(feature1)  # 1/8 resolution or 1/4
        # print(feature1.shape,coords1.shape)
        oords1 = coords1.detach()  # stop gradient
        corr_x = hori_correlation(coords1)
        corr_y = verti_correlation(coords1)
        correlation = torch.cat((corr_x, corr_y), dim=1)  # [B, 2(2R+1), H, W]
        # correlation = torch.cat((hori_correlation, verti_correlation),1)
        conv1 = self.conv1(correlation)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        flow = self.predict_flow(conv3)
        # print(correlation.shape,conv1.shape,conv2.shape,conv3.shape,flow.shape)
        # print(flow.shape)
        # conv3_flatten = conv3.contiguous().view(conv3.shape[0],-1)
        # offset=self.getoffset(conv3_flatten)
        return flow


class Encoder(torch.nn.Module):
    def __init__(self, batch_size, device,geno_mix, is_training=1):
        super().__init__()
        genotype=DARTS_fusion
        self.ir_conv = torch.nn.Sequential( nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(True))  
        self.vis_conv = torch.nn.Sequential( nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(True))
        self.vis_search = Cell(genotype, 32, mode='vis')
        self.ir_search = Cell(genotype, 32, mode='ir')
    def forward(self,ir_input1,ir_input2):
        # print(ir_input1.shape)
        ir_en1 = self.ir_conv(ir_input1)
        ir_en1 = self.ir_search(ir_en1)
        ir_en2 = self.ir_conv(ir_input2)
        ir_en2 = self.ir_search(ir_en2)
        return ir_en1, ir_en2
class Initialize_Flow(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, b):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//b, W//b).cuda()
        coords1 = coords_grid(N, H//b, W//b).cuda()

        return coords0, coords1
class Decoder(torch.nn.Module):
    def __init__(self, batch_size, device,geno_mix, is_training=1):
        super().__init__()
        genotype=DARTS_fusion
        self.decoder_conv1 = torch.nn.Sequential( nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(True))
        self.decoder_search = Cell(genotype, 32, mode = 'decoder')
        self.decoder_conv2 = torch.nn.Sequential( nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1),
                                        nn.Tanh())
    def forward(self,ir_en1, ir_en2, vis_en1, vis_en2):
        fus1=L1_norm(ir_en1,vis_en1)
        fus2=L1_norm(ir_en2,vis_en2)
        ######### fus1 #########################
        fus1 = self.decoder_conv1(fus1)
        fus1 = self.decoder_search(fus1)
        fus1 = self.decoder_conv2(fus1)
        ######### fus1 #########################
        fus2 = self.decoder_conv1(fus2)
        fus2 = self.decoder_search(fus2)
        fus2 = self.decoder_conv2(fus2)
        return fus1, fus2
        
class H_estimator(torch.nn.Module):
    def __init__(self, batch_size, device,geno_mix, is_training=1):
        super().__init__()
        genotype=DARTS_fusion
        
        self.device = device
        self.fus1_search=OPS[geno_mix[0][0]](128,128,1)
        self.fus2_search=OPS[geno_mix[1][0]](128,128,1)
        self.fus3_search=OPS[geno_mix[1][0]](128,128,1) 
        # self.Rnet1 = RegressionNet(is_training, 16, warped = False)#.cuda()
        self.Rnet1 = RegressionNet_Homo(is_training, 32, warped = False)#.cuda()
        self.Rnet2 = RegressionNet_Homo(is_training, 16, warped = True)#.cuda()
        # self.Rnet3 = RegressionNet(is_training, 8, warped = True)#.cuda()
        self.Rnet3 = RegressionNet_TPS(is_training, 8, warped = True)#.cuda()
        for m in self.Rnet1.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        for m in self.Rnet2.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        for m in self.Rnet3.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        self.M_tile_inv_256, self.M_tile_256 = self.to_transform_H(256, batch_size)
        self.M_tile_inv_64, self.M_tile_64 = self.to_transform_H(64, batch_size)
        self.M_tile_inv_128, self.M_tile_128 = self.to_transform_H(128, batch_size)
        self.src = torch.tensor([[[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]]]).repeat(batch_size, 1, 1) .to(self.device) # Bx5x2
        
        # self.transform64  = Transform(64, 64, self.device,batch_size).to(self.device)
        # self.transform128  = Transform(128, 128, self.device,batch_size).to(self.device)
        # self.transform256 = Transform(256, 256, self.device,batch_size).to(self.device)
        
        self.initialize_flow_4 = Initialize_Flow()
        self.update_block_4 = GMA_update(32)
        self.initialize_flow_2 = Initialize_Flow()
        self.update_block_2 = GMA_update(64)
        
        
        self.max = nn.MaxPool2d(2)        
        
    def to_transform_H(self, patch_size, batch_size):            
        M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                    [0., patch_size / 2.0, patch_size / 2.0],
                    [0., 0., 1.]]).astype(np.float32)
        M_tensor = torch.from_numpy(M)
        M_tile = torch.unsqueeze(M_tensor, 0).repeat( [batch_size, 1, 1])
        M_inv = np.linalg.inv(M)
        M_tensor_inv = torch.from_numpy(M_inv)
        M_tile_inv = torch.unsqueeze(M_tensor_inv, 0).repeat([batch_size, 1, 1])
        M_tile_inv=M_tile_inv.to(self.device)
        M_tile=M_tile.to(self.device)
        return M_tile_inv, M_tile
    def tps_src(self,batch, w,h):
        coords = torch.meshgrid(torch.arange(-w, w+1)/w, torch.arange(-h, h+1)/h)
        coords = torch.stack(coords[::-1], dim=0).float()
        # print(coords[None].shape)
        coords = coords[None].repeat(batch, 1, 1, 1)
        coords = coords.permute(0,2,3,1)
        coords = coords.reshape(batch,-1,2)
        return coords
    # def forward(self, ir_en1, ir_en2, vis_en1, vis_en2, ir_input1=None,ir_input2=None,\
    #             vis_input1=None,vis_input2=None,gt=None,patch_size=128.,test=False,size=False):
    # def forward(self, ir_en1, ir_en2, ir_input1=None,ir_input2=None,\
    #             vis_input1=None,vis_input2=None,homo=None,tps=None,patch_size=256.,test=False,size=False):
    def forward(self, ir_en1, ir_en2,  ir_input1=None,\
                vis_input1=None,vis_input2=None,homo=None,tps=None,patch_size=256.,test=False,size=False):
        batch_size = ir_en1.shape[0]
        ############### build_model ###################################
        ir_en1 = self.max(ir_en1)
        feature1_1 = self.fus1_search(ir_en1)
        feature1_1_5 = self.max(feature1_1)

        feature1_2 = self.fus2_search(feature1_1_5)
        feature1_2_5 = self.max(feature1_2)

        feature1_3 = self.fus3_search(feature1_2_5)

        ir_en2 = self.max(ir_en2)
        feature2_1 = self.fus1_search(ir_en2)
        feature2_1_5 = self.max(feature2_1)

        feature2_2 = self.fus2_search(feature2_1_5)
        feature2_2_5 = self.max(feature2_2)

        feature2_3 = self.fus3_search(feature2_2_5)
        
        # print(feature2_3.shape,feature2_2.shape,feature2_1.shape)
        four_point_disp = torch.zeros((ir_input1.shape[0], 2, 2, 2)).cuda()
        four_point_predictions = []
        ##############################  Regression Net 1 ##############################
        net1_f = self.Rnet1(feature1_3, feature2_3)#*8
        net1_f = torch.unsqueeze(net1_f, 2)#*128
        # print(feature1_3.shape)
        # coords0, coords1 = self.initialize_flow_4(ir_input1, 4)
        coords0, coords1 = self.initialize_flow_4(ir_input1, 8)
        coords0 = coords0.detach()
        sz = feature2_3.shape
        self.sz = sz
        self.get_flow_now_4 = Get_Flow(sz)
        net1_f = net1_f.reshape(-1, 2, 2, 2)
        four_point_disp =  four_point_disp + net1_f
        four_point_predictions.append(four_point_disp)
        coords1 = self.get_flow_now_4(four_point_disp, 8)
        flow_med = coords1 - coords0
        # print(flow_med.shape)
        flow_med_H1 = F.upsample_bilinear(flow_med, None, [8, 8]) *8
        flow_med = F.upsample_bilinear(flow_med, None, [2, 2]) * 2 # 32 -> 64         
        flow_med = flow_med.detach()   
        # print(flow_med.shape,feature2_2.shape)
        ir_feature2_warp = warp(nn.functional.normalize(feature2_2, dim=1, p=2), flow_med)
        # ir_feature2_warp = self.transform64(nn.functional.normalize(feature2_2, dim=1, p=2), ir_H1)
        # ##############################  Regression Net 2 (1/8 scale)##############################
        net2_f = self.Rnet2(feature1_2, ir_feature2_warp)#*4
        net2_f = torch.unsqueeze(net2_f, 2)#*128
        net2_f = net2_f.reshape(-1,2,2,2)
        
        coords0, coords1 = self.initialize_flow_4(ir_input1, 4)
        coords0 = coords0.detach()
        sz = feature2_2.shape
        self.sz = sz
        self.get_flow_now_4 = Get_Flow(sz)

        four_point_disp =  four_point_disp + net2_f
        four_point_predictions.append(four_point_disp)
        coords1 = self.get_flow_now_4(four_point_disp, 4)
        
        flow_med = coords1 - coords0
        # flow_med = F.upsample_bilinear(flow_med, None, [4, 4]) * 4 # 32 -> 128   
        flow_med_H2 = F.upsample_bilinear(flow_med, None, [4, 4]) *4
        flow_med = F.upsample_bilinear(flow_med, None, [2, 2]) * 2 # 32 -> 128         
        flow_med = flow_med.detach()         
        ir_feature3_warp = warp(nn.functional.normalize(feature2_1, dim=1, p=2), flow_med)
        # ##############################  Regression Net 3 ##############################
        # print(ir_feature3_warp.shape,feature1_1.shape)
        net3_f = self.Rnet3(feature1_1, ir_feature3_warp) # *2
        # print(net3_f.shape)
        # print(feature1_1.shape,net3_f.shape)
        # print(ir_input1.shape)
        flow_tps = F.upsample_bilinear(net3_f, None, [2, 2])*2
        # print(four_point_disp.shape)
        # print(flow_tps.shape)
		scale = ir_input1.shape[-1]
        flow_homo_tps =  flow_med_H2 + flow_tps*scale/2
        #######################################################################################################################
        coords0, coords1 = self.initialize_flow_4(ir_input1, 1)
        sz = ir_input1.shape
        self.get_flow_now_1 = Get_Flow(sz)
        # print(homo.shape)
        coords1 = self.get_flow_now_1(homo, 1)
        flow_homo_gt = coords1 - coords0
        scale = ir_input1.shape[-1] / 2.
        # flow_field_gt = tps * scale + flow_homo_gt
        # print(torch.max(tps),torch.max(flow_homo_gt))
        scale = ir_input1.shape[-1]
        flow_field_gt = tps*scale/2. + flow_homo_gt
        
        # flow_homo_tps =  tps*scale/2. + flow_med_H2
        vis_warp2_tpsgt = warp(vis_input2, flow_field_gt)
        mask_tpsgt = warp(torch.ones_like(ir_input1), flow_field_gt)
        vis_warp2_homogt = warp(vis_input2, flow_homo_gt)
        mask_homogt = warp(torch.ones_like(ir_input1), flow_homo_gt)
        
        one = torch.ones_like(ir_input1)
        mask_H1 = warp(one, flow_med_H1)
        mask_H2 = warp(one, flow_med_H2)
        # mask_H1 = self.transform256(one, H1_mat)
        # mask_H2 = self.transform256(one, H2_mat)
        mask_tps = warp(one, flow_homo_tps)

        ir_warp1_H1 = mask_H1 * ir_input1
        ir_warp1_H2 = mask_H2 * ir_input1
        ir_warp1_tps = mask_tps * ir_input1
        

        vis_warp1_H1 = mask_H1 * vis_input1
        vis_warp1_H2 = mask_H2 * vis_input1
        vis_warp1_tps = mask_tps * vis_input1

        # vis_warp2_H1 = self.transform256(vis_input2, H1_mat)
        # vis_warp2_H2 = self.transform256(vis_input2, H2_mat)
        vis_warp2_H1 = warp(vis_input2, flow_med_H1)
        vis_warp2_H2 = warp(vis_input2, flow_med_H2)
        vis_warp2_tps = warp(vis_input2, flow_homo_tps)

        
        # mask_Hgt = self.transform256(one, Hgt_mat)
        mask_Hgt = warp(one, flow_homo_gt)

        ir_warp1_homogt = mask_Hgt * ir_input1
        ir_warp1_tpsgt = mask_tpsgt * ir_input1
        vis_warp1_homogt = mask_Hgt * vis_input1
        vis_warp1_tpsgt = mask_tpsgt * vis_input1



        ir_warp1=torch.cat((ir_warp1_H1, ir_warp1_H2, ir_warp1_tps, ir_warp1_homogt,ir_warp1_tpsgt),1)
        vis_warp2=torch.cat((vis_warp2_H1, vis_warp2_H2, vis_warp2_tps, vis_warp2_homogt,vis_warp2_tpsgt),1)
        vis_warp1=torch.cat((vis_warp1_H1, vis_warp1_H2, vis_warp1_tps, vis_warp1_homogt,vis_warp1_tpsgt),1)
        return net1_f, net2_f, flow_tps, ir_warp1, vis_warp1, vis_warp2, tps*scale/2.
    
    def warp_image_tps(
        self,
        image: torch.Tensor,
        kernel_gt_centers: torch.Tensor,
        kernel_gt_weights: torch.Tensor,
        affine_gt_weights: torch.Tensor,
        align_corners: bool = False,
        return_grid:bool = False,
    ) -> torch.Tensor:

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input image is not torch.Tensor. Got {type(image)}")

        if not isinstance(kernel_gt_centers, torch.Tensor):
            raise TypeError(f"Input kernel_gt_centers is not torch.Tensor. Got {type(kernel_gt_centers)}")

        if not isinstance(kernel_gt_weights, torch.Tensor):
            raise TypeError(f"Input kernel_gt_weights is not torch.Tensor. Got {type(kernel_gt_weights)}")

        if not isinstance(affine_gt_weights, torch.Tensor):
            raise TypeError(f"Input affine_gt_weights is not torch.Tensor. Got {type(affine_gt_weights)}")

        if not len(image.shape) == 4:
            raise ValueError(f"Invalid shape for image, expected BxCxHxW. Got {image.shape}")

        if not len(kernel_gt_centers.shape) == 3:
            raise ValueError(f"Invalid shape for kernel_gt_centers, expected BxNx2. Got {kernel_gt_centers.shape}")

        if not len(kernel_gt_weights.shape) == 3:
            raise ValueError(f"Invalid shape for kernel_gt_weights, expected BxNx2. Got {kernel_gt_weights.shape}")

        if not len(affine_gt_weights.shape) == 3:
            raise ValueError(f"Invalid shape for affine_gt_weights, expected BxNx2. Got {affine_gt_weights.shape}")

        device, dtype = image.device, image.dtype
        batch_size, _, h, w = image.shape
        coords: torch.Tensor = create_meshgrid(h, w, device=device).to(dtype=dtype)
        coords = coords.reshape(-1, 2).expand(batch_size, -1, -1)
        warped: torch.Tensor = warp_points_tps(coords, kernel_gt_centers, kernel_gt_weights, affine_gt_weights)
        warped = warped.view(-1, h, w, 2)
        warped_image: torch.Tensor = nn.functional.grid_sample(image, warped, align_corners=align_corners)
        if not return_grid:
            return warped_image
        else:
            return warped_image, warped


class H_joint_out(torch.nn.Module):
    def __init__(self, batch_size, device, is_training=1):
        super().__init__()
        self.device = device
        self.keep_prob = 1.0
        self.getoffset = torch.nn.Sequential(torch.nn.Linear(in_features = 16, out_features = 64),
                                            nn.ReLU(True),
                                            nn.Dropout(p = self.keep_prob),
                                            torch.nn.Linear(in_features = 64, out_features = 8))
        self.transform_output=Transform_output()

        
    def forward(self, offset1, offset2,size, irs, viss):
        fusion=torch.cat([offset1,offset2],1)
        fusion = fusion.contiguous().view(fusion.shape[0],-1)
        ############### build_model ###################################
        offset_out=self.getoffset(fusion)
        offset_out = torch.unsqueeze(offset_out, 2)#*128

        size_tmp = torch.cat([size,size,size,size],axis=1)/128.
        resized_shift = torch.mul(offset_out, size_tmp)
        H_mat = output_solve_DLT(resized_shift, size)  
        # H = solve_DLT(shift, 128) 

        warps_ir = self.transform_output(irs.permute(0,3,1,2), H_mat,size,resized_shift)
        warps_vis = self.transform_output(viss.permute(0,3,1,2), H_mat,size,resized_shift)
        return warps_ir, warps_vis



