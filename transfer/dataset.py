import os
import sys
import cv2
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import numpy as np
import kornia
import torch
import random as rd
from typing import Tuple
from kornia.filters.kernels import get_gaussian_kernel2d
import time
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.conversions import normalize_homography
from kornia.geometry.transform import warp_points_tps
from kornia.utils.helpers import _torch_inverse_cast
from kornia.utils import create_meshgrid
from kornia.geometry.linalg import transform_points
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
NPY_EXTENSIONS = ['.npy']

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



class Image_style(data.Dataset):
    #"数据集"

    # def __init__(self, ir1_path: str, ir2_path: str, vis1_path: str, vis2_path: str, gt_path: str,mode='mix'):
    def __init__(self, ir_path: str, vis_path: str):
        super(Image_style, self).__init__()


        # self.ir1_path, self.ir2_path ,self.vis1_path, self.vis2_path, self.gt_path= ir1_path, ir2_path, vis1_path, vis2_path, gt_path
        self.ir_path, self.vis_path= ir_path, vis_path
        self.irs = sorted([x for x in os.listdir(ir_path) if is_image(x)])
        # self.ir2s = sorted([x for x in os.listdir(ir2_path) if is_image(x)])
        self.viss = sorted([x for x in os.listdir(vis_path) if is_image(x)])
        self.trans = kornia.augmentation.RandomThinPlateSpline(scale=0.15,p=1.0, return_transform=True)
        # self.vis2s = sorted([x for x in os.listdir(vis2_path) if is_image(x)])
        # self.vis1s.reverse()
        # self.vis2s.reverse()
        # self.shifts =sorted([x for x in os.listdir(gt_path) if is_npy(x)])
        try:
            if len(self.irs) != len(self.viss):
                sys.exit(0)
        except:
            print("[Src Image] and [Sal Image] don't match.")

    def __getitem__(self, index):
        name=self.irs[index]
        # gt_name=self.shifts[index]
        ir = cv2.imread(os.path.join(self.ir_path, name))
        vis = cv2.imread(os.path.join(self.vis_path, name))
        # shift=np.load(os.path.join(self.gt_path, gt_name))
        addx=rd.randint(0,20)
        addy=rd.randint(0,20)
        ir = cv2.resize(ir,(512+addx,512+addy))
        vis = cv2.resize(vis,(512+addx,512+addy))
        ir  = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)/255.#[...,None]
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)/255.#[...,None]
        height = ir.shape[0] 
        width = ir.shape[1]  
        ir = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0).float()
        vis = torch.from_numpy(vis).unsqueeze(0).unsqueeze(0).float()
        
        input1_loc = np.zeros((4,2))
        x_o = rd.randint(int(-0.25*256),int(0.25*256))
        y_o = rd.randint(int(-0.25*256),int(0.25*256))
        input1_loc[0] = [128 + x_o, 128 + y_o]
        input1_loc[1] = [384 + x_o, 128 + y_o]
        input1_loc[2] = [128 + x_o, 384 + y_o]
        input1_loc[3] = [384 + x_o, 384 + y_o]
        random_offset = np.zeros((4,2))
        for i in range(random_offset.shape[0]):
            for j in range(random_offset.shape[1]):
                    random_offset[i][j] = rd.randint(-int(0.20*256), int(0.20*256))
        input2_loc = input1_loc + random_offset
        input1_loc = torch.from_numpy(input1_loc).unsqueeze(0)
        input2_loc = torch.from_numpy(input2_loc).unsqueeze(0)
        H = kornia.geometry.homography.find_homography_dlt(input2_loc,input1_loc).to(ir.device)
        vis_warp_H = self.warp_perspective(vis, H.float(), dsize=[height,width])
        ir_warp_H = self.warp_perspective(ir, H.float(), dsize=[height,width])
        if rd.random()>0.8:
            vis_warp_H, _, params = self.trans(vis_warp_H)
            src = params['src']
            dst = params['dst']
            kernel, affine = kornia.geometry.transform.get_tps_transform(dst, src)
            size = np.array([width, height], dtype=np.float32)
            size=np.expand_dims(size, 1)
            ir_warp_H, grid_tps = self.warp_image_tps(ir_warp_H, src.to(ir_warp_H.device), kernel.to(ir_warp_H.device), affine.to(ir_warp_H.device), return_grid=True)
        ir_warp_H=ir_warp_H[0][0].numpy()[None,...]
        # print(vis_warp_H.shape)
        vis_warp_H=vis_warp_H[0][0].numpy()[None,...]
        # print(ir_warp_H.shape)
        # print("haha")
        # print(ir_warp_H[128+x_o:384+x_o,128+y_o:384+y_o].shape)
        # print(vis_warp_H[128+x_o:384+x_o,128+y_o:384+y_o].shape)
        # time.sleep(2)
        # return ir_warp_H[256+x_o:512+x_o,256+y_o:512+y_o],vis_warp_H[256+x_o:512+x_o,256+y_o:512+y_o]
        return vis_warp_H[:,128+x_o:384+x_o,128+y_o:384+y_o],ir_warp_H[:,128+x_o:384+x_o,128+y_o:384+y_o]

        # ir1gray= (cv2.cvtColor(ir1, cv2.COLOR_BGR2GRAY)/255.)[None]
        # vis1gray= (cv2.cvtColor(vis1, cv2.COLOR_BGR2GRAY)/255.)[None]
        # irlc1= (cv2.cvtColor(irlc1, cv2.COLOR_BGR2GRAY)/255.)[None]
        # vislc1= (cv2.cvtColor(vislc1, cv2.COLOR_BGR2GRAY)/255.)[None]
        # ir1gray= (cv2.cvtColor(ir1, cv2.COLOR_BGR2GRAY)/255.)[...,None]
        # vis1gray= (cv2.cvtColor(vis1, cv2.COLOR_BGR2GRAY)/255.)[...,None]
        # irlc1= (cv2.cvtColor(irlc1, cv2.COLOR_BGR2GRAY)/255.)[...,None]
        # vislc1= (cv2.cvtColor(vislc1, cv2.COLOR_BGR2GRAY)/255.)[...,None]
        
        # gt_fus= (cv2.cvtColor(gt_fus, cv2.COLOR_BGR2GRAY)/255.)[...,None]
        # gt_fus= (cv2.cvtColor(gt_fus, cv2.COLOR_BGR2GRAY)/255.)[...,None]
        # ir1= (cv2.cvtColor(ir1, cv2.COLOR_BGR2GRAY)/255.)[...,None]
        # ir2= (cv2.cvtColor(ir2, cv2.COLOR_BGR2GRAY)/255.)[...,None]
        # vis1= (cv2.cvtColor(vis1, cv2.COLOR_BGR2GRAY)/255.)[...,None]
        # vis2= (cv2.cvtColor(vis2, cv2.COLOR_BGR2GRAY)/255.)[...,None]
        # gt_fus= (cv2.cvtColor(gt_fus, cv2.COLOR_BGR2GRAY)/255.)[...,None]
        
        # return ir, vis#,shift#,ir1gray,vis1gray, irlc1, vislc1, shift#, name

    def __len__(self):
        return len(self.viss)
    def warp_perspective(
        self,
        src: torch.Tensor,
        M: torch.Tensor,
        dsize: Tuple[int, int],
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
        align_corners: bool = True,
        fill_value: torch.Tensor = torch.zeros(3),  # needed for jit
        return_grid=False
    ) -> torch.Tensor:
        if not isinstance(src, torch.Tensor):
            raise TypeError(f"Input src type is not a torch.Tensor. Got {type(src)}")

        if not isinstance(M, torch.Tensor):
            raise TypeError(f"Input M type is not a torch.Tensor. Got {type(M)}")

        if not len(src.shape) == 4:
            raise ValueError(f"Input src must be a BxCxHxW tensor. Got {src.shape}")

        if not (len(M.shape) == 3 and M.shape[-2:] == (3, 3)):
            raise ValueError(f"Input M must be a Bx3x3 tensor. Got {M.shape}")

        # fill padding is only supported for 3 channels because we can't set fill_value default
        # to None as this gives jit issues.
        if padding_mode == "fill" and fill_value.shape != torch.Size([3]):
            raise ValueError(f"Padding_tensor only supported for 3 channels. Got {fill_value.shape}")

        B, _, H, W = src.size()
        h_out, w_out = dsize

        # we normalize the 3x3 transformation matrix and convert to 3x4
        dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M, (H, W), (h_out, w_out))  # Bx3x3

        src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)  # Bx3x3

        # this piece of code substitutes F.affine_grid since it does not support 3x3
        grid = (
            create_meshgrid(h_out, w_out, normalized_coordinates=True, device=src.device).to(src.dtype).repeat(B, 1, 1, 1)
        )
        grid = transform_points(src_norm_trans_dst_norm[:, None, None], grid)

        if padding_mode == "fill":
            return self._fill_and_warp(src, grid, align_corners=align_corners, mode=mode, fill_value=fill_value)
        if not return_grid:
            return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)
        else:
            return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode),grid
        
    def _fill_and_warp(
        self,
        src: torch.Tensor,
        grid: torch.Tensor,
        mode: str,
        align_corners: bool,
        fill_value: torch.Tensor,
    ) -> torch.Tensor:
        ones_mask = torch.ones_like(src)
        fill_value = fill_value.to(ones_mask)[None, :, None, None]  # cast and add dimensions for broadcasting
        inv_ones_mask = 1 - F.grid_sample(ones_mask, grid, align_corners=align_corners, mode=mode, padding_mode="zeros")
        inv_color_mask = inv_ones_mask * fill_value
        return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode="zeros") + inv_color_mask
    def warp_image_tps(
        self,
        image: torch.Tensor,
        kernel_centers: torch.Tensor,
        kernel_weights: torch.Tensor,
        affine_weights: torch.Tensor,
        align_corners: bool = False,
        return_grid:bool = False,
    ) -> torch.Tensor:

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input image is not torch.Tensor. Got {type(image)}")

        if not isinstance(kernel_centers, torch.Tensor):
            raise TypeError(f"Input kernel_centers is not torch.Tensor. Got {type(kernel_centers)}")

        if not isinstance(kernel_weights, torch.Tensor):
            raise TypeError(f"Input kernel_weights is not torch.Tensor. Got {type(kernel_weights)}")

        if not isinstance(affine_weights, torch.Tensor):
            raise TypeError(f"Input affine_weights is not torch.Tensor. Got {type(affine_weights)}")

        if not len(image.shape) == 4:
            raise ValueError(f"Invalid shape for image, expected BxCxHxW. Got {image.shape}")

        if not len(kernel_centers.shape) == 3:
            raise ValueError(f"Invalid shape for kernel_centers, expected BxNx2. Got {kernel_centers.shape}")

        if not len(kernel_weights.shape) == 3:
            raise ValueError(f"Invalid shape for kernel_weights, expected BxNx2. Got {kernel_weights.shape}")

        if not len(affine_weights.shape) == 3:
            raise ValueError(f"Invalid shape for affine_weights, expected BxNx2. Got {affine_weights.shape}")

        device, dtype = image.device, image.dtype
        batch_size, _, h, w = image.shape
        coords: torch.Tensor = create_meshgrid(h, w, device=device).to(dtype=dtype)
        coords = coords.reshape(-1, 2).expand(batch_size, -1, -1)
        warped: torch.Tensor = warp_points_tps(coords, kernel_centers, kernel_weights, affine_weights)
        warped = warped.view(-1, h, w, 2)
        warped_image: torch.Tensor = nn.functional.grid_sample(image, warped, align_corners=align_corners)
        if not return_grid:
            return warped_image
        else:
            return warped_image, warped


class Image_style_test(data.Dataset):
    #"数据集"
    def __init__(self, vis_path: str):
        super(Image_style_test, self).__init__()
        self.vis_path=  vis_path
        self.viss = sorted([x for x in os.listdir(vis_path) if is_image(x)])

    def __getitem__(self, index):
        name=self.viss[index]
        vis = cv2.imread(os.path.join(self.vis_path, name))
        # vis = cv2.resize(vis,(256, 256))
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)/255.#[...,None]
        height = vis.shape[0] 
        width = vis.shape[1]  
        vis = torch.from_numpy(vis).unsqueeze(0).float()
        return vis,name

    def __len__(self):
        return len(self.viss)
   