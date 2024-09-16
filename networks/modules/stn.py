import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable

import numpy as np

from base.base_config import BaseConfig

# N 参数
N_PARAMS = {'affine': 6, # 仿射变换矩阵 2*3
            'translation': 2, # 平移
            'rotation': 1, # 旋转
            'scale': 2, # 缩放
            'shear': 2, # 剪切
            'rotation_scale': 3, # 旋转+缩放
            'translation_scale': 4, # 平移+缩放
            'rotation_translation': 3, # 旋转+平移 
            'rotation_translation_scale': 5} # 旋转+平移+缩放

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, 
                     kernel_size=1, stride=stride, 
                     bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride, # 1
        padding=dilation, dilation=dilation, # 1 padding 
        groups=groups, # 1
        bias=False, # False
    ) # (N-3+1*2)*1+1 = N

class STN(nn.Module):
    def __init__(self, 
                 cfg: BaseConfig,
                 feat_in_num=128, feat_in_size=256,
                 stn_mode = 'rotation_translation'):
        super().__init__()
    
        self.is_train = cfg.get_config(['model', 'is_train'])
        self.stn_mode = stn_mode
        self.stn_n_params = N_PARAMS[self.stn_mode]
        self.fn1_num = feat_in_num
        self.fn2_num = int(self.fn1_num/2)
        self.fn3_num = int(self.fn2_num/4) # 64/8 = 8
        self.fn_in_size = feat_in_size
        self.fn_out_size = feat_in_size/4/4 # 256/16 = 16  128/16 = 8
        self.feat_out = int(self.fn3_num * self.fn_out_size * self.fn_out_size) / 10
        self.fn = nn.Sequential(
            # SPPF(128, self.feat_out), # shape 不变 （自己添加的）
            conv3x3(in_planes=self.fn1_num, out_planes=self.fn2_num), # 256 
            nn.BatchNorm2d(self.fn2_num),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=1), # 256-4+2 / 4 + 1=  252/4 + 1 = 63
            conv3x3(in_planes=self.fn2_num, out_planes=self.fn3_num), # 64
            nn.BatchNorm2d(self.fn3_num),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=1), # 16
        )
        self.fc = nn.Sequential(
            # (6, 16, 8, 8)
            nn.Linear(in_features=int(self.fn3_num * self.fn_out_size * self.fn_out_size), # 16*16*16
                      out_features=int(self.feat_out)),
            nn.ReLU(True),
            nn.Linear(in_features=int(self.feat_out), out_features=self.stn_n_params),
        )
        self.fc[2].weight.data.fill_(0)
        self.fc[2].weight.data.zero_()
        if self.stn_mode == 'affine':
            self.fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        elif self.stn_mode in ['translation', 'shear']:
            self.fc[2].bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        elif self.stn_mode == 'scale':
            self.fc[2].bias.data.copy_(torch.tensor([1, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation':
            self.fc[2].bias.data.copy_(torch.tensor([0], dtype=torch.float))
        elif self.stn_mode == 'rotation_scale':
            self.fc[2].bias.data.copy_(torch.tensor([0, 1, 1], dtype=torch.float))
        elif self.stn_mode == 'translation_scale':
            self.fc[2].bias.data.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation':
            self.fc[2].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation_scale':
            self.fc[2].bias.data.copy_(torch.tensor([0, 0, 0, 1, 1], dtype=torch.float))
        
    def forward(self, x):
        mode = self.stn_mode
        batch_size = x.size(0)
        fix_x = self.fn(x) 
        theta = self.fc(fix_x.view(batch_size, -1)) # 3个参数=旋转+缩放

        if mode == 'affine':
            # affine 是 什么模式：仿射变换矩阵
            theta1 = theta.view(batch_size, 2, 3)
        else:
            if self.is_train:
                theta1 = Variable(torch.zeros([batch_size, 2, 3], 
                                            dtype=torch.float32, 
                                            device=x.get_device()), # gpu
                                  requires_grad=True) # 2*3的张量
            else:
                theta1 = Variable(torch.zeros([batch_size, 2, 3], 
                                            dtype=torch.float32, 
                                            device=x.get_device()),
                                  requires_grad=True) # 2*3的张量
            theta1 = theta1 + 0
            theta1[:, 0, 0] = 1.0
            theta1[:, 1, 1] = 1.0
            if mode == 'translation':
                theta1[:, 0, 2] = theta[:, 0]
                theta1[:, 1, 2] = theta[:, 1]
            elif mode == 'rotation':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle)
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle)
            elif mode == 'scale':
                theta1[:, 0, 0] = theta[:, 0]
                theta1[:, 1, 1] = theta[:, 1]
            elif mode == 'shear':
                theta1[:, 0, 1] = theta[:, 0]
                theta1[:, 1, 0] = theta[:, 1]
            elif mode == 'rotation_scale':
                # 旋转+缩放 = 3个参数
                # theta = [angle, scale_x, scale_y]
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle) * theta[:, 1]
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle) * theta[:, 2]
            elif mode == 'translation_scale':
                theta1[:, 0, 2] = theta[:, 0]
                theta1[:, 1, 2] = theta[:, 1]
                theta1[:, 0, 0] = theta[:, 2]
                theta1[:, 1, 1] = theta[:, 3]
            elif mode == 'rotation_translation':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle)
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle)
                theta1[:, 0, 2] = theta[:, 1]
                theta1[:, 1, 2] = theta[:, 2]
            elif mode == 'rotation_translation_scale':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle) * theta[:, 3]
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle) * theta[:, 4]
                theta1[:, 0, 2] = theta[:, 1]
                theta1[:, 1, 2] = theta[:, 2]

        # theta1 = theta1.view(batch_size, 2, 3):变换矩阵
        grid = F.affine_grid(theta1, torch.Size(x.shape)) # 旋转和缩放作用在x上
        # theta: 一个 N*2*3的张量，N是batch size。
        # size： 是得到的网格的尺度，也就是希望仿射变换之后得到的图像大小
        img_transform = F.grid_sample(x, grid, padding_mode="reflection") # 插值在x上

        return img_transform, theta1
