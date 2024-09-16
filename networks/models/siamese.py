from turtle import forward
import torch
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.models import resnet18

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 1 if kernel_size == 3 else 3

        self.convsa = nn.Conv2d(in_channels=2, out_channels=1, 
                                kernel_size=kernel_size, padding=padding, 
                                bias=False) # 10-3+2*1 / 1 + 1 = 10
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.convsa(x)
        return self.sigmoid(x)

class Project(nn.Module):
    def __init__(self, planes=384):
        super(Project, self).__init__()
        self.conv1 = conv1x1(in_planes=planes, out_planes=planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.sa1 = SpatialAttention()

        self.conv2 = conv1x1(in_planes=planes, out_planes=planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.sa2 = SpatialAttention()

        self.conv3 = conv1x1(in_planes=planes, out_planes=planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU()
    
    def forward(self, x):
        # projector MLP
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.sa1(x) * x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.sa2(x) * x

        out = self.conv3(x) # shape: (B, 256, H, W)
        # out = self.relu3(x) # 不需要relu

        return out

class Predictor(nn.Module):
    def __init__(self, planes=384):
        super(Predictor, self).__init__()
        self.conv1 = conv1x1(in_planes=planes, out_planes=planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()

        self.conv2 = conv1x1(in_planes=planes, out_planes=planes)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        out = self.conv2(x) # shape: (B, 256, H, W)
        # out = self.relu2(x) # 不需要relu

        return out



