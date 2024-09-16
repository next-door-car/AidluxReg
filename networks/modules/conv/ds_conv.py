import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义深度可分离卷积模块
class DSConv(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, 
                 kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        # 深度可分离卷积包括深度卷积和逐点卷积两个步骤
        self.depthConv = nn.Conv2d(in_channels, in_channels, 
                                   kernel_size=kernel_size, stride=stride, padding=padding,  
                                   groups=in_channels)
        self.pointConv = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.depthConv(x)
        x = self.pointConv(x)
        return x