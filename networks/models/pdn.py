import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.models.resnet import *
from networks.modules.pasa import Downsample_PASA_group_softmax

class PDN_S(nn.Module):

    def __init__(self, last_kernel_size=384,with_bn=False) -> None:
        super().__init__()
        # Layer Name Stride Kernel Size Number of Kernels Padding Activation
        # Conv-1 1×1 4×4 128 3 ReLU
        # AvgPool-1 2×2 2×2 128 1 -
        # Conv-2 1×1 4×4 256 3 ReLU
        # AvgPool-2 2×2 2×2 256 1 -
        # Conv-3 1×1 3×3 256 1 ReLU
        # Conv-4 1×1 4×4 384 0 -
        # 可以修改为深度可分离的卷积
        # 尺度到32才能感受到全局的变化
        # 卷积需要先考虑感受野（核）在考虑是否padding
        # 一级卷积可以适当扩大感受野
        # 二级卷积同上
        # 三级卷积需要缩小感受野，因为需要提取局部特征
        self.with_bn = with_bn
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, last_kernel_size, kernel_size=4, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(128)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(last_kernel_size)

    def forward(self, x):
        # 输入是[Batch, 3, 256, 256]
        x = self.conv1(x) # shape: [Batch, 128, 256, 256] 256 = (256-4+2*3)/1+1
        x = self.bn1(x) if self.with_bn else x
        x = F.relu(x)
        x = self.avgpool1(x) # shape: [Batch, 128, 128, 128] 128 = (256-2+2*1)/2+1
        
        x = self.conv2(x) # shape: [Batch, 256, 128, 128] 128 = (128-4+2*3)/1+1
        x = self.bn2(x) if self.with_bn else x
        x = F.relu(x)
        x = self.avgpool2(x) # shape: [Batch, 256, 64, 64] 64 = (128-2+2*1)/2+1
        
        x = self.conv3(x) # shape: [Batch, 256, 64, 64] 64 = (64-3+2*1)/1+1
        x = self.bn3(x) if self.with_bn else x
        x = F.relu(x)
        
        x = self.conv4(x) # shape: [Batch, 384, 61, 61] 61 = (64-4+2*0)/1+1
        x = self.bn4(x) if self.with_bn else x
        return x

class PDN_M(nn.Module):

    def __init__(self, last_kernel_size=384,with_bn=False) -> None:
        super().__init__()
        # Layer Name Stride Kernel Size Number of Kernels Padding Activation
        # Conv-1 1×1 4×4 256 3 ReLU
        # AvgPool-1 2×2 2×2 256 1 -
        # Conv-2 1×1 4×4 512 3 ReLU
        # AvgPool-2 2×2 2×2 512 1 -
        # Conv-3 1×1 1×1 512 0 ReLU
        # Conv-4 1×1 3×3 512 1 ReLU
        # Conv-5 1×1 4×4 384 0 ReLU
        # Conv-6 1×1 1×1 384 0 -
        # 输入是[Batch, 3, 256, 256]
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, last_kernel_size, kernel_size=4, stride=1, padding=0)
        self.conv6 = nn.Conv2d(last_kernel_size, last_kernel_size, kernel_size=1, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(256)
            self.bn2 = nn.BatchNorm2d(512)
            self.bn3 = nn.BatchNorm2d(512)
            self.bn4 = nn.BatchNorm2d(512)
            self.bn5 = nn.BatchNorm2d(last_kernel_size)
            self.bn6 = nn.BatchNorm2d(last_kernel_size)

    def forward(self, x):
        # 输入是[Batch, 3, 256, 256]
        x = self.conv1(x) # shape: [Batch, 256, 256, 256] 256 = (256-4+2*3)/1+1
        x = self.bn1(x) if self.with_bn else x 
        x = F.relu(x) 
        x = self.avgpool1(x) # shape: [Batch, 256, 128, 128] 128 = (256-2+2*1)/2+1
        
        x = self.conv2(x) # shape: [Batch, 512, 128, 128] 128 = (128-4+2*3)/1+1
        x = self.bn2(x) if self.with_bn else x 
        x = F.relu(x) 
        x = self.avgpool2(x) # shape: [Batch, 512, 64, 64] 64 = (128-2+2*1)/2+1
        
        x = self.conv3(x) # shape: [Batch, 512, 64, 64] 64 = (64-1+2*0)/1+1
        x = self.bn3(x) if self.with_bn else x 
        x = F.relu(x)  
        
        x = self.conv4(x) # shape: [Batch, 512, 64, 64] 64 = (64-3+2*1)/1+1
        x = self.bn4(x) if self.with_bn else x 
        x = F.relu(x)
        
        x = self.conv5(x) # shape: [Batch, 384, 61, 61] 61 = (64-4+2*0)/1+1
        x = self.bn5(x) if self.with_bn else x
        x = F.relu(x)
        
        x = self.conv6(x) # shape: [Batch, 384, 61, 61] 61 = (61-1+2*0)/1+1
        x = self.bn6(x) if self.with_bn else x 
        return x