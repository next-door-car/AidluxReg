import torch
import torch.nn as nn
import torch.nn.functional as F
import antialiased_cnns
import numpy as np
from networks.models.resnet import *
from networks.modules.stn import STN
from base.base_config import BaseConfig

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
class ResNet_With_STN(nn.Module):
    '''
    仅修改S
    '''
    def __init__(self,
                 cfg: BaseConfig,
                 **kwargs) -> None:
        super().__init__()
        self.cfg = cfg
        self.ann_resnet = antialiased_cnns.wide_resnet50_2(pretrained=True)
        self.stn_1 = STN(cfg, feat_in_num=64, feat_in_size=128)
        self.stn_2 = STN(cfg, feat_in_num=128, feat_in_size=64)
        self.stn_3 = STN(cfg, feat_in_num=256, feat_in_size=32)
        
    def forward(self, x):
        # 输入是[Batch, 3, 256, 256]
        # x = imagenet_norm_batch(x) # Comments on Algorithm 3: We use the image normalization of the pretrained models of torchvision [44].
        ret = []
        x0 = self.ann_resnet.conv1(x)
        x0 = self.ann_resnet.bn1(x0)
        x0 = self.ann_resnet.relu(x0)
        x0 = self.ann_resnet.maxpool(x0) # shape: [Batch, 64, 128, 128]
        
        x1 = self.ann_resnet.layer1(x0) # shape: [Batch, 64, 128, 128]
        x1, theta1 = self.stn_1(x1)
        tmp1 = np.tile(np.array([0, 0, 1]), (x1.shape[0], 1, 1)).astype(np.float32) # B 个 (0,0,1)
        fixthea1 = torch.from_numpy(np.linalg.inv(np.concatenate((theta1.detach().cpu().numpy(), tmp1), axis=1))[:,:-1,:]).cpu() # cuda(1)
        feat1 = self._fixstn(x1.detach().cpu(), fixthea1) # cpu上计算
        ret.append(feat1) # shape: [Batch, 64, 128, 128]
        
        x2 = self.ann_resnet.layer2(x1) # shape: [Batch, 128, 64, 64]
        tmp2 = np.tile(np.array([0, 0, 1]), (x2.shape[0], 1, 1)).astype(np.float32)
        fixthea2 = torch.from_numpy(np.linalg.inv(np.concatenate((theta1.detach().cpu().numpy(), tmp2), axis=1))[:,:-1,:]).cpu() # cuda(1)
        feat2 = self._fixstn(self._fixstn(x2.detach().cpu(), fixthea1), fixthea2) # 恢复
        ret.append(feat2) # shape: [Batch, 128, 64, 64]
        
        x3 = self.ann_resnet.layer3(x2) # shape: [Batch, 256, 32, 32]
        x3, theta3 = self.stn_2(x2)
        tmp3 = np.tile(np.array([0, 0, 1]), (x3.shape[0], 1, 1)).astype(np.float32)
        fixthea3 = torch.from_numpy(np.linalg.inv(np.concatenate((theta3.detach().cpu().numpy(), tmp3), axis=1))[:,:-1,:]).cpu()
        feat3 = self._fixstn(self._fixstn(self._fixstn(x3.detach().cpu(), fixthea2), fixthea1), fixthea3)
        ret.append(feat3) # shape: [Batch, 256, 32, 32]
        
        out = self.ann_resnet.layer4(x3) # shape: [Batch, 512, 16, 16] 舍弃掉
        
        return ret, out
    def _fixstn(self, x, theta):
        # 含义是stn网络=>变换后的图像
        grid = F.affine_grid(theta, torch.Size(x.shape))
        img_transform = F.grid_sample(x, grid, padding_mode="reflection")

        return img_transform