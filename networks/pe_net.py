import os
from base.base_config import BaseConfig
from base.base_network import BaseNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F

class PELoss(nn.Module):
    def __init__(self, mean=True):
        super(PELoss, self).__init__()
        self.mean = mean
        self.cos = nn.CosineSimilarity(dim=1) # 方法是计算两个输入张量之间的余弦相似度 值越接近1表示相似度越高，值越接近-1表示相似度越低。
    def forward(self, p, z):
        z = z.detach()
        if self.mean:
            return 1-self.cos(p, z).mean()
        else:
            return 1-self.cos(p, z)

class PENet(BaseNetwork):
    """pretrain encoder net using siamese"""
    def __init__(self, 
                 cfg: BaseConfig,
                 ):
        super().__init__()
        self.cfg = cfg
      
    def forward(self, y_img, x_img):
        # Batch 代表 dataloder自己取的时候，默认都为1
        # SelfBatch 代表 自定义的，已经取好了
        # query_img.shape = (Batch,SelfBatch,3,224,224)
        # support_img_list.shape: (Batch,SelfBatch,shot,3,224,224)
        y_img = y_img.squeeze(0) 
        B,C,H,W = y_img.shape
        x_img = x_img.squeeze(0)
        B,K,C,H,W = x_img.shape # 批次不重要，多点好
        x_img = x_img.view(B*K, C, H, W)
        
        return
