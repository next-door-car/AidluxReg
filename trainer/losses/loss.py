from typing import Type, Union, List, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.losses.msssim import MSSSIM

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def L1Loss(x, y):
    # data shape: BCHW
    pass

def L2Loss(x, y):
    # data shape: BCHW
    norm_data = torch.norm(x-y, p=2, dim=1)
    # norm_data shape: BHW
    loss = norm_data.mean()
    # print("loss: ", loss)
    return loss

class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
    def forward(self, x, y):
        cos = nn.functional.cosine_similarity(x, y, dim=1)
        ano_map = torch.ones_like(cos) - cos
        loss = (ano_map.view(ano_map.shape[0],-1).mean(-1)).mean()
        return loss

class CosineReconstructLoss(nn.Module):
    def __init__(self):
        super(CosineReconstructLoss, self).__init__()
    def forward(self, x, y):
        return torch.mean(1 - torch.nn.CosineSimilarity()(x, y))
    
class CosineContrastLoss(nn.Module):
    def __init__(self,margin=0.5):
        super(CosineContrastLoss, self).__init__()
        self.cos_contrast = nn.CosineEmbeddingLoss(margin=margin)
    def forward(self, x, y):
        target = -torch.ones(x.shape[0]).to(x.device) # 期望最大距离
        return self.cos_contrast(x.view(x.shape[0],-1), 
                                 y.view(x.shape[0],-1),
                                 target = target)