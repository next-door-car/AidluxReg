import torch
from torch.functional import norm
import torch.nn.functional as F
import torch.nn as nn
from math import exp
from functools import partial


def CosLoss(data1, data2, Mean=True):
    # data1: P
    # data2: Z 抑制编码
    data2 = data2.detach() # 停止梯度传播
    cos = nn.CosineSimilarity(dim=1) # 方法是计算两个输入张量之间的余弦相似度 值越接近1表示相似度越高，值越接近-1表示相似度越低。
    if Mean:
        return 1-cos(data1, data2).mean()
    else:
        return 1-cos(data1, data2)

def averCosineSimilatiry(A, B, Mean=True):

    # param A: 表示特征图1:[N, C, H, W]
    # param B: 表示特征图2:[N, C, H, W]
    # return: 返回均值相似度:[N]
    
    N = A.shape[0] 

    B = B.detach() # 停止梯度传播(原始没加入，不知道有啥问题没)
    
    cos = nn.CosineSimilarity(dim=1) # 不放入cuda()方法是计算两个输入张量之间的余弦相似度 值越接近1表示相似度越高，值越接近-1表示相似度越低。

    A = F.adaptive_avg_pool2d(A, [1, 1])  # [N, C, 1, 1]
    B = F.adaptive_avg_pool2d(B, [1, 1])  # [N, C, 1, 1]

    A = A.view(A.shape[0], A.shape[1])  # [N, C]
    B = B.view(B.shape[0], B.shape[1])  # [N, C]

    A = F.normalize(A, dim=1)  # l2
    B = F.normalize(B, dim=1)

    if Mean:
        return 1-cos(A, B).mean()
    else:
        return 1-cos(A, B)

'''
def averCosineSimilatiry(A, B):

    # param A: 表示特征图1:[N, C, H, W]
    # param B: 表示特征图2:[N, C, H, W]
    # return: 返回均值相似度:[N]
    
    N = A.shape[0] 

    criterion_similarity = nn.CosineSimilarity(dim=1).cuda()

    A = F.adaptive_avg_pool2d(A, [1, 1])  # [N, C, 1, 1]
    B = F.adaptive_avg_pool2d(B, [1, 1])  # [N, C, 1, 1]

    A = A.view(A.shape[0], A.shape[1])  # [N, C]
    B = B.view(B.shape[0], B.shape[1])  # [N, C]

    A = F.normalize(A, dim=1)  # l2
    B = F.normalize(B, dim=1)
    
    similarity = criterion_similarity(A, B)

    return similarity
'''