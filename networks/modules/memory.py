import os
import time
from typing import Any, Type, Union, List, Optional, Callable

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def hard_shrink_relu(x, lambd=0.0, epsilon=1e-12):
    ''' Hard Shrinking '''
    return (F.relu(x-lambd) * x) / (torch.abs(x-lambd) + epsilon)

class Memory(nn.Module):
    ''' Memory Module '''
    def __init__(self, ch, feat, which_conv=nn.Conv2d,
                 mem_dim=50, fea_dim=256, hidden=500, shrink_thres=0.0): # shrink_thres=0.25
        super().__init__()
        self.ch = ch  # input channel
        self.feat = feat # feature dim channel
        self.norm_layer = nn.BatchNorm2d(ch)
        self.which_conv = which_conv # 这是一个卷积层
        self.i = self.which_conv(
            self.ch, self.feat, kernel_size=1, padding=0, bias=False) # 生成查询（query）特征。
        self.o = self.which_conv(
            self.feat, self.ch, kernel_size=1, padding=0, bias=False) # 将加权后的注意力特征映射回原始特征维度，生成最终的输出。
        # attention
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))   # [M, C]
        # 定义两个全连接层
        self.fc1 = nn.Linear(self.mem_dim, hidden)  # 第一个全连接层的大小可以根据需要调整
        self.fc2 = nn.Linear(hidden, self.mem_dim)  # 第二个全连接层输出大小与输入相同
        self.bias = None
        self.shrink_thres = shrink_thres
        self.reset_parameters()

    def reset_parameters(self):
        ''' init memory elements : Very Important !! '''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def gumbel_softmax_sampling(self, logits, temperature=1.0, hard=False, dim=-1):
        """
        使用Gumbel-Softmax分布进行采样。

        参数:
        logits : torch.Tensor
            模型输出的原始分数，维度为 [batch_size, num_classes]。
        temperature : float
            温度参数，控制分布的平滑程度，温度越低分布越尖锐。

        返回:
        samples : torch.Tensor
            Gumbel-Softmax采样的结果，与logits具有相同的形状。
        """
        # Step 1: 为每个logit生成Gumbel噪声 从标准指数分布中抽取的随机数
        gumbel_noise = -torch.empty_like(logits).exponential_().log() # 从标准Gumbel分布中采样
        # Step 2: 将Gumbel噪声添加到logits上
        logits_with_noise = logits + gumbel_noise
        # Step 3: 应用softmax函数
        # 使用温度参数调整Softmax的平滑度
        soft_probs = F.softmax(logits_with_noise / temperature, dim=dim) 
        # Step 4: 可选：硬Gumbel-Softmax
        if hard:
            # 使用one-hot编码
            _, idx = soft_probs.max(dim=dim, keepdim=True)
            gumbel_softmax_sample = torch.zeros_like(logits).scatter_(dim, idx, 1.0) # 对索引处为1
            return gumbel_softmax_sample
        else:
            # 返回软概率
            return soft_probs
    def forward(self, x):
        ''' x [B,C,H,W] : latent code Z'''
        x = self.i(x) # shape [B,C,H,W]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).flatten(end_dim=2) # Fea : [NxC]  N=BxHxW
        # calculate attention weight
        mem_T = self.weight.permute(1, 0) # [C,M]
        att_weight = F.linear(x, self.weight)   # Fea*Mem^T : [NxC] x [CxM] = [N, M]
        
        if self.shrink_thres > 0:
            # softmax-hard shrink
            # att_weight = F.softmax(att_weight, dim=1)   # [N, M]
            att_weight = self.gumbel_softmax_sampling(att_weight, dim=1) # gumbel
            sorted_bins = att_weight.size(1)
            sorted_affinity, _ = torch.sort(att_weight, dim=1, descending=False)  # 对每个特征的相似度进行降序排列
            histograms = torch.stack([torch.histc(sorted_affinity[i], bins=sorted_bins, 
                                                min=sorted_affinity[i].min().item(), max=sorted_affinity[i].max().item())
                                        for i in range(sorted_affinity.size(0))])  # 计算每个特征的相似度直方图
            cumulative_histograms = histograms.cumsum(dim=1)  # 沿着每行累积直方图
            relative_frequency = cumulative_histograms / cumulative_histograms[:, -1].unsqueeze(1)  # 计算相对频率
            threshold_indices = (relative_frequency > self.shrink_thres).long().argmax(dim=1)  # 转为int64，并找到每行的阈值索引
            threshold_index = threshold_indices * (sorted_affinity.size(1) // sorted_bins)
            threshold = sorted_affinity.gather(1, threshold_index.unsqueeze(1))  # 根据阈值索引获取阈值
            shrinked_att_weight = hard_shrink_relu(att_weight, threshold)  # 进行硬收缩操作
            att_weight = F.normalize(shrinked_att_weight, p=2, dim=1)    # [N, K]
        else:
            # gumbel-softmax
            att_weight = F.softmax(att_weight, dim=1)   # [N, M]
            average_att_weight = torch.mean(att_weight, dim=0)  # [M] 计算每个M下的平均
            activate_att_weight = self.fc2(F.relu(self.fc1(average_att_weight))) # fc1 => relu => fc2
            gumbel_att_weight = self.gumbel_softmax_sampling(activate_att_weight) # gumbel [M]
            final_att_weight = gumbel_att_weight.unsqueeze(0) * att_weight # shape [1,M] [N,M]
            att_weight = final_att_weight/final_att_weight.sum(dim=-1, keepdim=True) # [N,M]/[N,1]
        
        # generate code z'
        output = F.linear(att_weight, mem_T) # [N, M] x [M, C] = [N, C]
        output = output.view(B,H,W,C).permute(0,3,1,2)  # [N,C,H,W]

        output = self.o(output)

        # return att_weight, output
        return self.norm_layer(output)