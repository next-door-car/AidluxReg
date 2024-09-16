import os
import time
from base.base_config import BaseConfig
from base.base_network import BaseNetwork
from typing import Any, Type, Union, List, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


import geomloss

from trainer.losses.loss import CosineLoss, CosineReconstructLoss, CosineContrastLoss

from networks.modules.memory import Memory
from networks.modules.proto import ProtoMemory

# from antialiased_cnns import wide_resnet50_2
from networks.models.resnet import wide_resnet50_2
from networks.models.de_resnet import de_wide_resnet50_2
from networks.modules.ocbe import OCBE_l1, OCBE_l2, OCBE_l3

def imagenet_norm_batch(x):
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to('cuda')
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to('cuda')
    x_norm = (x - mean) / (std + 1e-11)
    return x_norm

class ProjLayer(nn.Module):
    '''
    inputs: features of encoder block
    outputs: projected features
    '''
    def __init__(self, in_c, out_c):
        super(ProjLayer, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_c, in_c//2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//2, in_c//4, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//4),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//4, in_c//2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//2, out_c, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(out_c),
                                  torch.nn.LeakyReLU(),
                                  )
    def forward(self, x):
        return self.proj(x)
    
class MultiProjectionLayer(nn.Module):
    def __init__(self, base = 64):
        super(MultiProjectionLayer, self).__init__()
        self.proj_a = ProjLayer(base * 4, base * 4)
        self.proj_b = ProjLayer(base * 8, base * 8)
        self.proj_c = ProjLayer(base * 16, base * 16)
        
    def forward(self, features, features_noise = False):
        if features_noise is not False:
            features_1 = self.proj_a(features[0])
            features_2 = self.proj_b(features[1])
            features_3 = self.proj_c(features[2])
            features_noise_1 = self.proj_a(features_noise[0])
            features_noise_2 = self.proj_b(features_noise[1])
            features_noise_3 = self.proj_c(features_noise[2])
            return([features_1, features_2, features_3], \
                   [features_noise_1, features_noise_2, features_noise_3])
        else:
            features_1 = self.proj_a(features[0])
            features_2 = self.proj_b(features[1])
            features_3 = self.proj_c(features[2])
            return [features_1, features_2, features_3]

class Encoder(nn.Module):
    '''
    仅修改S
    '''
    def __init__(self,
                 cfg: BaseConfig,
                 base: int = 64,
                 **kwargs) -> None:
        super().__init__()
        self.cfg = cfg
        self.resnet, self.bn = wide_resnet50_2(pretrained=True)
        self.proj = MultiProjectionLayer(base=base)
        self.memory_l1 = Memory(ch=base*4, feat=base*2,
                                which_conv=nn.Conv2d,
                                mem_dim=500,fea_dim=base*2,hidden=base*4)
        self.memory_l2 = Memory(ch=base*8, feat=base*4,
                                which_conv=nn.Conv2d,
                                mem_dim=500,fea_dim=base*4,hidden=base*8)
        self.memory_l3 = Memory(ch=base*16, feat=base*8,
                                which_conv=nn.Conv2d,
                                mem_dim=500,fea_dim=base*8,hidden=base*16)
        # self.memory_l1 = ProtoMemory(ch=base*4, feat=base*2,
        #                              init_num_k=500, init_pool_size_per_cluster=10, 
        #                              warmup_total_iter=500.0,
        #                              which_conv=nn.Conv2d,
        #                              cp_momentum=1, 
        #                              cp_phi_momentum=0.6, 
        #                              device='cuda:0')
        # self.memory_l2 = ProtoMemory(ch=base*8, feat=base*4,
        #                              init_num_k=500, init_pool_size_per_cluster=10, 
        #                              warmup_total_iter=500.0,
        #                              which_conv=nn.Conv2d,
        #                              cp_momentum=1, 
        #                              cp_phi_momentum=0.6, 
        #                              device='cuda:0')
        # self.memory_l3 = ProtoMemory(ch=base*16, feat=base*8,
        #                              init_num_k=500, init_pool_size_per_cluster=10, 
        #                              warmup_total_iter=500.0,
        #                              which_conv=nn.Conv2d,
        #                              cp_momentum=1, 
        #                              cp_phi_momentum=0.6, 
        #                              device='cuda:0')  
    def forward(self, x, x_noise):
        # 输入是[Batch, 3, 256, 256]
        x0 = self.resnet.conv1(x) # b,64,128,128
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        x0 = self.resnet.maxpool(x0) # shape: [Batch, 64, 64, 64]
        en = []
        x1 = self.resnet.layer1(x0) # shape: [Batch, 64, 64, 64]
        en.append(x1) 
        x2 = self.resnet.layer2(x1) # shape: [Batch, 128, 32, 32]
        en.append(x2) 
        x3 = self.resnet.layer3(x2) # shape: [Batch, 256, 16, 16]
        en.append(x3) 
        # out = self.ann_resnet.layer4(x3) # shape: [Batch, 512*expansion, 8, 8] 舍弃掉
        
        # 输入是[Batch, 3, 256, 256]
        x0 = self.resnet.conv1(x_noise) # b,64,128,128
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        x0 = self.resnet.maxpool(x0) # shape: [Batch, 64, 64, 64]
        en_noise = []
        x1 = self.resnet.layer1(x0) # shape: [Batch, 64, 64, 64]
        en_noise.append(x1) 
        x2 = self.resnet.layer2(x1) # shape: [Batch, 128, 32, 32]
        en_noise.append(x2) 
        x3 = self.resnet.layer3(x2) # shape: [Batch, 256, 16, 16]
        en_noise.append(x3) 
        # out = self.ann_resnet.layer4(x3) # shape: [Batch, 512*expansion, 8, 8] 舍弃掉
        
        ret_proj = []
        proj, proj_noise = self.proj(en, en_noise)
        ret_proj.append(self.memory_l1(proj[0]))
        ret_proj.append(self.memory_l2(proj[1]))
        ret_proj.append(self.memory_l3(proj[2]))
        out = self.bn(ret_proj) # shape: [B, 512*expansion, 8, 8]

        return en, en_noise, proj, proj_noise, out 

class Decoder(nn.Module):
    def __init__(self,
                 cfg: BaseConfig,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.de_resnet = de_wide_resnet50_2(pretrained=False)
    def forward(self, x):
        # 输入是[Batch, 256, 32, 32]
        de = []
        x1 = self.de_resnet.layer1(x) 
        de.append(x1)
        x2 = self.de_resnet.layer2(x1) 
        de.append(x2)
        x3 = self.de_resnet.layer3(x2)
        de.append(x3)
        return de

class ADNet(BaseNetwork):
    def __init__(self, 
                 cfg: BaseConfig,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg) # 换为非抗锯齿，感觉挺好的，主要是因为蒸馏的能力弱，拉大差距，编码器抗锯齿反而不太好
        self.decoder = Decoder(cfg)
    
    def forward(self, y_img, y_noise_img, x_img) -> Any:
        # Batch 代表 dataloder自己取的时候，默认都为1
        # SelfBatch 代表 自定义的，已经取好了
        # query_img.shape = (Batch,SelfBatch,3,224,224)
        # support_img_list.shape: (Batch,SelfBatch,shot,3,224,224)
        # B = 2 会出错，在dcn处
        y_img = y_img.squeeze(0)
        B,C,H,W = y_img.shape
        y_noise_img = y_noise_img.squeeze(0)
        B,C,H,W = y_noise_img.shape
        x_img = x_img.squeeze(0)
        B,K,C,H,W = x_img.shape 
        # x_img = x_img.view(B*K, C, H, W) # 批次不重要，多点好=>显存不够
        imagenet_norm_batch(y_img)
        en, en_noise, proj, proj_noise, out  = self.encoder(y_img, y_noise_img)
        # oce = self.ocbe_l3(en_list)
        de = self.decoder(out) # en_list[2]
        # 调整顺序(同解码顺序)
        en = [en[0], en[1], en[2]]
        en = [de[2], de[1], de[0]]
        return en, de, en_noise, proj, proj_noise

def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

class Revisit_RDLoss(nn.Module):
    """
    receive multiple inputs feature
    return multi-task loss:  SSOT loss, Reconstruct Loss, Contrast Loss
    """
    def __init__(self, consistent_shuffle = True):
        super(Revisit_RDLoss, self).__init__()
        self.sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05, \
                              reach=None, diameter=10000000, scaling=0.95, \
                                truncate=10, cost=None, kernel=None, cluster_scale=None, \
                                  debias=True, potentials=False, verbose=False, backend='auto')
        self.reconstruct = CosineReconstructLoss()       
        self.contrast = torch.nn.CosineEmbeddingLoss(margin = 0.5)
    def forward(self, noised_feature, projected_noised_feature, projected_normal_feature):
        """
        noised_feature : output of encoder at each_blocks : [noised_feature_block1, noised_feature_block2, noised_feature_block3]
        projected_noised_feature: list of the projection layer's output on noised_features, projected_noised_feature = projection(noised_feature)
        projected_normal_feature: list of the projection layer's output on normal_features, projected_normal_feature = projection(normal_feature)
        """
        current_batchsize = projected_normal_feature[0].shape[0]

        target = -torch.ones(current_batchsize).to('cuda')

        normal_proj1 = projected_normal_feature[0]
        normal_proj2 = projected_normal_feature[1]
        normal_proj3 = projected_normal_feature[2]
        # shuffling samples order for caculating pair-wise loss_ssot in batch-mode , (for efficient computation)
        shuffle_index = torch.randperm(current_batchsize)
        # Shuffle the feature order of samples in each block
        shuffle_1 = normal_proj1[shuffle_index]
        shuffle_2 = normal_proj2[shuffle_index]
        shuffle_3 = normal_proj3[shuffle_index]

        abnormal_proj1, abnormal_proj2, abnormal_proj3 = projected_noised_feature
        noised_feature1, noised_feature2, noised_feature3 = noised_feature
        loss_ssot = self.sinkhorn(torch.softmax(normal_proj1.view(normal_proj1.shape[0], -1), -1), torch.softmax(shuffle_1.view(shuffle_1.shape[0], -1),-1)) +\
               self.sinkhorn(torch.softmax(normal_proj2.view(normal_proj2.shape[0], -1),-1),  torch.softmax(shuffle_2.view(shuffle_2.shape[0], -1),-1)) +\
               self.sinkhorn(torch.softmax(normal_proj3.view(normal_proj3.shape[0], -1),-1),  torch.softmax(shuffle_3.view(shuffle_3.shape[0], -1),-1))
        loss_reconstruct = self.reconstruct(abnormal_proj1, normal_proj1)+ \
                   self.reconstruct(abnormal_proj2, normal_proj2)+ \
                   self.reconstruct(abnormal_proj3, normal_proj3)
        loss_contrast = self.contrast(noised_feature1.view(noised_feature1.shape[0], -1), normal_proj1.view(normal_proj1.shape[0], -1), target = target) +\
                           self.contrast(noised_feature2.view(noised_feature2.shape[0], -1), normal_proj2.view(normal_proj2.shape[0], -1), target = target) +\
                           self.contrast(noised_feature3.view(noised_feature3.shape[0], -1), normal_proj3.view(normal_proj3.shape[0], -1), target = target)
        return (loss_ssot + 0.01 * loss_reconstruct + 0.1 * loss_contrast)/1.11

class ADLoss(nn.Module):
    def __init__(self):
        super(ADLoss, self).__init__()
        self.cos = CosineLoss()
        self.proj_loss = Revisit_RDLoss()
    def forward(self, **kwargs):
        # 解析包
        en_list = kwargs['out_0']
        de_list = kwargs['out_1']
        en_noise_list = kwargs['out_2']
        proj_list = kwargs['out_3']
        proj_noise_list = kwargs['out_4']
        # 计算损失
        # detach 用于 分离梯度图
        # en_list[0] = en_list[0].detach()
        # en_list[1] = en_list[1].detach()
        # en_list[2] = en_list[2].detach()
        # input = input.detach()
        
        loss_proj = self.proj_loss(en_noise_list, proj_noise_list, proj_list)
        
        # l1
        # loss_l1 = self.l1(output, input) # 目标放后面
        # loss_l1 = 0 # 不计算梯度
        # hard
        distance_l0 = torch.pow(en_list[0]-de_list[0],2)
        dhard_l0 = torch.quantile(distance_l0[:,:,:,:],0.999) # 0.999分位数
        hard_data_l0 = distance_l0[distance_l0>=dhard_l0] 
        Lhard_l0 = torch.mean(hard_data_l0) 
        distance_l1 = torch.pow(en_list[1]-de_list[1],2)
        dhard_l1 = torch.quantile(distance_l1[:,:,:,:],0.999) # 0.999分位数
        hard_data_l1 = distance_l1[distance_l1>=dhard_l1] 
        Lhard_l1 = torch.mean(hard_data_l1) 
        distance_l2 = torch.pow(en_list[2]-de_list[2],2)
        dhard_l2 = torch.quantile(distance_l2[:,:,:,:],0.999) # 0.999分位数
        hard_data_l2 = distance_l2[distance_l2>=dhard_l2] 
        Lhard_l2 = torch.mean(hard_data_l2) 
        loss_hard = Lhard_l0 + Lhard_l1 + Lhard_l2
        # soft
        # mse不需要
        # mse_loss = torch.mean(torch.pow(en - de, 2))
        # loss_mse_l0 = self.mse(de_list[0], en_list[0])
        # loss_mse_l1 = self.mse(de_list[1], en_list[1])
        # loss_mse_l2 = self.mse(de_list[2], en_list[2])
        # loss_mse = loss_mse_l0 + loss_mse_l1 + loss_mse_l2
        
        # cos需要归一化
        # en_list[0] = F.normalize(en_list[0], p=1, dim=1)  # L1 归一化
        # en_list[1] = F.normalize(en_list[1], p=1, dim=1)  # L1 归一化
        # en_list[2] = F.normalize(en_list[2], p=1, dim=1)  # L1 归一化
        # de_list[0] = F.normalize(de_list[0], p=1, dim=1)  # L1 归一化
        # de_list[1] = F.normalize(de_list[1], p=1, dim=1)  # L1 归一化
        # de_list[2] = F.normalize(de_list[2], p=1, dim=1)  # L1 归一化
        loss_cos_l0 = self.cos(de_list[0], en_list[0])
        loss_cos_l1 = self.cos(de_list[1], en_list[1])
        loss_cos_l2 = self.cos(de_list[2], en_list[2])
        loss_cos = loss_cos_l0 + loss_cos_l1 + loss_cos_l2
        # loss_cos = loss_fucntion(en_list, de_list)
        
        # loss = loss_l1 + loss_cos
        # print('loss is {:.2f} = loss_l1 is {:.2f} + loss_cos is {:.2f}({:.2f}, {:.2f}, {:.2f})' \
        #       .format(loss, loss_l1, 
        #               loss_cos, loss_cos_l0, loss_cos_l1, loss_cos_l2))
        loss = loss_cos + loss_proj + loss_hard * 0.01
        print('loss is {:.2f} = loss_cos is {:.2f}({:.2f}, {:.2f}, {:.2f}) + loss_hard is {:.2f}({:.2f}, {:.2f}, {:.2f})' \
              .format(loss,
                      loss_cos, loss_cos_l0, loss_cos_l1, loss_cos_l2,
                      loss_hard, Lhard_l0, Lhard_l1, Lhard_l2))
        # loss = loss_l1 + loss_mse + loss_cos
        # print('loss is {:.2f} = loss_l1 is {:.2f} + loss_mse is {:.2f}({:.2f}, {:.2f}, {:.2f}) + loss_cos is {:.2f}({:.2f}, {:.2f}, {:.2f})' \
        #       .format(loss, loss_l1,
        #               loss_mse, loss_mse_l0, loss_mse_l1, loss_mse_l2,
        #               loss_cos, loss_cos_l0, loss_cos_l1, loss_cos_l2))
        return loss
