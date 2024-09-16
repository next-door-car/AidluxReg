from base.base_config import BaseConfig
from base.base_network import BaseNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type, Union, List, Optional, Callable

from networks.modules.memory import Memory
from networks.modules.proto import ProtoMemory

# from antialiased_cnns import wide_resnet50_2
from networks.models.resnet import wide_resnet50_2
from networks.models.de_resnet import de_wide_resnet50_2
from networks.modules.ocbe import OCBE_l1, OCBE_l2, OCBE_l3

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.ConvTranspose2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)


def imagenet_norm_batch(x):
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to('cuda')
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to('cuda')
    x_norm = (x - mean) / (std + 1e-11)
    return x_norm

# 定义深度可分离卷积模块
class DSConv(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, 
                 kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        # 深度可分离卷积包括深度卷积和逐点卷积两个步骤
        # 分组卷积处理大量通道数，减小模型的参数量
        self.depthConv = nn.Conv2d(in_channels, in_channels, 
                                   kernel_size=kernel_size, stride=stride, padding=padding,  
                                   groups=in_channels) 
        # 4 是 LL, LH, HL, HH 通道数
        # in_channels/4为1组，每组产生out_channels/4个通道
        # depthConv 卷积核参数为 3*3*(in_channels/groups)*(out_channels/groups) 增大了16倍参数
        self.pointConv = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.depthConv(x)
        x = self.pointConv(x)
        return x
    
class Reco(nn.Module):
    def __init__(self, 
                 cfg: BaseConfig,
                 **kwargs):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU(inplace=True)
        # self.upsample = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.dsconv3 = DSConv(256*4, 256)
        self.dsconv2 = DSConv(128*4, 128)
        self.dsconv1 = DSConv(64*4, 64)
        
        self.reduce_conv3 = conv3x3(256, 128) # 128 * 4 = 512
        self.reduce_bn3 = norm_layer(128) 
        
        self.concat_conv_23 = conv3x3(256, 128)
        self.concat_bn_23 = norm_layer(128)
        self.fusion_conv_23 = conv3x3(128, 128) # 128 * 4 = 512
        self.fusion_bn_23 = norm_layer(128)
        
        self.reduce_conv2 = conv3x3(128, 64)
        self.reduce_bn2 = norm_layer(64)
        
        self.concat_conv_12 = conv3x3(128, 64)
        self.concat_bn_12 = norm_layer(64)
        self.fusion_conv_12 = conv3x3(64, 64) # 128 * 4 = 512
        self.fusion_bn_12 = norm_layer(64) # [b,64,64,64]
        
        self.upsample_last = nn.PixelShuffle(2) # [b,16,128,128]
        self.last_deconv = nn.ConvTranspose2d(16, 3, 
                                            kernel_size=3, stride=2,    
                                            padding=1, output_padding=1, bias=False) # 254-2+3+1 = 256
        self.last_bn = norm_layer(3) 
        
        
    def forward(self, x: List):
        x0 = x[0].detach()
        x1 = x[1].detach()
        x2 = x[2].detach()
        feat3 = self.dsconv3(x0) # b,64,64,64 
        feat2 = self.dsconv2(x1) # b,128,32,32
        feat1 = self.dsconv1(x2) # b,256,16,16
        
        up_reduce_feat3 = self.relu(self.reduce_bn3(self.reduce_conv3(self.upsample(feat3)))) # b,128,32,32
        
        cat_feat23 = torch.cat([feat2, up_reduce_feat3], dim=1)
        feat23 = self.relu(self.fusion_bn_23(self.fusion_conv_23(
                    self.relu(self.concat_bn_23(self.concat_conv_23(cat_feat23)))))) # b,128,32,32
        
        up_reduce_feat2 = self.relu(self.reduce_bn2(self.reduce_conv2(self.upsample(feat23)))) # b,64,64,64
        
        cat_feat12 = torch.cat([feat1, up_reduce_feat2], dim=1)
        feat12 = self.relu(self.fusion_bn_12(self.fusion_conv_12(
                    self.relu(self.concat_bn_12(self.concat_conv_12(cat_feat12)))))) # b,64,64,64
        
        up_reduce_feat1= self.upsample_last(feat12) # b,16,128,128
        
        out = self.last_bn(self.last_deconv(up_reduce_feat1))
        
        return out 
    
class Encoder(nn.Module):
    '''
    仅修改S
    '''
    def __init__(self,
                 cfg: BaseConfig,
                 **kwargs) -> None:
        super().__init__()
        self.cfg = cfg
        self.resnet, self.bn = wide_resnet50_2(pretrained=True)
    def forward(self, x):
        # 输入是[Batch, 3, 256, 256]
        ret = []
        x0 = self.resnet.conv1(x) # b,64,128,128
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        x0 = self.resnet.maxpool(x0) # shape: [Batch, 64, 64, 64]
        
        x1 = self.resnet.layer1(x0) # shape: [Batch, 64, 64, 64]
        ret.append(x1) 
        
        x2 = self.resnet.layer2(x1) # shape: [Batch, 128, 32, 32]
        ret.append(x2) 
        
        x3 = self.resnet.layer3(x2) # shape: [Batch, 256, 16, 16]
        ret.append(x3) 
        
        out = self.bn(ret) # shape: [B, 512*expansion, 8, 8]

        return ret, out 
    
class Decoder(nn.Module):
    def __init__(self,
                 cfg: BaseConfig,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.de_resnet = de_wide_resnet50_2(pretrained=False)
        self.reco = Reco(cfg)
        # self.memory_l1 = Memory(ch=256*4, feat=256,
        #                         which_conv=nn.Conv2d,
        #                         mem_dim=500,fea_dim=256,hidden=120)
        # self.memory_l2 = Memory(ch=128*4, feat=128,
        #                         which_conv=nn.Conv2d,
        #                         mem_dim=500,fea_dim=128,hidden=60)
        # self.memory_l3 = Memory(ch=64*4, feat=64,
        #                         which_conv=nn.Conv2d,
        #                         mem_dim=500,fea_dim=64,hidden=30)
        self.memory_l1 = ProtoMemory(ch=256*4, feat=256,
                                     init_num_k=200, init_pool_size_per_cluster=10, 
                                     warmup_total_iter=500.0,
                                     which_conv=nn.Conv2d,
                                     cp_momentum=1, 
                                     cp_phi_momentum=0.6, 
                                     device='cuda:0')
        self.memory_l2 = ProtoMemory(ch=128*4, feat=128,
                                     init_num_k=200, init_pool_size_per_cluster=10, 
                                     warmup_total_iter=500.0,
                                     which_conv=nn.Conv2d,
                                     cp_momentum=1, 
                                     cp_phi_momentum=0.6, 
                                     device='cuda:0')
        self.memory_l3 = ProtoMemory(ch=64*4, feat=64,
                                     init_num_k=200, init_pool_size_per_cluster=10, 
                                     warmup_total_iter=500.0,
                                     which_conv=nn.Conv2d,
                                     cp_momentum=1, 
                                     cp_phi_momentum=0.6, 
                                     device='cuda:0')        
        # self.memory_l1 = MomemtumConceptAttentionProto(ch=256*4, feature_dim=256,
        #                                         num_k=300, pool_size_per_cluster=10, 
        #                                         warmup_total_iter=500.0,
        #                                         which_conv=nn.Conv2d,
        #                                         cp_momentum=1, 
        #                                         cp_phi_momentum=0.6, 
        #                                         device='cuda:0')
        # self.memory_l2 = MomemtumConceptAttentionProto(ch=128*4, feature_dim=128,
        #                                         num_k=300, pool_size_per_cluster=10, 
        #                                         warmup_total_iter=500.0,
        #                                         which_conv=nn.Conv2d,
        #                                         cp_momentum=1, 
        #                                         cp_phi_momentum=0.6, 
        #                                         device='cuda:0')
        # self.memory_l3 = MomemtumConceptAttentionProto(ch=64*4, feature_dim=64,
        #                                         num_k=300, pool_size_per_cluster=10, 
        #                                         warmup_total_iter=500.0,
        #                                         which_conv=nn.Conv2d,
        #                                         cp_momentum=1, 
        #                                         cp_phi_momentum=0.6, 
        #                                         device='cuda:0')        
        # self.last_conv = nn.Conv2d(64*4, 3, 1)
    def forward(self, x):
        # 输入是[Batch, 256, 32, 32]
        ret_rd = []
        x1 = self.de_resnet.layer1(x) # [B, 256, 16, 16] 
        ret_rd.append(x1) 
        x2 = self.de_resnet.layer2(x1) # [B, 128, 32, 32]
        ret_rd.append(x2) 
        x3 = self.de_resnet.layer3(x2) # [B, 64, 64, 64]
        ret_rd.append(x3) 
        
        ret_reco = []
        feat1 = x1.detach()
        feat1 = self.memory_l1(feat1) # [B, 256, 16, 16] 
        ret_reco.append(feat1)
        feat2 = x2.detach()
        feat2 = self.memory_l2(feat2) # [B, 128, 32, 32]
        ret_reco.append(feat2)
        feat3 = x3.detach()
        feat3 = self.memory_l3(feat3) # [B, 64, 64, 64]
        ret_reco.append(feat3)
        out = self.reco(ret_reco)
        
        return ret_rd, ret_reco, out

class ADNet(BaseNetwork):
    def __init__(self, 
                 cfg: BaseConfig,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg) # 换为非抗锯齿，感觉挺好的，主要是因为蒸馏的能力弱，拉大差距，编码器抗锯齿反而不太好
        # self.ocbe_l3 = OCBE_l3(cfg, block, blocks) # l3直接注入可能会有问题（损失高）
        self.decoder = Decoder(cfg)
    
    def forward(self, y_img, x_img) -> tuple:
        # Batch 代表 dataloder自己取的时候，默认都为1
        # SelfBatch 代表 自定义的，已经取好了
        # query_img.shape = (Batch,SelfBatch,3,224,224)
        # support_img_list.shape: (Batch,SelfBatch,shot,3,224,224)
        # B = 2 会出错，在dcn处
        y_img = y_img.squeeze(0)
        B,C,H,W = y_img.shape
        x_img = x_img.squeeze(0)
        B,K,C,H,W = x_img.shape 
        # x_img = x_img.view(B*K, C, H, W) # 批次不重要，多点好=>显存不够
        # imagenet_norm_batch(y_img)
        en_list, en_out = self.encoder(y_img)
        # oce = self.ocbe_l3(en_list)
        de_rd_list, de_reco_list, de_out = self.decoder(en_out) # en_list[2]
        # 调整顺序(同解码顺序)
        en_list = [en_list[0], en_list[1], en_list[2]]
        de_rd_list = [de_rd_list[2], de_rd_list[1], de_rd_list[0]]
        de_reco_list = [de_reco_list[2], de_reco_list[1], de_reco_list[0]]
        return en_list, de_rd_list, de_reco_list, y_img, de_out

class ADLoss(nn.Module):
    def __init__(self):
        super(ADLoss, self).__init__()
    def forward(self):
        pass
