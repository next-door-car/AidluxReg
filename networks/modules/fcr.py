import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.models.pwcnn_stn import PWCNN_STN
from networks.modules.dwt import DWT_DSCNN, IWT_DSCNN
from networks.modules.memory import Memory
from base.base_config import BaseConfig
from loggers.logger import get_root_logger

try:
    from networks.modules.ops.dcn.deform_conv import (ModulatedDeformConvPack, modulated_deform_conv) 
except ImportError:
    print('Cannot import dcn. Ignore this warning if dcn is not used. '
          'Otherwise install BasicSR with compiling dcn.')
    ModulatedDeformConvPack = object
    modulated_deform_conv = None

class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """
    def __init__(self, *args, **kwargs):
        super(DCNv2Pack, self).__init__(*args, **kwargs)

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        offset_absmean = torch.mean(torch.abs(offset)) # 偏移量度量
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(
                f'Offset abs mean is {offset_absmean}, larger than 50.') # 偏移量代表着偏移量过大

        return modulated_deform_conv(x, offset, mask, 
                                     self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)

class FCA(nn.Module):
    def __init__(self, 
                 cfg: BaseConfig,
                 align_feat=64, 
                 deformable_groups=8,
                 **kwargs):
        super().__init__()
        
        self.cfg = cfg
        # 每个L级别都要经历=>conv1 conv2 conv3
        self.offset_conv1 = nn.ModuleDict() # cat nbr
        self.offset_conv2 = nn.ModuleDict() # cat np
        self.offset_conv3 = nn.ModuleDict() # project
        self.memory = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict() # dcn_feat
        self.feat_conv = nn.ModuleDict() # cat dcn+np
        self.iwt_dscnn = nn.ModuleDict()

        # Pyramids
        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        # feat_conv
        # feat_conv[l1] (128, 64, 3, 1)
        # feat_conv[l2] (128, 64, 3, 1)
        # offset_conv1
        # offset_conv1[l1] (128, 64, 3, 1)
        # offset_conv1[l2] (128, 64, 3, 1)
        # offset_conv1[l3] (128, 64, 3, 1)
        # offset_conv2
        # offset_conv2[l1] (128, 64, 3, 1)
        # offset_conv2[l2] (128, 64, 3, 1)
        # offset_conv2[l3] (128, 64, 3, 1) # L3级无conv2
        # offset_conv3
        # offset_conv3[l1] (64, 64, 3, 1)
        # offset_conv3[l2] (64, 64, 3, 1)
        # offset_conv3[l3] (64, 64, 3, 1) 
        for i in range(3, 0, -1): # range(3, 0, -1)顺序是3,2,1
            level = f'l{i}'
            # offset
            ## 每层都需要cat nbr
            self.offset_conv1[level] = nn.Conv2d(align_feat * 2, align_feat, 3, 1, 1) # shape (num_feat * 2, 1/4, 1/4) 
            ## 每层都需要记忆模块
            self.memory[level] = Memory(mem_dim=256, fea_dim=align_feat)
            if i != 3:
                ## 顶层需要cat np
                self.offset_conv2[level] = nn.Conv2d(align_feat * 2, align_feat, 3, 1, 1)
                self.offset_conv3[level] = nn.Conv2d(align_feat, align_feat, 3, 1, 1)
            if i == 3:
                ## L3级 映射,没有融合层 cat np
                self.offset_conv3[level] = nn.Conv2d(align_feat, align_feat, 3, 1, 1) # shape (num_feat, 1/4, 1/4) 
            # dcn_lfreq_feat
            self.dcn_pack[level] = DCNv2Pack(
                  in_channels=align_feat,
                  out_channels=align_feat,
                  kernel_size=3,
                  padding=1,
                  deformable_groups=deformable_groups
            )
            # feat
            if i < 3:
                self.feat_conv[level] = nn.Conv2d(align_feat * 2, align_feat, 3, 1, 1)
            # act+np
            if i > 1:
                self.upsample = nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=False)
                self.iwt_dscnn[level] = IWT_DSCNN(feat=align_feat)
                self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True) # 激活
            else:
                # Cascading dcn i=1
                self.cas_offset_conv1 = nn.Conv2d(align_feat * 2, align_feat, 3, 1, 1) # 融合
                self.cas_offset_conv2 = nn.Conv2d(align_feat, align_feat, 3, 1, 1) # 映射
                self.cas_dcnpack = DCNv2Pack(
                    align_feat,
                    align_feat,
                    3,
                    padding=1,
                    deformable_groups=deformable_groups) # dcn_feat
                break        

    def forward(self, ref_feat_l, nbr_feat_l):
        # Pyramids 融合上采样
        # 特征不同的卷积层
        # feat_conv
        # feat_conv[l1] (128, 64, 3, 1) # size-3+2 / 1 + 1 = size
        # feat_conv[l2] (128, 64, 3, 1)
        # offset_conv1 => 融合nbr特征
        # offset_conv1[l1] (128, 64, 3, 1)
        # offset_conv1[l2] (128, 64, 3, 1)
        # offset_conv1[l3] (128, 64, 3, 1)
        # offset_conv2 => 融合up特征
        # offset_conv2[l1] (128, 64, 3, 1)
        # offset_conv2[l2] (128, 64, 3, 1)
        # offset_conv2[l3] (128, 64, 3, 1) # L3级无conv2
        # offset_conv3 => 映射特征（64=>64）
        # offset_conv3[l1] (64, 64, 3, 1)
        # offset_conv3[l2] (64, 64, 3, 1)
        # offset_conv3[l3] (64, 64, 3, 1) 
        up_offset, iwt_feat = None, None
        for i in range(3, 0, -1): # L3, L2, L1 即 3，2，1
            # 跨步卷积得到L级别金字塔
            level = f'l{i}'
            
            # offset 操作
            ## 每层都需要cat nbr
            cat_nb_offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1) # 取最后一级（层）的特征图，dim=1沿着batch方向拼接
            nb_offset = self.lrelu(self.offset_conv1[level](cat_nb_offset)) # 128 => 64
            if i != 3:
                ## 不是底层则需要cat up
                integrated_offset = torch.cat([nb_offset, up_offset], dim=1)
                offset = self.lrelu(self.offset_conv2[level](integrated_offset)) # 128 => 64
                offset = self.lrelu(self.offset_conv3[level](offset)) # 64 => 64
            if i == 3:
                ## 底层 不需要cat,再经过一层卷积
                offset = self.lrelu(self.offset_conv3[level](nb_offset)) # conv2 用来融合的
            # 经过记忆模块
            weight, offset = self.memory[level](offset)
            # dcn_feat
            dcn_feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset) # dcn_lfreq_feat 64
            
            # feat 操作
            if i < 3:
                # 顶层级别都要cat
                cat_dcn_iwt_feat = torch.cat([dcn_feat, iwt_feat], dim=1) # 2 * 64 (低频＋高频激活)
                feat = self.feat_conv[level](cat_dcn_iwt_feat) # 2feat => feat
            else:
                # i=3 底层不要cat了
                feat = dcn_feat
                
            # act 操作
            if i > 1: # 还没结束循环
                # 没有到顶层
                iwt_feat = self.lrelu(self.iwt_dscnn[level](feat)) # 补个卷积
                # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                up_offset = self.upsample(offset) * 2 # 不太懂
                
            else:
                # Cascading 顶层=> dcn 或 上采样的两种处理
                # 顶层 i=1 不用做激活（即将退出循环了）
                # feat 不需要激活
                # cat_feat_offset = torch.cat([feat, ref_feat_l[0]], dim=1) # L0级别基准图像
                # fuse_offset_feat = self.lrelu(self.cas_offset_conv1(cat_feat_offset)) # 融合 offset
                # proj_offset_feat = self.lrelu(self.cas_offset_conv2(fuse_offset_feat)) # 映射 offset
                # out_feat = self.lrelu(self.cas_dcnpack(feat, proj_offset_feat)) # dcn_feat
                out_feat = feat # 不要级联层l了
                break
            
        return out_feat

class FCR(nn.Module):
    def __init__(self,
                 cfg: BaseConfig,
                 align_feat_num=64,
                 align_feat_size=256,
                 deformable_groups=8,
                 with_bn=False,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.shot = cfg.get_config(['datasets', 'shot'])
        self.pwcnn_stn = PWCNN_STN(cfg=cfg, feat_num=align_feat_num, feat_size=align_feat_size, with_bn=with_bn)
        self.fca = FCA(cfg=cfg, align_feat=align_feat_num, deformable_groups=deformable_groups) # 特征级联对齐
        self.fusion = nn.Conv2d(self.shot * align_feat_num, align_feat_num, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def forward(self, ref, nbr):
        # ref.shape: (B,  C,H,W)
        # nbr.shape: (B×K,C,H,W)
        B, C, H, W = ref.shape
        BK, C, H, W = nbr.shape
        K = int(BK/B)
        _, ref_feat1, ref_feat2, ref_feat3 = self.pwcnn_stn(ref) # cpu上
        _, nbr_feat1, nbr_feat2, nbr_feat3 = self.pwcnn_stn(nbr) # cpu上
        ref_feat_l = [
            ref_feat1.view(B, 1, C, ref_feat1.size(2), ref_feat1.size(3)).to(ref.device),
            ref_feat2.view(B, 1, C, ref_feat2.size(2), ref_feat2.size(3)).to(ref.device),
            ref_feat3.view(B, 1, C, ref_feat3.size(2), ref_feat3.size(3)).to(ref.device)
        ]
        nbr_feat_l = [
            nbr_feat1.view(B, K, C, nbr_feat1.size(2), nbr_feat1.size(3)).to(nbr.device),
            nbr_feat2.view(B, K, C, nbr_feat2.size(2), nbr_feat2.size(3)).to(nbr.device),
            nbr_feat3.view(B, K, C, nbr_feat3.size(2), nbr_feat3.size(3)).to(nbr.device)
        ]
    
        assert 1 == ref_feat_l[0].shape[1]
        assert self.shot == nbr_feat_l[0].shape[1]
        
        ref_ext = [any,any,any]
        nbr_ext = [any,any,any]
        align_feat = []
        for L in range(len(ref_feat_l)):
            ref_ext[L] = ref_feat_l[L][:, 0, :, :, :]
        for i in range(self.shot):
            for L in range(len(nbr_feat_l)):
                nbr_ext[L] = nbr_feat_l[L][:, i, :, :, :] 
            align_feat.append(self.fca(ref_ext, nbr_ext))
        align_feat = torch.stack(align_feat, dim=1)  # (b, shot, c, h, w)
        fusion_feat = align_feat.view(B, -1, align_feat.size(3), align_feat.size(4))
        reco_feat = self.lrelu(self.fusion(fusion_feat))
        return reco_feat, align_feat.view(B*K, C, H, W)