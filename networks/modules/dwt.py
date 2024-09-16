import torch
import torch.nn as nn
import torch.nn.functional as F


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class DWT_DSCNN(nn.Module):
    def __init__(self, 
                 in_feat = 64, 
                 out_feat = 128, 
                 groups = 4):
        super().__init__()
        self.dwt = DWT()
        self.groupConv = nn.Conv2d(in_channels=in_feat*4, out_channels=out_feat, 
                                   kernel_size=3, stride=1, padding=1,  
                                   groups=groups)  # 分组融合
        # 4 是 LL, LH, HL, HH 通道数
        # in_channels/4为1组，每组产生out_channels/4个通道
        # depthConv 卷积核参数为 3*3*(in_channels/groups)*(out_channels/groups) 增大了4倍参数
        self.pointConv = nn.Conv2d(out_feat, out_feat, 
                                   kernel_size=1, stride=1, padding=0) # 频域融合

    def forward(self, x):
        x = self.dwt(x)
        x = self.groupConv(x)
        x = self.pointConv(x)
        return x

class IWT_DSCNN(nn.Module):
    def __init__(self, 
                 in_feat = 64*4,
                 out_feat = 64*4,
                 groups = 4):
        super().__init__()
        self.iwt = IWT()
        self.depthConv = nn.Conv2d(in_channels=in_feat, out_channels=out_feat, 
                                   kernel_size=3, stride=1, padding=1,  
                                   groups=groups) # 此处分组融合
        # 4 是 LL, LH, HL, HH 通道数
        # in_channels/4为1组，每组产生out_channels/4个通道
        # depthConv 卷积核参数为 3*3*(in_channels/groups)*(out_channels/groups) 增大了4倍参数
        self.pointConv = nn.Conv2d(out_feat, out_feat,
                                   kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        # x.shape = (B, 64*4, 256, 256)
        # 整合
        x = self.depthConv(x)
        x = self.pointConv(x) # * 4
        x = self.iwt(x) # /4 x的特征是64
        return x

def dwt_init(x):
    '''
    2D DWT 作用是降采样，生成4个子特征图像
    '''
    x01 = x[:, :, 0::2, :] / 2 # 0:空:2 表示隔2个取值
    x02 = x[:, :, 1::2, :] / 2 # 1:空:2 表示隔2个取值
    x1 = x01[:, :, :, 0::2] # 隔2个取值
    x2 = x02[:, :, :, 0::2] # 隔2个取值
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4 # x_LL （更低级细节特征）尺度模糊严重,相当于平均池化（2*2的核）
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    # return torch.cat((x_LL, x_HL, x_LH, x_HH), 1) # 下采样
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1) # 下采样

def iwt_init(x):
    r = 2 # r的作用是因为下采样是2倍
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width 
    # out_height 和 out_width 是输出图像的高度和宽度，它们分别是输入图像高度和宽度的 r 倍，这表明在降采样后，代码意图通过某种方式恢复（或上采样）图像的尺寸。
    x1 = x[:, 0:out_channel, :, :] / 2 
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h # 恢复尺度（上采样）