import torch
from torch import nn
from torch.nn import init 
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Type, Union, List, Optional, Callable


from base.base_config import BaseConfig
from networks.modules.dwt import *
from networks.models.de_resnet import deconv2x2
from antialiased_cnns.resnet import conv1x1, conv3x3, BlurPool, BasicBlock, Bottleneck

@torch.no_grad()
def default_init_weights(module_list, scale=1.0, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list] # 转换为列表
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale # 卷积核初始化，scale代表缩放初始化权重的比例因子
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill) # 偏置初始化
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs) # 线性层初始化
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill) # 偏置初始化
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1) # BN初始化
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill) # 偏置初始化

class OCBE_l1(nn.Module):
    def __init__(self,
                 cfg: BaseConfig,
                 block, blocks: int,
                 groups: int = 4, 
                 width_per_group: int = 64,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 **kwargs) -> None:
        super(OCBE_l1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.cfg = cfg
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 64 * 3 * block.expansion # 输入
        self.dilation = 1
        self.block_extract_layer = self._make_layer(block, 64, blocks) # 64 * block.expansion，不下采样

        self.relu = nn.ReLU(inplace=True)
        self.bn = norm_layer(256) # 输出是256

        # self.l1 = nn.Sequential(*[
        #     conv3x3(64*block.expansion, 64*block.expansion, 1), 
        #     norm_layer(64*block.expansion),
        #     nn.ReLU(inplace=True)])
        
        self.l2 = nn.Sequential(
            deconv2x2(128 * block.expansion, 64 * block.expansion, 2),
            norm_layer(64*block.expansion),
            nn.ReLU(inplace=True))
        
        self.l3 = nn.Sequential(
            deconv2x2(256 * block.expansion, 128 * block.expansion, 2),
            norm_layer(128 * block.expansion),
            nn.ReLU(inplace=True),
            deconv2x2(128 * block.expansion, 64 * block.expansion, 2),
            norm_layer(64 * block.expansion),
            nn.ReLU(inplace=True))
        
        default_init_weights([self.block_extract_layer], 0.1)

    def _make_layer(self, 
                    block: Type[Union[BasicBlock, Bottleneck]], 
                    planes: int, 
                    blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        downsample = None
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride # 目的是为了下采样
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion: # 256*3 != 256*4
            # 目的是为了维度相同再连接
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer)) # 首层进行了融合
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: List):
        # See note [paddleScript super()]
        # x[0].shape = (B, 64, 256, 256)
        # x[1].shape = (B, 128, 128, 128)
        # x[2].shape = (B, 256, 64, 64)
        # 融合 （B,512,32,32）
        # 可以上个互注意力
        # x = self.cbam(x)
        # l1 = self.l1(x[0])
        l2 = self.l2(x[1])
        l3 = self.l3(x[2])
        mff = torch.concat([x[0], l2, l3], 1) # 拼接,feature.shape = (B, 256*3, 32, 32)
        oce = self.block_extract_layer(mff) # 作用紧凑特征层，shape = (B, 256*4, 32, 32)
        # x = self.avgpool(feature_d)
        # x = paddle.flatten(x, 1)
        # x = self.fc(x)
        return oce

    def forward(self, x):
        return self._forward_impl(x)

class OCBE_l2(nn.Module):
    def __init__(self,
                 cfg: BaseConfig,
                 block, blocks: int,
                 groups: int = 4, 
                 width_per_group: int = 64,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 **kwargs) -> None:
        super(OCBE_l2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.cfg = cfg
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 128 * 3 * block.expansion # 输入
        self.dilation = 1
        self.block_extract_layer = self._make_layer(block, 128, blocks) # 输出 128 * block.expansion，不下采样

        self.relu = nn.ReLU(inplace=True)
        self.bn = norm_layer(256) # 输出是256

        self.l1 = nn.Sequential(
            conv3x3(64 * block.expansion, 128 * block.expansion, 2), 
            norm_layer(128 * block.expansion),
            nn.ReLU(inplace=True))
        
        # self.l2 = nn.Sequential(
        #     conv3x3(128 * block.expansion, 128 * block.expansion, 1), 
        #     norm_layer(128 * block.expansion),
        #     nn.ReLU(inplace=True))
        
        self.l3 = nn.Sequential(
            deconv2x2(256 * block.expansion, 128 * block.expansion, 2),
            norm_layer(128 * block.expansion),
            nn.ReLU(inplace=True))
        
        default_init_weights([self.block_extract_layer], 0.1)

    def _make_layer(self, 
                    block: Type[Union[BasicBlock, Bottleneck]], 
                    planes: int, 
                    blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        downsample = None
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride # 目的是为了下采样
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion: # 256*3 != 256*4
            # 目的是为了维度相同再连接
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer)) # 首层进行了融合
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: List):
        # See note [paddleScript super()]
        # x[0].shape = (B, 64, 256, 256)
        # x[1].shape = (B, 128, 128, 128)
        # x[2].shape = (B, 256, 64, 64)
        # 融合 （B,512,32,32）
        # 可以上个互注意力
        # x = self.cbam(x)
        l1 = self.l1(x[0])
        # l2 = self.l2(x[1])
        l3 = self.l3(x[2])
        mff = torch.concat([l1, x[1], l3], 1) # 拼接,feature.shape = (B, 128*3, 32, 32)
        oce = self.block_extract_layer(mff) # 作用紧凑特征层，shape = (B, 128*4, 32, 32)
        # x = self.avgpool(feature_d)
        # x = paddle.flatten(x, 1)
        # x = self.fc(x)
        return oce

    def forward(self, x):
        return self._forward_impl(x)


class OCBE_l3(nn.Module):
    def __init__(self,
                 cfg: BaseConfig,
                 block, blocks: int,
                 groups: int = 4, 
                 width_per_group: int = 64,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 **kwargs) -> None:
        super(OCBE_l3, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.cfg = cfg
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * 3 * block.expansion # 输入
        self.dilation = 1
        self.block_extract_layer = self._make_layer(block, 256, blocks) # 输出 256 * block.expansion，不下采样

        self.relu = nn.ReLU(inplace=True)
        self.bn = norm_layer(256) # 输出是256

        self.l1 = nn.Sequential(
            conv3x3(64 * block.expansion, 128 * block.expansion, 2), 
            norm_layer(128 * block.expansion),
            nn.ReLU(inplace=True),
            conv3x3(128 * block.expansion, 256 * block.expansion, 2), 
            norm_layer(256 * block.expansion),
            nn.ReLU(inplace=True))
        
        self.l2 = nn.Sequential(
            conv3x3(128 * block.expansion, 256 * block.expansion, 2), 
            norm_layer(256 * block.expansion),
            nn.ReLU(inplace=True))
        
        # self.l3 = nn.Sequential(
        #     conv3x3(256 * block.expansion, 256 * block.expansion, 1), 
        #     norm_layer(256 * block.expansion),
        #     nn.ReLU(inplace=True))
        
        default_init_weights([self.block_extract_layer], 0.1)

    def _make_layer(self, 
                    block: Type[Union[BasicBlock, Bottleneck]], 
                    planes: int, 
                    blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        downsample = None
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride # 目的是为了下采样
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion: # 256*3 != 256*4
            # 目的是为了维度相同再连接
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer)) # 首层进行了融合
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: List):
        # See note [paddleScript super()]
        # x[0].shape = (B, 64, 256, 256)
        # x[1].shape = (B, 128, 128, 128)
        # x[2].shape = (B, 256, 64, 64)
        # 融合 （B,512,32,32）
        # 可以上个互注意力
        # x = self.cbam(x)
        l1 = self.l1(x[0])
        l2 = self.l2(x[1])
        # l3 = self.l3(x[2])
        mff = torch.concat([l1, l2, x[2]], 1) # 拼接,feature.shape = (B, 256*3, 32, 32)
        oce = self.block_extract_layer(mff) # 作用紧凑特征层，shape = (B, 256*4, 32, 32)
        # x = self.avgpool(feature_d)
        # x = paddle.flatten(x, 1)
        # x = self.fc(x)
        return oce

    def forward(self, x):
        return self._forward_impl(x)

