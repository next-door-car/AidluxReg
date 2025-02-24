import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2

def get_pad_layer(pad_type):
    '''
    填充层
    '''
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

def conv_identify(weight, bias):
    weight.data.zero_()
    if bias is not None:
        bias.data.zero_()
    o, i, h, w = weight.shape
    y = h//2
    x = w//2
    for p in range(i):
        for q in range(o):
            if p == q:
                weight.data[q, p, :, :] = 1.0

class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat(self.channels,1,1,1))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

class Downsample_PASA_group_softmax(nn.Module):
    def __init__(self, 
                 in_channels, 
                 pad_type='reflect',
                 kernel_size=1, stride=2, group=2):
        '''
        kernel_size 保持默认即可
        '''
        super().__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size//2) # 填充层（保证尺度对应）
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        
        # 自适应低通滤波器
        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, 
                              kernel_size=kernel_size, stride=1, padding=0, # 此处是默认
                              bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)

        n,c,h,w = sigma.shape
 
        sigma = sigma.reshape(n,1,c,h*w) # sigma作用是卷积核的权重，所以要reshape成(n,1,c,h*w)
        
        # c = c1 = c2 不一定相等！
        n,c,h,w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))

        n,c1,p,q = x.shape # p q 分别代表卷积核大小，p是卷积核大小，q是图像大小
        # x.permute(1,0,2,3) = (c1,n,p,q)
        # reshape = (group, c1//group, n, p, q)
        # permute = (n, group, c1//group, p, q)
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4) # 将c1分组

        n,c2,p,q = sigma.shape
        # sigma.permute(1,0,2,3) = (c2,n,p,q)
        # reshape = (group, c2//group, n, p, q)
        # permute = (n, group, c2//group, p, q)
        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)

        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w) # 融合

        # torch.arange(h) 是 PyTorch 中的一个函数调用，用于创建一个包含从 0 到 h-1 的整数序列的一维张量（tensor）。
        return x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0] # 作用是缩小输出大小即滤波器大小 下采样
