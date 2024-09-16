import os
import time
from abc import ABC, abstractmethod
from base.base_config import BaseConfig
from typing import Any, Type, Union, List, Optional, Callable

import logging
import torch
import torch.nn as nn
import numpy as np

"""
表示所有神经网络模型的基础结构。它提供了一些通用功能，以便在派生的网络模型中使用
"""
class BaseNetwork(nn.Module):
    """Base class for all neural networks."""

    def __init__(self): # 构造函数，初始化神经网络模型的基础结构。它会调用父类 nn.Module 的构造函数。
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__) # 实例变量，用于记录日志信息。在初始化时，它会创建一个与当前类名相关的 logger 对象，以便在训练过程中记录网络的相关信息。
        
        # 实例变量，用于记录日志信息。在初始化时，它会创建一个与当前类名相关的 logger 对象，以便在训练过程中记录网络的相关信息。
        self.rep_dim = None  # representation dimensionality, i.e. dim of the last layer

    def forward(self, *input):
        """
        Forward pass logic
        :return: Network output
        """
        # 抽象方法，子类必须实现此方法。它定义了前向传播逻辑，即如何从输入计算网络的输出。在这个基类中，前向传播方法被标记为抽象，因为不同的网络模型将具有不同的前向传播逻辑。
        raise NotImplementedError # NotImplementedError代表尚未实现的功能或方法。

    def summary(self):
        """Network summary."""
        # 方法，用于打印网络的摘要信息。它计算网络中所有可训练参数的数量，并记录到日志中。这对于查看网络的规模和复杂性非常有用
        net_parameters = filter(lambda p: p.requires_grad, self.parameters()) #  获取网络中需要训练的参数
        params = sum([np.prod(p.size()) for p in net_parameters]) # 计算参数的总数
        self.logger.info('Trainable parameters: {}'.format(params)) # 记录参数的总数
        self.logger.info(self) # 记录网络结构
        
# BaseNet 是一个用于构建神经网络模型的基础类。它提供了一个通用的网络结构，包括日志记录和参数数量的计算。子类需要继承这个基类，并根据具体的任务实现自己的前向传播逻辑。这种设计模式允许在不同的网络模型上共享一些通用的功能和属性。