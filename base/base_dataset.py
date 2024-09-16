import os
import time
from abc import ABC, abstractmethod
from typing import Any, Type, Union, List, Optional, Callable

from base.base_config import BaseConfig

import random
import numpy as np

from torch.utils.data import DataLoader
from runner.dist_runner import get_dist_info

def worker_init_fn(worker_id, num_workers, seed):
    '''
    rank: 进程id（手动内部获取）
    worker_id: 线程id（自动传入）
    num_workers: 线程数量
    seed: 随机种子
    '''
    rank, _ = get_dist_info()
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = rank * num_workers + worker_id + seed
    np.random.seed(worker_seed) # 所有使用NumPy的随机操作（例如数据洗牌）都会基于这个种子产生随机数，从而确保随机性是可控的。
    random.seed(worker_seed) # 使用random模块的随机操作也具有相同的可控随机性。


class BaseDataset(ABC):
    """Anomaly detection dataset base class."""
    '''
    root: 数据的根路径。
    n_classes: 数据集中的类别数，通常为2，分别表示正常和异常。
    normal_classes: 定义正常类别的原始类标签的元组。
    outlier_classes: 定义异常类别的原始类标签的元组。
    train_set: 训练集，必须是torch.utils.data.Dataset类型。
    test_set: 测试集，必须是torch.utils.data.Dataset类型。
    '''
    def __init__(self, 
                 cfg: BaseConfig,
                 root: str,
                 seed: int,
                 # 数据集
                 train_set: Any,
                 test_set: Any,
                 # 分布式
                 dist: bool,
                 num_gpu: int,
                 rank: int,
                 world_size: int,
                 # 数据集加载
                 batch_size_per_gpu: int,
                 num_worker_per_cpu: int,
                 dataset_enlarge_ratio: int):
        super().__init__()
        self.cfg = cfg
        # 核心部分
        self.root = root  # root path to data
        self.seed = seed 
        self.train_set = train_set  # must be of type torch.utils.data.Dataset
        self.test_set = test_set  # must be of type torch.utils.data.Dataset
        # 分布式
        self.dist = dist
        self.num_gpu = num_gpu
        self.rank = rank
        self.world_size = world_size
        
        self.batch_size_per_gpu = batch_size_per_gpu
        self.num_worker_per_cpu = num_worker_per_cpu
        self.dataset_enlarge_ratio = dataset_enlarge_ratio
        
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # tuple with original class labels that define the normal class
        self.outlier_classes = None  # tuple with original class labels that define the outlier class

    def samplers(self) -> Any:
        """
        仅仅支持分布式
        """
        pass
    
    @abstractmethod
    def loaders(self, 
                shuffle_train=True, 
                shuffle_test=False) -> tuple: # (DataLoader, DataLoader)
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        '''
        一个抽象方法，子类必须实现该方法以返回用于训练集和测试集的torch.utils.data.DataLoader类型的数据加载器。
        '''
        pass

    def __repr__(self):
        # 这个基类提供了一个框架，可以在其基础上构建特定异常检测数据集的子类，以适应不同的数据集和需求。
        return self.__class__.__name__
