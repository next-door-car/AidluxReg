import os
import time
from abc import ABC, abstractmethod
from typing import Any, Type, Union, List, Optional, Callable

import logging

import torch
from torch import nn

from copy import deepcopy

from loggers.logger import get_root_logger
# 引入 Python 的 ABC（Abstract Base Class）类以及 abstractmethod 装饰器。这两个是用于创建抽象基类和抽象方法的标准库。

from utils.misc import master_only
from networks.ad_net import ADNet
from networks.pe_net import PENet
from base.base_config import BaseConfig
from base.base_dataset import BaseDataset
from base.base_network import BaseNetwork
from trainer.losses.msssim import MSSSIM
from torch.nn.parallel import DataParallel, DistributedDataParallel
from trainer.optimal.scheduler import MultiStepRestartLR, CosineAnnealingRestartLR

# 以下两句等价，但后者更简洁
# logger = get_root_logger()
logger = logging.getLogger('basicad')

class BaseTrainer(ABC):
    """Trainer base class."""

    def __init__(self, 
                 cfg: BaseConfig,
                 net: Union[ADNet, PENet, BaseNetwork, nn.Module],
                 dataset: BaseDataset,
                 device: str,
                 n_epochs: int):
        super().__init__()
        # 跟类构造时，只初始参数，不做逻辑处理和调用函数
        self.cfg = cfg
        self.net = net # 单个模型
        self.dataset = dataset
        self.n_epochs = n_epochs # 训练的总轮数（整数）
        # 单卡=>普通放置
        # 多卡
        # # DP=>默认的主gpu
        # # DDP=>自动获取到
        self.device = device
        
        # 训练器状态
        self.start_epoch = 0 
        self.start_batch = 0
        self.lossers = []
        self.optimizers = []
        self.schedulers = []

    def init_training_setting(self):
        self.load_data()
        self.model_to_device()
        self.setup_lossers()
        self.net.train() # 训练模式 
        # 在train()之后才能使用
        self.setup_optimizers()
        self.setup_schedulers()    
        pass

    def init_testing_setting(self):
        self.load_data()
        self.model_to_device()
        self.net.eval() # 训练模式      
        pass
    
    def load_data(self):
        # Get train data loader
        self.train_loader, self.test_loader = self.dataset.loaders()
    
    @abstractmethod # 声明必须实例化
    def feed_data(self, data) -> Any:
        '''只是个举例'''
        data = data.to(self.device)
        out = self.net(data)
        return out
    
    def model_to_device(self):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.
        """
        self.net = self.net.to(self.device) # device 已经确定了
        if self.cfg.get_config(['model','dist']):
            # DDP
            find_unused_parameters = self.cfg.config.get('find_unused_parameters',False) # 存在返回值，不存在返回false
            self.net = DistributedDataParallel(
                     self.net,
                     device_ids=[torch.cuda.current_device()], # device_ids代表当前设备,计算当前设备的数据和网络输入
                     # output_device=0, # output_device代表输出设备（不指定默认在设备的第一个GPU上）
                     find_unused_parameters=find_unused_parameters)
        elif self.cfg.get_config(['model','num_gpu']) > 1:
            # DP
            self.net = DataParallel(self.net)
    
    def setup_lossers(self, type='l1'):
        if type == 'l1':
            losser = nn.L1Loss()
        elif type == 'l2':
            losser = nn.MSELoss()
        elif type  == 'ssim':
            losser = MSSSIM()
        else:
            raise SystemError('【Error】no such type of Losser')
        self.lossers.append(losser)
        pass

    def setup_optimizers(self):
        '''SR_Model 的优化器'''
        optimizer_params = []
        for k, v in self.net.named_parameters():
            if v.requires_grad: # 如果需要梯度
                optimizer_params.append(v) 
            else:
                logger.warning(f'Params {k} will not be optimized.')
        optimizer_name = self.cfg.config['trainner']['optimizer'].pop('name') # 弹出去,下面才能传值
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(optimizer_params, # 只更新需要的参数
                                         **self.cfg.config['trainner']['optimizer'])
        else:
            raise NotImplementedError(f'optimizer {optimizer_name} is not supperted yet.')
        self.optimizers.append(optimizer) # 只追加了一个
        pass

    def setup_schedulers(self):
        """Set up schedulers."""
        scheduler_name = self.cfg.config['trainner']['scheduler'].pop('name')
        if scheduler_name in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    MultiStepRestartLR(
                        optimizer, **self.cfg.config['trainner']['scheduler']))
        elif scheduler_name == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    CosineAnnealingRestartLR(
                        optimizer, **self.cfg.config['trainner']['scheduler']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_name} is not implemented yet.')
        pass
    
    def update_adjust_lr(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step() # 采用调度器
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)
        pass
    
    @abstractmethod
    def update_optimize_parameters(self, current_iter, targets, outputs) -> Any:
        '''这是个举例'''
        return targets-outputs
    
    @abstractmethod
    def save(self):
        pass
    
    def get_bare_network(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net
    
    @master_only
    def print_network(self):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(self.net, (DataParallel, DistributedDataParallel)):
            net_cls_str = (f'{self.net.__class__.__name__} - '
                           f'{self.net.module.__class__.__name__}')
        else:
            net_cls_str = f'{self.net.__class__.__name__}'

        net = self.get_bare_network(self.net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger.info(
            f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        logger.info(net_str)
    
    def print_network_loading_different_keys(self, load_net, strict=True):
        """Print keys with differnet name or different size when loading models.

        1. Print keys with differnet names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_network(self.net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(
                        f'Size different, ignore [{k}]: crt_net: '
                        f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)
    
    def load_network(self, load_path, strict=False, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        self.net = self.get_bare_network(self.net) # 获取网络的基本网络结构
        logger.info(f'Loading {self.net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(
            load_path, 
            map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self.print_network_loading_different_keys(load_net, strict)
        self.net.load_state_dict(load_net, strict=strict)
        
    @master_only
    def save_network(self, current_epoch, current_batch, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_batch == -1:
            current_batch = 'latest'
        net_name = self.cfg.get_config(['network', 'name'])
        save_name = f'{net_name}_{current_epoch}_{current_batch}.pth'
        save_path = os.path.join(self.cfg.get_config(['path', 'networks']), save_name)
    
        net_list = self.net if isinstance(self.net, list) else [self.net] # 转为列表
        param_key_list = param_key if isinstance(param_key, list) else [param_key] # 转为列表
        assert len(net_list) == len(param_key_list), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net_list, param_key_list):
            net_ = self.get_bare_network(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        torch.save(save_dict, save_path)
        pass
        
    @master_only
    def save_training_state(self, current_epoch, current_batch):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_batch (int): Current batch.
        """
        if current_batch != -1:
            state = {
                'epoch': current_epoch,
                'batch': current_batch,
                'lossers': [],
                'optimizers': [],
                'schedulers': []
            }
            for l in self.lossers:
                state['lossers'].append(l.state_dict())
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            net_name = self.cfg.get_config(['network', 'name'])
            save_name = f'{net_name}_{current_epoch}_{current_batch}.state'
            save_path = os.path.join(self.cfg.get_config(['path', 'training_states']),
                                     save_name)
            torch.save(state, save_path)
        pass
    
    def check_resume(self, resume_path: str) -> str:
        """Check resume_state and pretrain_load paths.
        """
        resume_network = os.path.splitext(os.path.basename(resume_path))[0] # 无扩展名
        resume_network_name = resume_network + '.pth'
        resume_network_path = os.path.join(
            os.path.dirname(os.path.dirname(resume_path)), # 父级
            'networks',
            resume_network_name
        )
        # 检查文件是否存在
        if os.path.exists(resume_network_path):
            logger.info(f"Set resume_network_path: {resume_network_path}")
        else:
            raise NotImplementedError('resume_network_path({resume_network_path}) not exists.')
        return resume_network_path
                    
    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        self.start_epoch = resume_state['epoch']
        self.start_batch = resume_state['batch']
        resume_lossers   = resume_state['lossers']
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_lossers)    == len(self.lossers), 'Wrong lengths of lossers'
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, l in enumerate(resume_lossers):
            self.lossers[i].load_state_dict(l) 
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o) 
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
        pass
            
    @abstractmethod # 声明必须实例化
    def train(self) -> BaseNetwork:
      	# 使用 @abstractmethod 装饰器来声明抽象方法。这些方法是在子类中必须被实现的，否则子类也会被标记为抽象类
        """
        Implement train method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        # 抽象方法 train，用于在给定的数据集和神经网络上训练模型。它接受一个 BaseADDataset 类型的数据集和一个 BaseNet 类型的神经网络模型作为输入，并返回训练后的网络模型。
        return self.net

    @abstractmethod
    def test(self, dataset: BaseDataset, net: BaseNetwork):
        """
        Implement test method that evaluates the test_set of dataset on the given network.
        """
        # 抽象方法 test，用于在给定的数据集和神经网络上进行测试或评估。它接受一个 BaseADDataset 类型的数据集和一个 BaseNet 类型的神经网络模型作为输入，通常用于计算测试集上的性能指标。
        pass
    
    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append(
                [
                    v['initial_lr'] for v in optimizer.param_groups
                ]
            )
        return init_lr_groups_l
    def _get_current_lr(self):
        return [
            param_group['lr'] for param_group in self.optimizers[0].param_groups
        ]
    