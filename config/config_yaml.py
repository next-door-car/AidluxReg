import yaml
import click
import torch
import random
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional
from collections import OrderedDict
from os import path as osp

from base.base_config import BaseConfig
from utils.misc import mkdir_and_rename, make_exp_dirs
from runner.dist_runner import init_dist, get_dist_info

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dict2str(dic: dict, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in dic.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

class YamlConfig(BaseConfig):
    def __init__(self, settings):
        super().__init__(settings)
    def load_config(self, import_config):
        # click.echo("Loading configuration from %s." %import_config)
        with open(import_config, mode='r') as f:
            Loader, _ = ordered_yaml()
            self.config = yaml.load(f, Loader=Loader) # yaml 是个 dict
        # click.echo("Loading configuration %s succucess." %import_config)
        # 更新 config
        self._update_yaml(self.config, self.settings) # 更新 yaml
        self._update_settings(self.config, self.settings) # 更新 settings
        # 解析PATH
        self.parse_config(self.config) # 解析配置
        # 设置必要参数
        self.setup_config(self.config) # 设置配置
        # 创建文件夹
        self.mkdir_config(self.config) # 创建文件夹
        
    def save_config(self, export_config):
        with open(export_config, mode='w') as f:
            yaml.dump(self.settings, f)
    # 设置配置
    def set_config(self, keys, value):
        quer = self.config # 从最外层开始查询
        for key in keys:
            if key in quer:
                # 在keys中有key，则进入下一层
                if isinstance(key, dict): 
                    quer = quer[key]
                    continue
                else:
                    quer[key] = value # 设置值
            else:
                # 如果不在
                quer[key] = value
                
    # 读取配置参数
    def get_config(self, keys) -> Any:
        # key接受列表参数
        quer = self.config # 从最外层开始查询
        for key in keys:
            if key in quer:
                quer = quer[key]
            else:
                raise KeyError("Key %s not in config." %key)
        if quer is None:
            # 没值
            value = None
            # print("config key {} not found".format(keys))
            # raise ValueError('config keys {} not found'.format(keys))
        else:
            # 有值，但是可能是字典形式
            value = quer
        # click.echo("config %s = %s." %(keys,value))
        return value
    
    def parse_config(self, config):
        # paths 展开为绝对路径
        if config['path'] is None:
            config['path'] = {} # 创建个空字典
        for key, val in config['path'].items():
            if (val is not None) and ('resume_state' in key
                                  or  'strict_load' in key
                                  or  'pretrain_load' in key):
                # 'path' 下 有值
                config['path'][key] = osp.expanduser(val) # 展开为绝对路径
        config['path']['root'] = osp.abspath(
            osp.join(__file__, osp.pardir, osp.pardir)) # 当前文件父目录的父目录的绝对路径  
        if config['model']['is_train']:
            experiments_root = osp.join(config['path']['root'], 'experiments', config['name']) # 实验的路径
            config['path']['experiments_root'] = experiments_root
            config['path']['models'] = osp.join(experiments_root, 'models') # 最终模型.pt
            config['path']['networks'] = osp.join(experiments_root, 'networks') # 中间网络.pt
            config['path']['pre_networks'] = osp.join(experiments_root, 'pre_networks')
            config['path']['training_states'] = osp.join(experiments_root, 'training_states')
            config['path']['log'] = experiments_root
            config['path']['visualization'] = osp.join(experiments_root, 'visualization')
            # change some options for debug mode
            if 'debug' in config['name']:
                if 'val' in config:
                    config['val']['val_freq'] = 10
                config['logger']['print_freq'] = 10
                config['logger']['save_checkpoint_freq'] = 15
        # else:  # test
        #     results_root = osp.join(config['path']['root'], 'results', config['name'])
        #     config['path']['results_root'] = results_root
        #     config['path']['log'] = results_root
        #     config['path']['visualization'] = osp.join(results_root, 'visualization')

    def setup_config(self, config):
        # 确定device
        if not torch.cuda.is_available():
            config['model']['device'] = 'cpu'
        elif config['model']['num_gpu'] > 1:
            # distributed settings
            # 有cuda才用，没有cuda就不用 同时 gpu 要大于1 分布式
            if config['model']['dist'] is False:
                # DP模式=>单进程控单卡=>模型加载到默认gpu（主）=>数据加载到默认gpu（主）即可
                print('Disable distributed.', flush=True)
                config['model']['device'] = config['model']['device'] # 主gpu 
            else:
                # DDP模式=>多进程控多卡=>数据和模型要对应放到gpu下
                if config['dist_params']['launcher'] == None:
                    raise ValueError('Distributed launcher is not specified.')
                elif config['dist_params']['launcher'] == 'pytorch':
                    # pytorch
                    init_dist(config['dist_params']['launcher'])
                elif config['dist_params']['launcher'] == 'slurm' and 'dist_params' in config:
                    # slurm
                    init_dist(launcher=config['dist_params']['launcher'], 
                            backend=config['dist_params']['backend'], # nccl
                            **config['dist_params']) # 双星号代表解包
                config['model']['device'] = torch.cuda.current_device() # DDP模式指定了设备和进程的对应关系，所以可以获取活跃的device
        else:
            # 默认参数
            config['model']['device'] = config['model']['device'] 
            
        rank, world_size= get_dist_info() # dist 检查 rank
        config['dist_params']['rank'] = rank
        config['dist_params']['world_size'] = world_size
        
        # random seed
        seed = config['seed']
        if seed is None: # 即无配置
            seed = random.randint(1, 10000)
            config['seed'] = seed
        set_random_seed(seed + config['dist_params']['rank'])

    def mkdir_config(self, config):
        # load resume states if necessary
        if config['path'].get('resume_state') is None:
            # 表明第一次训练
            # mkdir for experiments and logger
            make_exp_dirs(config)
            if config['dist_params']['rank'] == 0 and \
               config['logger'].get('use_tb_logger') and \
               'debug' not in config['name']: 
                mkdir_and_rename(osp.join('tb_logger', config['name']))

    def _update_yaml(self, config, settings):
        # 只更新不同级别下相同key的配置参数
        # yaml = {} # 字典可以使用右边的值覆盖左边的值（yolo）
        # 首先更新yaml => 覆盖
        for key, value in config.items():
            if isinstance(value, dict):
                self._update_yaml(value, settings) # 递归暂时不考虑其余作用
            elif key in settings:
                config[key] = settings[key] # 覆盖

    def _update_settings(self, config, settings):
        # 然后更新setting => 增添
        for key, value in settings.items():
            if key not in config:
                config[key] = settings[key] # 添加
                

