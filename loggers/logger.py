import time
import datetime
import logging

from os import path as osp
from base.base_config import BaseConfig

from runner.dist_runner import get_dist_info, master_only

class MessageLogger():
    """Message logger for printing.

    Args:
        cfg.config (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iters.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iter. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default： None.
    """

    def __init__(self, 
                 cfg: BaseConfig, 
                 start_iter=1, tb_logger=None):
        self.exp_name = cfg.get_config(['name'])
        self.interval = cfg.get_config(['logger', 'print_freq'])
        self.start_iter = start_iter
        self.max_iters = cfg.get_config(['trainner', 'train_epochs'])
        self.use_tb_logger = cfg.get_config(['logger', 'use_tb_logger'])
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    @master_only
    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                batch (int): Current batch number.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        # epoch, iter, learning rates
        current_epoch = log_vars.pop('epoch')
        current_batch = log_vars.pop('batch')
        lrs = log_vars.pop('lrs')

        message = (f'[{self.exp_name[:-6]}..][epoch:{current_epoch:5d}, ' # 3d 代表的长度
                   f'batch:{current_batch:3d}, lr:(') # 3d 代表的长度
        for v in lrs:
            message += f'{v:.3e},'
        message += ')] '

        # time and estimated time
        if 'time' in log_vars.keys():
            data_time = log_vars.pop('data_time')
            batch_time = log_vars.pop('batch_time')

            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_batch - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_batch - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, '
            message += f'time (data): {batch_time:.3f} ({data_time:.3f})] '

        # other items, especially losses
        for k, v in log_vars.items():
            message += f'{k}: {v:.4e} ' # 4e 代表的精度
            # tensorboard logger
            if self.use_tb_logger and 'debug' not in self.exp_name:
                if k.startswith('l_'):
                    self.tb_logger.add_scalar(f'losses/{k}', v, current_batch)
                else:
                    self.tb_logger.add_scalar(k, v, current_batch)
        self.logger.info(message)

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def init_loggers(cfg: BaseConfig):
    # logger
    log_file = osp.join(cfg.get_config(['path','log']),
                        f"train_{cfg.get_config(['name'])}_{get_time_str()}.log")
    logger = get_root_logger(logger_name=cfg.get_config(['logger','name']), 
                             log_level=logging.INFO, 
                             log_file=log_file)
    # 打印logge信息
    from config.config_yaml import dict2str
    # logger.info(get_env_info()) # 输出环境信息
    # logger.info(dict2str(cfg.config)) # 输出配置参数

    # initialize tensorboard logger and wandb logger
    # tb_logger
    tb_logger = None
    if cfg.get_config(['logger','use_tb_logger']) and \
       'debug' not in cfg.get_config(['name']):
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', cfg.get_config(['name'])))
    # wandb_logger
    if (cfg.get_config(['logger','wandb']) is not None) and \
        cfg.get_config(['logger','wandb','project']) is not None and \
        'debug' not in cfg.get_config(['name']):
        assert cfg.get_config(['logger','use_tb_logger']) is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(cfg)
    return logger, tb_logger

@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


@master_only
def init_wandb_logger(cfg: BaseConfig):
    """We now only use wandb to sync tensorboard log."""
    # yaml 是个 字典 dict
    import wandb
    logger = logging.getLogger('basicsr')

    project = cfg.get_config(['logger','wandb','project'])
    resume_id = cfg.get_config(['logger','wandb','resume_id'])
    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f'Resume wandb logger with id={wandb_id}.')
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'

    wandb.init(
        id=wandb_id,
        resume=resume,
        name= cfg.get_config(['name']),
        config=cfg.config,
        project=project,
        sync_tensorboard=True)

    logger.info(f'Use wandb logger with id={wandb_id}; project={project}.')

def get_root_logger(logger_name='basicad',
                    log_level=logging.INFO,
                    log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    format = '%(asctime)s %(levelname)s: %(message)s' # %(asctime)s - %(name)s - %(levelname)s - %(message)s
    formatter = logging.Formatter(format)
    logging.basicConfig(format=format, level=log_level) # 配置日志格式
    logger = logging.getLogger(logger_name) # 获取日志器
    # if the logger has been initialized, just return it
    if logger.hasHandlers(): # 查看是否配置了logger
        return logger
    rank, _ = get_dist_info() # 获取进程id
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        # 如果log_file存在，则设置日志文件
        file_handler = logging.FileHandler(log_file, 'w') # 设置日志
        file_handler.setFormatter(formatter) # 设置日志格式
        file_handler.setLevel(log_level) # 作用是设置日志级别
        logger.addHandler(file_handler) # 将日志添加到logger中

    return logger


def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision

    from config.version import __version__
    msg = r"""
                ____                _       _____  ____
               / __ ) ____ _ _____ (_)_____/ ___/ / __ \
              / __  |/ __ `// ___// // ___/\__ \ / /_/ /
             / /_/ // /_/ /(__  )/ // /__ ___/ // _, _/
            /_____/ \__,_//____//_/ \___//____//_/ |_|
     ______                   __   __                 __      __
    / ____/____   ____   ____/ /  / /   __  __ _____ / /__   / /
   / / __ / __ \ / __ \ / __  /  / /   / / / // ___// //_/  / /
  / /_/ // /_/ // /_/ // /_/ /  / /___/ /_/ // /__ / /<    /_/
  \____/ \____/ \____/ \____/  /_____/\____/ \___//_/|_|  (_)
    """
    msg += ('\nVersion Information: '
            f'\n\tBasicAD: {__version__}'
            f'\n\tPyTorch: {torch.__version__}'
            f'\n\tTorchVision: {torchvision.__version__}')
    return msg
