# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py  # noqa: E501
import functools
import os
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp # mp是多进程模块

'''
DDP
'''
os.environ['MASTER_ADDR'] = 'localhost'         
os.environ['MASTER_PORT'] = '8888'  
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn') # 多进程启动方法,默认spawn,暂时不考虑
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')

def _init_dist_pytorch(backend, **kwargs):
    rank = int(os.environ['RANK']) # 自动获取
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus) # 确定每个进程的设备索引,进程号=>第几个gpu
    # torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu') 才可以正确的放到对应的gpu上
    dist.init_process_group(backend=backend, **kwargs) # 采用torch.dist.launch启动 自动识别到rank和word_size

def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

def get_dist_info():
    # rank 是代表当前进程在当前节点上的编号，rank=0代表第一个进程，rank=1代表第二个进程，以此类推。
    # world_size 是代表当前进程所在的节点上总的进程数，world_size=2代表当前节点上启动了两个进程。
    if dist.is_available():
        initialized = dist.is_initialized() # 被初始化过
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def master_only(func):
    # 含义是：只有主进程才能执行该函数
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            # rank 为0 代表主进程
            return func(*args, **kwargs)

    return wrapper
