import os
import logging
from os import path as osp

from runner.dist_runner import master_only
from loggers.logger import get_root_logger, get_time_str

# 以下两句等价，但后者更简洁
# logger = get_root_logger()
logger = logging.getLogger('basicad')

def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


@master_only
def make_exp_dirs(config: dict):
    """Make dirs for experiments."""
    path_config = config['path'].copy()
    if config['model']['is_train']:
        # 训练根目录
        mkdir_and_rename(path_config.pop('experiments_root'))
   
    
    for key, path in path_config.items():
        # 'resume_state' 代表是否加载预训练模型
        # 'strict_load' 代表是否恢复训练
        # 'pretrain_load' 代表预训练模型路径
        if (key != 'resume_state') and \
           (key != 'strict_load') and \
           (key != 'pretrain_load'):
            # 都没有定义则自动创建path下的路径，为当前新的实验
            os.makedirs(path, exist_ok=True) 

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)



def sizeof_fmt(size, suffix='B'):
    """Get human readable file size.

    Args:
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
        str: Formated file siz.
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f'{size:3.1f} {unit}{suffix}'
        size /= 1024.0
    return f'{size:3.1f} Y{suffix}'
