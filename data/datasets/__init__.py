import logging
import importlib
from os import path as osp
from base.base_dataset import BaseDataset
from runner.dist_runner import get_dist_info
from utils.misc import get_root_logger, scandir

# 以下两句等价
# logger = get_root_logger()
logger = logging.getLogger('basicad')

# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) # 获取data文件夹下的所有文件名，并去掉文件扩展名。
    # osp.basename(v) 获取路径v的最后一部分，即文件名。
    # osp.splitext() 将文件名分割为两部分：文件名本身和它的扩展名。
    # [0]表示我们只取分割结果的第一部分，即文件名（不包括扩展名）。
    if v.endswith('_dataset.py') # 以_dataset.py结尾的文件
]
# import all the _dataset.py modules
_dataset_modules = [
    importlib.import_module(f'data.datasets.{file_name}')
    for file_name in dataset_filenames
]

def create_dataset(dataset_config: dict, **kwargs) -> BaseDataset:
    """Create dataset.

    Args:
        dataset_opt (dict): Configuration for dataset. It constains:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_name = dataset_config['dataset_name']
    dataset_type = dataset_config['dataset_type']

    # dynamic instantiation
    if dataset_name in dataset_filenames:
        dataset_module= importlib.import_module(f'data.datasets.{dataset_name}')
        dataset_class = getattr(dataset_module, dataset_type, None) # 类的名字
        if dataset_class is None:
            raise ValueError(f'Dataset {dataset_type} is not found.')
    else:
        raise ValueError(f'Dataset {dataset_name} is not found.')

    dataset = dataset_class(**kwargs) # 传递关键字参数

    logger.info(
        f'Dataset {dataset.__class__.__name__} - {dataset_config["dataset_name"]} '
        'is created.')
    return dataset



