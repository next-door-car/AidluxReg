import click # 用于解析命令行参数的库。
import argparse # 未用
import logging

from ad import AD
from base import *
from loggers.logger import init_loggers
from config.config_json import JsonConfig
from config.config_yaml import YamlConfig

# 定义一个命令行接口的装饰器，会传递给main函数。
# 使用：https://blog.csdn.net/xixihahalelehehe/article/details/106124675
@click.command() # 装饰器，用于创建一个命令行接口的命令。它创建了一个名为main的命令，该命令将用于执行深度异常检测任务
# 固定参数
@click.argument('config_path', type=click.Path(exists=True), default='config/config.yml') 
# 以下参数yml都有
@click.argument('load_supportset', type=bool, default=False) # 是否加载支持集，用于测试
# 可选参数
@click.option('--launcher', type=click.Choice(['none', 'pytorch', 'slurm']), default='none',
              help='job launcher') # 是否采样分布式训练
# 选项参数 
@click.option('--device', type=str, default='cuda', 
              help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--label', type=str, default="pcb3")
@click.option('--is_train', type=bool, default=False)
@click.option('--pre_train', type=bool, default=False)
@click.option('--train_epochs', type=int, default=60)
@click.option('--pretrain_epochs', type=int, default=3)
def main(device,
         label,
         is_train,
         pre_train,
         launcher,
         config_path,
         load_supportset, # 加载支持集
         train_epochs, pretrain_epochs):
    # 配置器
    cfg = YamlConfig(locals().copy()) # cfg 会获取到click 的参数
    cfg.load_config(config_path) # 加载，更新，解析，设置，创建文件夹
    
    # 日志器
    logger, tb_logger = init_loggers(cfg) 
    
    # 初始模型（任务）
    ad = AD(cfg)
    # 创建数据
    ad.create_dataset()
    # 加载支持集
    if load_supportset:
        ad.load_supportset()
    # 构建网络
    ad.create_network('ADNet')
    if is_train:
        # 是否预训练
        if pre_train:
            logger.info('Pretraining...')
            ad.pretrain(device=device, n_epochs=pretrain_epochs)
        # 模型训练
        ad.train(device=device, n_epochs=train_epochs)
        # 模型保存
        ad.save_model(save_pe=False) 
    else:
        # 模型测试
        ad.test(device=device, n_epochs=train_epochs)
    # 结果可视化
    # 结果保存  

if __name__ == '__main__':
    main()