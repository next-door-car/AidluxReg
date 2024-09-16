import argparse
import logging
import cv2
from ad import AD ,ImageProcessor
from base import *
from loggers.logger import init_loggers
from config.config_json import JsonConfig
from config.config_yaml import YamlConfig

def main(args):
    # 配置器
    cfg = YamlConfig(vars(args)) # cfg 会获取到argparse 的参数
    cfg.load_config(args.config_path) # 加载，更新，解析，设置，创建文件夹
    
    processor = ImageProcessor(args.road_picture, args.road_mask)
    CL_picture = processor.process_image()
    origin_picture = cv2.imread(args.road_picture)
    # 初始模型（任务）
    ad = AD(cfg)
    # 创建数据
    ad.create_dataset()
    # 加载支持集
    if args.load_supportset:
        ad.load_supportset()
    # 构建网络
    ad.create_network('ADNet')
    ad.test(device=args.device, n_epochs=args.train_epochs , origin_picture = origin_picture , CL_picture = CL_picture) 


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='深度异常检测任务')
    # 添加参数
    parser.add_argument('--config_path', type=str, default='config/config.yml', help='配置文件路径')

    parser.add_argument('--load_supportset', type=bool, default=False)
    parser.add_argument('--launcher', type=str, choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--label', type=str, default="pcb")
    parser.add_argument('--is_train', type=bool, default=False)
    parser.add_argument('--pre_train', type=bool, default=False)
    parser.add_argument('--train_epochs', type=int, default=1)
    parser.add_argument('--pretrain_epochs', type=int, default=3)
    parser.add_argument('--road_picture', type=str, default="./Start/refer/1.png")
    parser.add_argument('--road_mask', type=str, default="./Mask/1.png")
    args = parser.parse_args()

    main(args)
