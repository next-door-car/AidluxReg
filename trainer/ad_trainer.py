import os
import time
from base.base_config import BaseConfig
from base.base_dataset import BaseDataset
from base.base_trainer import BaseTrainer
from typing import Any, Type, Union, List, Optional, Callable
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import logging

from tqdm import tqdm
from networks.ad_net import ADNet, ADLoss
from sklearn.metrics import roc_auc_score
from loggers.logger import MessageLogger, get_root_logger
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.cm as cm

from utils.utils_test import *

# 以下两句等价，但后者更简洁
# logger = get_root_logger()
logger = logging.getLogger('basicad')
def show(image):
    plt.imshow(image) # 热力图camp='hot'
    plt.show()

class ADTrainer(BaseTrainer):
    def __init__(self, 
                 cfg: BaseConfig,
                 net: ADNet,
                 dataset: BaseDataset,
                 device: str = 'cuda',
                 n_epochs: int = 150):
        """
        构造函数，用于初始化 AETrainer 对象。
        optimizer_name: 字符串，指定用于训练自动编码器的优化器的名称，默认为 'adam'。
        lr: 学习率，默认为 0.001。
        n_epochs: 训练的总轮次，默认为 150。
        其他参数包括学习率里程碑、批量大小、权重衰减、设备等。
        """
        super().__init__(cfg,
                         net,
                         dataset,
                         device,
                         n_epochs)

        self.train_time: Any
        self.test_auc: Any
        self.test_time: Any
        self.test_scores: Any
        
        if self.cfg.get_config(['model','is_train']):
            self.init_training_setting() 
        else:
            self.init_testing_setting()

    def init_training_setting(self):
        self.load_data()
        self.model_to_device()
        self.setup_lossers()
        self.net.train() # 全设为训练模式
        self.net.encoder.resnet.eval() # 单独设为评估,将self.net.encoder.resnet.named_parameters() 不存在
        # 冻结模型的参数(不可能的)
        for param in self.net.encoder.resnet.parameters():
            param.requires_grad = False
        # 在eval()之后才能使用
        self.setup_optimizers()
        self.setup_schedulers()   

    def init_testing_setting(self):
        self.load_data()
        self.model_to_device()
        self.net.eval()# 训练模式  

    def load_data(self):
        return super().load_data()
    
    def feed_data(self, query_img, noise_img, support_img) -> dict:
        # 汇合数据
        query_img = query_img.to(self.device)
        noise_img = noise_img.to(self.device)
        support_img = support_img.to(self.device)
        # 调用网络并获取输出，这里假设返回的是一个元组
        outputs = self.net(query_img, noise_img, support_img)
        # 使用字典推导式将元组转换为字典
        # 这里我们使用输出的索引作为键
        kwargs = {f'out_{i}': output for i, output in enumerate(outputs)}
        return kwargs # 喂入数据
    
    def model_to_device(self):
        return super().model_to_device()
    
    def setup_lossers(self):
        losser = ADLoss()
        self.lossers.append(losser)

    def setup_optimizers(self):
        optimizer_params = []
        for k, v in self.net.named_parameters():
            print(f'Params({k}) : requires_grad({v.requires_grad})')
            if v.requires_grad: # 如果需要梯度
                optimizer_params.append(v) 
            else:
                print(f'Params {k} will not be optimized.')
                logger.warning(f'Params {k} will not be optimized.')
        optimizer_name = self.cfg.config['trainner']['optimizer'].pop('name') # 弹出去,下面才能传值
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(optimizer_params, # 只更新需要的参数
                                         **self.cfg.config['trainner']['optimizer'])
        else:
            raise NotImplementedError(f'optimizer {optimizer_name} is not supperted yet.')
        self.optimizers.append(optimizer) # 只追加了一个


    def setup_schedulers(self):
        return super().setup_schedulers()
    
    def update_adjust_lr(self, current_iter, warmup_iter=-1):
        return super().update_adjust_lr(current_iter, warmup_iter)
    
    def update_optimize_parameters(self, 
                                   current_iter, 
                                   **kwargs):
        ######################################
        for losser in self.lossers:
            loss = losser(**kwargs)
        ######################################
        loss.backward()
        for optimizer in self.optimizers:
            optimizer.step() # 更新模型参数
            optimizer.zero_grad() # 避免后续梯度累加
        return loss.item()
    
    def save(self, current_epoch, current_batch):
        self.save_network(current_epoch=current_epoch, current_batch=current_batch)
        self.save_training_state(current_epoch=current_epoch, current_batch=current_batch)
        pass
    
    def train(self) -> ADNet:
        """
        一种通用的训练方法，用于训练AD任务。
        """
        msg_logger = MessageLogger(self.cfg, start_iter=1, tb_logger=None)
        # Training
        logger.info(f'Start training from epoch: {self.start_epoch}, batch: {self.start_batch}') # 训练开始 iter 代表迭代次数可以考虑继续上次的训练
        train_time = time.time()
        epoch_time, data_time, batch_time = time.time(), time.time(), time.time()
        for epoch in range(self.start_epoch, self.n_epochs+1):
            batch = self.start_batch # n_batches
            # 更新学习率
            self.update_adjust_lr(epoch) # 放外部,由epoch控制
            loss_epoch = 0.0
            for (query_img, noise_img, support_img, _) in tqdm(self.train_loader):
                data_time = time.time() - data_time # 加载数据的时间（cpu）
                # Zero the network parameter gradients
                # 喂入数据
                kwargs = self.feed_data(query_img, noise_img, support_img) # 训练正常数据
                # 优化参数 = 计算损失 + backward + optimize
                # Update network parameters via backpropagation: forward 
                loss_epoch += self.update_optimize_parameters(epoch, **kwargs)
                batch += 1
                batch_time = time.time() - batch_time # 一个batch的耗时
                # 打印这一批次的耗时
                if batch % self.cfg.get_config(['logger', 'print_freq']) == 0:
                    log_vars = {'epoch': epoch, 'batch': batch}
                    log_vars.update({'lrs': self._get_current_lr()}) 
                    log_vars.update({'data_time': data_time, 'batch_time': batch_time})
                    msg_logger(log_vars)
                # save models and training states
                # if batch % self.cfg.get_config(['logger', 'save_checkpoint_freq']) == 0:
                #     logger.info('Saving networks and training states.')
                #     self.save(current_epoch=epoch, current_batch=batch)
            # log epoch statistics
            epoch_time = time.time() - epoch_time
            # save models and training states
            if epoch % self.cfg.get_config(['logger', 'save_checkpoint_freq']) == 0:
                logger.info('Saving networks and training states.')
                self.save(current_epoch=epoch, current_batch=batch)
            logger.info('\t Epoch {}/{} \t Time: {:.3f} \t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_time, loss_epoch / batch))
        # 总的时间
        self.train_time = time.time() - train_time # 总的时间
        logger.info('Training total time: %.3f' %self.train_time)
        logger.info('Finished pretraining.')
        
        return self.net
        # 缩放图片的函数
    
    
    def test(self, origin_picture , CL_picture):
        def resize_image(image, target_width):
            height, width = image.shape[:2]
            scale = target_width / width
            new_size = (target_width, int(height * scale))
            return cv2.resize(image, new_size)
        
        origin_pictured = cv2.resize(CL_picture, (256, 256))
        with torch.no_grad():
            for query_img, noise_img, support_img, mask, label in tqdm(self.test_loader):
                kwargs = self.feed_data(query_img, noise_img, support_img) # 训练正常数据
                anomaly_map, a_map_list = cal_anomaly_map(kwargs['out_0'], kwargs['out_1'], query_img.shape[-1], amap_mode='a')
                
                # 计算最小值和最大值
                min_val = np.min(anomaly_map)
                max_val = np.max(anomaly_map)
                # 归一化到 [0, 1]
                anomaly_map_normalized = (anomaly_map - min_val) / (max_val - min_val + 1e-8) 
                
                anomaly_map_255 = (anomaly_map_normalized * 255).astype(np.uint8)
                anomaly_map_black = gaussian_filter(anomaly_map_255, sigma=2)
                threshold = 190
                anomaly_map_binary = np.where(anomaly_map_black > threshold, 255, 0).astype(np.uint8)



                # 将二值图像转换为三通道
                anomaly_map_binary_3ch = cv2.cvtColor(anomaly_map_binary, cv2.COLOR_GRAY2BGR)
                # 创建一个与 origin_picture 相同大小的红色掩码
                red_mask = np.zeros_like(origin_pictured)
                red_mask[:, :, 2] = 255  # 设置红色通道为 255
                # 使用 anomaly_map_binary 的黑色区域作为掩码应用红色
                masked_image = np.where(anomaly_map_binary_3ch == [255, 255, 255], red_mask, origin_pictured)

                cv2.imwrite('1.png', origin_picture)        #原图
                cv2.imwrite('2.png', origin_pictured)     # 256*256 原图
                cv2.imwrite('3.png', masked_image)        # 加入掩码后的图像
                plt.imsave('4.png', anomaly_map_255, cmap='jet')  #激活图
                cv2.imwrite('5.png', anomaly_map_binary)    #缺陷分割图
            
        # 获取原图的尺寸
        # height, width = origin_picture.shape[:2]

        # # 计算目标宽度（原图宽度的四分之一）
        # target_width = width // 4

        # # 调整图像的宽度与目标宽度一致
        # def resize_to_match_width(img, width):
        #     aspect_ratio = img.shape[0] / img.shape[1]
        #     new_height = int(width * aspect_ratio)
        #     return cv2.resize(img, (width, new_height), interpolation=cv2.INTER_AREA)

        # # 调整所有图像的宽度
        # origin_pictured_resized = resize_to_match_width(origin_pictured, target_width)
        # masked_image_resized = resize_to_match_width(masked_image, target_width)
        # anomaly_map_255_resized = resize_to_match_width(anomaly_map_255, target_width)
        # anomaly_map_binary_resized = resize_to_match_width(anomaly_map_binary, target_width)

        # # 将所有图像纵向拼接
        # def resize_height_to_match(img, height):
        #     return cv2.resize(img, (img.shape[1], height), interpolation=cv2.INTER_AREA)

        # # 目标高度应该是拼接后图像的总高度
        # target_height = height + origin_pictured_resized.shape[0] + masked_image_resized.shape[0] + anomaly_map_255_resized.shape[0] + anomaly_map_binary_resized.shape[0]

        # # 调整各图像的高度，以便于拼接
        # resized_origin_picture = resize_height_to_match(origin_picture, target_width)
        # resized_origin_pictured = resize_height_to_match(origin_pictured_resized, target_width)
        # resized_masked_image = resize_height_to_match(masked_image_resized, target_width)
        # resized_anomaly_map_255 = resize_height_to_match(anomaly_map_255_resized, target_width)
        # resized_anomaly_map_binary = resize_height_to_match(anomaly_map_binary_resized, target_width)

        # # 拼接图像
        # top_row = np.vstack((resized_origin_pictured, resized_masked_image))
        # bottom_row = np.vstack((resized_anomaly_map_255, resized_anomaly_map_binary))

        # # 合并原图和其他图像
        # result_image = np.vstack((resized_origin_picture, top_row, bottom_row))

        # # 保存结果图像
        # cv2.imwrite('result_image.png', result_image)