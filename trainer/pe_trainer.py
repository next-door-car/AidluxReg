import os
from typing import Any
from tqdm import tqdm
from trainer.losses.loss import AverageMeter
from base.base_config import BaseConfig
from base.base_trainer import BaseTrainer
from base.base_dataset import BaseDataset
from networks.pe_net import PENet
from utils.misc import master_only
from sklearn.metrics import roc_auc_score
from loggers.logger import get_root_logger

import logging
import time
import math
import torch
import torch
import numpy as np

"""
BaseTrainer：用于训练神经网络的基础训练器。
BaseADDataset：自定义数据集的基类。
BaseNet：自定义神经网络模型的基类。
roc_auc_score：用于计算AUC-ROC（接收者操作特征曲线下面积）的函数。
其他Python标准库和PyTorch库。
"""

class PETrainer(BaseTrainer):

    def __init__(self, 
                 cfg: BaseConfig,
                 net: PENet,
                 dataset: BaseDataset,
                 device: str = 'cuda',
                 n_epochs: int = 150,):
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
        
        self.pretrain_time: Any
        self.test_auc: Any
        self.test_time: Any
        self.test_scores: Any
        
        self.init_training_setting() 
    def init_training_setting(self):
        self.load_data()
        self.model_to_device()
        self.setup_lossers()
        self.setup_optimizers()
        self.setup_schedulers()    
        self.net.train() # 训练模式  
    def load_data(self):
        return super().load_data()
    def feed_data(self, query_img, support_img_list) -> Any:
        query_img = query_img.to(self.device)
        support_img_list = support_img_list.to(self.device)
        z1, z2, p1, p2 = self.net(query_img, support_img_list)
        return z1, z2, p1, p2
    
    def model_to_device(self):
        return super().model_to_device()
    
    def setup_lossers(self):
        self.losser = PELoss(mean=True)
        self.lossers.append(self.losser)
        
    def setup_optimizers(self):
        # Set optimizer (Adam optimizer for now)
        encoder_optimizer = torch.optim.SGD(self.net.encoder.parameters(), 
                                            lr=self.cfg.config['trainner']['optimizer']['lr'],
                                            momentum=0.9)
        pwcnn_optimizer = torch.optim.SGD(self.net.pwcnn.parameters(), 
                                          lr=self.cfg.config['trainner']['optimizer']['lr'],
                                          momentum=0.9)
        proj_optimizer = torch.optim.SGD(self.net.proj.parameters(), 
                                         lr=self.cfg.config['trainner']['optimizer']['lr'],
                                         momentum=0.9)
        pred_optimizer = torch.optim.SGD(self.net.pred.parameters(), 
                                         lr=self.cfg.config['trainner']['optimizer']['lr'],
                                         momentum=0.9)
        self.init_lrs = [self.cfg.config['trainner']['optimizer']['lr'], 
                         self.cfg.config['trainner']['optimizer']['lr'], 
                         self.cfg.config['trainner']['optimizer']['lr'], 
                         self.cfg.config['trainner']['optimizer']['lr']]
        self.optimizers = [encoder_optimizer, pwcnn_optimizer, proj_optimizer, pred_optimizer]
    def setup_schedulers(self):
        '''无'''
        pass
    
    def update_adjust_lr(self, optimizers, init_lrs, epoch, epochs):
        """Decay the learning rate based on schedule"""
        for i in range(3):
           cur_lr = init_lrs[i] * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
           for param_group in optimizers[i].param_groups:
              param_group['lr'] = cur_lr
    def update_optimize_parameters(self, current_iter, targets, outputs):
        '''暂无'''
        pass 
    
    def save(self):
        self.save_network()
        pass 

    @master_only
    def save_network(self, param_key='params'):

        net_name = self.cfg.get_config(['network', 'name'])
        save_name = f'{net_name}_pre.pth'
        save_path = os.path.join(self.cfg.get_config(['path', 'pre_networks']), save_name)
    
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
    
    def train(self) -> PENet:
           
        logger = get_root_logger()
        
        # Set learning rate scheduler
        # encoder_scheduler = optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=self.lr_milestones, gamma=0.1)
        # proj_optimizer = optim.lr_scheduler.MultiStepLR(proj_optimizer, milestones=self.lr_milestones, gamma=0.1)
        # pred_optimizer = optim.lr_scheduler.MultiStepLR(pred_optimizer, milestones=self.lr_milestones, gamma=0.1)
        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        for epoch in range(self.n_epochs+1):
            if epoch in self.cfg.get_config(['trainner','scheduler','milestones']):
                # logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
                logger.info(' updata learning rate... ')
            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            total_losses = AverageMeter() # 初始化一个计算平均值的类
            for (query_img, noise_img, support_img, _) in tqdm(self.train_loader):
                # Zero the network parameter gradients
                # 此处不能多步数
                self.optimizers[0].zero_grad()
                self.optimizers[1].zero_grad()
                self.optimizers[2].zero_grad()
                self.optimizers[3].zero_grad()
                self.update_adjust_lr(self.optimizers, self.init_lrs, epoch, self.n_epochs)
                # 数据
                z1, z2, p1, p2 = self.feed_data(query_img, support_img) # 训练正常的
                # 计算相似度损失 
                loss = self.losser(p1,z2)/2 + self.losser(p2,z1)/2 # siames loss
                total_losses.update(loss.item(), noise_img.size(1)) # 求其平均值
                loss.backward()
                
                # 分别更新模型参数
                self.optimizers[0].step()
                self.optimizers[1].step()
                self.optimizers[2].step()
                self.optimizers[3].step()
                
                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}\t Total_Loss: {:.6f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches, total_losses.avg))

        self.pretrain_time = time.time() - start_time
        logger.info('Pretraining time: %.3f' %self.pretrain_time)
        logger.info('Finished pretraining.')
        self.save() # 直接保存

        return self.net

    def test(self, dataset: BaseDataset, pe_net: PENet):
        """
        方法，用于在测试数据上测试自动编码器。
        dataset: 数据集对象，继承自 BaseADDataset。
        ae_net: 自动编码器模型。
        用于在测试数据上测试自动编码器。
        将自动编码器模型 ae_net 移动到指定的计算设备。
        获取测试数据的数据加载器。
        开始测试循环，迭代测试数据。
        初始化损失、批次计数和存储 (idx, label, score) 三元组的列表。
        将自动编码器设置为评估模式。
        使用无梯度计算上下文 torch.no_grad()，以减少内存占用和加速测试过程。
        遍历测试数据加载器中的每个批次，将批次数据移动到指定的设备。
        通过前向传播计算输出，并计算重建误差。
        累积损失、批次计数和 (idx, label, score) 三元组。
        计算并记录测试集上的平均损失和AUC-ROC。
        记录测试时间并输出结果。
        """
        logger = logging.getLogger()

        # Set device for network
        pe_net = pe_net.to(self.device)

        # Testing
        logger.info('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        pe_net.eval()
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = pe_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                n_batches += 1

        logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_scores = scores
        self.test_auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

        self.test_time = time.time() - start_time
        logger.info('Autoencoder testing time: %.3f' %self.test_time)
        logger.info('Finished testing autoencoder.')
    