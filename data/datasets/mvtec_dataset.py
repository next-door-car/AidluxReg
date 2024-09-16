import os
from typing import Any
from PIL import Image
import random
import math
import numpy as np
import torch
import torch.nn.functional as F

from functools import partial
from data.dataprocess.noise.noise import Simplex_CLASS
from base.base_config import BaseConfig
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from base.base_dataset import BaseDataset, worker_init_fn
from data.datasets.samplers.enlarged import EnlargedSampler


CLASS_NAMES = [
    'pcb'
]

class FSADDataset(BaseDataset):
    def __init__(self,
                 cfg: BaseConfig,
                 root: str,
                 seed: int,
                 # 数据集=>训练集和测试集的划分方式
                 size: int,
                 label: str,
                 shot: int = 2,
                 batch: int = 1, # 手动的 默认为1
                 # 分布式
                 dist: bool = False,
                 num_gpu: int = 1,
                 rank: int = 0,
                 world_size: int = 1,
                 # 数据集加载
                 batch_size_per_gpu: int = 1, # loader
                 num_worker_per_cpu: int = 2, # 线程数量
                 dataset_enlarge_ratio = 1):
        # set suport transforms
        self.x_resize = size
        self.transform_x = transforms.Compose([
            transforms.Resize((size,size), Image.LANCZOS), # Image.ANTIALIAS),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # set quer transforms
        self.y_resize = size
        self.transform_y = transforms.Compose([
            transforms.Resize((size,size), Image.LANCZOS), # Image.ANTIALIAS),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_resize = self.x_resize
        self.transform_mask = transforms.Compose([
            transforms.Resize((size,size), Image.NEAREST),
            transforms.ToTensor()
        ])
        self.train_set = TrainDataset(root=root, label=label, phase= "train",
                                      x_resize = self.x_resize,
                                      y_resize = self.y_resize,
                                      mask_resize = self.mask_resize, 
                                      transform_x = self.transform_x,
                                      transform_y = self.transform_y,
                                      shot=shot,
                                      batch=batch)
        self.test_set = TestDatasets(root=root, label=label, phase= "test", # 此处暂时为train
                                     x_resize = self.x_resize,
                                     y_resize = self.y_resize,
                                     mask_resize = self.mask_resize, 
                                     transform_x = self.transform_x,
                                     transform_y = self.transform_y,
                                     transform_mask= self.transform_mask,
                                     shot=shot) # batch=1 默认
        super().__init__(cfg,
                         root=root,
                         seed=seed,
                         # 数据集
                         train_set=self.train_set,
                         test_set=self.test_set,
                         # 分布式
                         dist=dist,
                         num_gpu = num_gpu,
                         rank = rank,
                         world_size = world_size,
                         # 数据集加载
                         batch_size_per_gpu=batch_size_per_gpu,
                         num_worker_per_cpu=num_worker_per_cpu,
                         dataset_enlarge_ratio = dataset_enlarge_ratio)
        
    def samplers(self) -> Any:
        '''只训练集'''
        train_sampler = EnlargedSampler(self.train_set, 
                                        self.rank, 
                                        self.world_size,
                                        self.dataset_enlarge_ratio)
        return train_sampler
    
    def loaders(self, 
                shuffle_train = False, # 外部做了
                shuffle_test = False) -> (DataLoader, DataLoader): # 外部做了
        '''
        覆盖抽象类
        '''
        if self.dist:
            # DDP模式：samplers采样
            batch_size_per_gpu = self.batch_size_per_gpu
            num_worker_per_cpu = self.num_worker_per_cpu
            train_sampler = self.samplers()
        else:
            # DP模式：简单易用
            multiplier = 1 if self.num_gpu == 0 else self.num_gpu # 此处 num_gup = 1
            # 多个gpu时，每个gpu的batch_size = batch_size_per_gpu * num_gpu
            batch_size_per_gpu = self.batch_size_per_gpu * multiplier
            num_worker_per_cpu = self.num_worker_per_cpu * multiplier
            train_sampler = None
        # 加载训练集
        train_dataloader_args = dict(
            dataset=self.train_set,
            shuffle=False,
            sampler=train_sampler,
            batch_size=batch_size_per_gpu,
            num_workers=num_worker_per_cpu,
            drop_last=True,
            worker_init_fn=partial(
                worker_init_fn, 
                num_workers=num_worker_per_cpu, 
                seed=self.seed) # 创建一个新的部分函数（partial function）将固定原始函数的一些参数，使得在调用时只需要传递剩余的参数。
                if self.seed is not None else None
            )
        if train_dataloader_args['sampler'] is None:
            train_dataloader_args['shuffle'] = shuffle_train # 不是分布式则由外部决定
        train_loader = DataLoader(**train_dataloader_args)
        # 加载测试集
        test_dataloader_args = dict(
            dataset=self.test_set,
            shuffle=shuffle_test,
            batch_size=1,
            num_workers=0) # 0表示不使用多线程
        test_loader = DataLoader(**test_dataloader_args)
        return train_loader, test_loader
        
class TrainDataset(Dataset):
    def __init__(self,
                 root: str,
                 label: str,
                 phase: str, # 训练train，而不是test
                 x_resize: int,
                 y_resize: int,
                 mask_resize: int,
                 transform_x: transforms.Compose,
                 transform_y: transforms.Compose,
                 shot=2,
                 batch=32):
        assert label in CLASS_NAMES, 'class_name: {}, should be in {}'.format(label, CLASS_NAMES)
        self.root = root
        self.label = label
        self.phase = phase
        self.x_resize = x_resize
        self.y_resize = y_resize
        self.mask_resize = mask_resize
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.shot = shot
        self.batch = batch
        self.simplexNoise = Simplex_CLASS()
        # load dataset
        self.query_dir, self.support_dir = self.load_dataset_folder()

    def __getitem__(self, idx):
        query_list, support_list = self.query_dir[idx], self.support_dir[idx]
        query_img = None
        noise_img = None
        support_sub_img = None
        support_img = None

        for i in range(len(query_list)):
            # 查询集（1）
            image = Image.open(query_list[i]).convert('RGB')
            image = self.transform_y(image) #image_shape torch.Size([3, 224, 224])
            noise_image = add_noise_to_image(image, self.simplexNoise, self.y_resize)
            image = image.unsqueeze(dim=0) #image_shape torch.Size([1, 3, 224, 224])
            noise_image = noise_image.unsqueeze(dim=0)
            if query_img is None:
                query_img = image
            else:
                query_img = torch.cat([query_img, image],dim=0)
            if noise_img is None:
                noise_img = noise_image
            else:
                noise_img = torch.cat([noise_img, noise_image],dim=0)
            # 支持集
            for k in range(self.shot):
                image = Image.open(support_list[i][k]).convert('RGB') # type: PIL.Image.Image
                image = self.transform_x(image)
                image = choose_random_aug_image(image) # 支持集增强处理
                image = image.unsqueeze(dim=0) # image_shape torch.Size([k, 3, 224, 224])
                if support_sub_img is None:
                    support_sub_img = image
                else:
                    support_sub_img = torch.cat([support_sub_img, image], dim=0) # support_sub_img_shape torch.Size([2, 3, 224, 224])
            # 按照类别
            support_sub_img = support_sub_img.unsqueeze(dim=0)
            if support_img is None:
                support_img = support_sub_img
            else:
                support_img = torch.cat([support_img, support_sub_img], dim=0) # 按照类别
            support_sub_img = None
        mask = torch.zeros([self.batch, self.mask_resize, self.mask_resize])
        return query_img, noise_img, support_img, mask

    def __len__(self):
        return len(self.query_dir)
    
    def load_dataset_folder(self):
        data_img = {}
        for class_name_one in CLASS_NAMES:
            # if class_name_one != self.label: # 不是目标类别
            if class_name_one == self.label: # 加载目标类别（单类）
                data_img[class_name_one] = []
                img_dir = os.path.join(self.root, class_name_one, self.phase, 'good') # 训练集加载good
                img_types = sorted(os.listdir(img_dir))
                for img_type in img_types:
                    img_type_dir = os.path.join(img_dir, img_type)
                    data_img[class_name_one].append(img_type_dir)
                random.shuffle(data_img[class_name_one])

        query_dir, support_dir = [], []
        for class_name_one in data_img.keys():

            for image_index in range(0, len(data_img[class_name_one]), self.batch):
                query_sub_dir = []
                support_sub_dir = []

                for batch_count in range(0, self.batch):
                    if image_index + batch_count >= len(data_img[class_name_one]):
                        break
                    image_dir_one = data_img[class_name_one][image_index + batch_count]
                   
                    support_dir_one = []
                    query_sub_dir.append(image_dir_one)
                    for k in range(self.shot):
                        random_choose = random.randint(0, (len(data_img[class_name_one]) - 1))
                        while data_img[class_name_one][random_choose] == image_dir_one:
                            random_choose = random.randint(0, (len(data_img[class_name_one]) - 1))
                        support_dir_one.append(data_img[class_name_one][random_choose])
                    support_sub_dir.append(support_dir_one)
                
                query_dir.append(query_sub_dir)
                support_dir.append(support_sub_dir)

        assert len(query_dir) == len(support_dir), 'number of query_dir and support_dir should be same'
        return query_dir, support_dir
    
    def shuffle_dataset(self):
        data_img = {}
        # data_img includes all image pathes, key: class_name like wood, zipper. value: each image path.
        for class_name_one in CLASS_NAMES:
            # 差异化训练其它类，测试当前类
            if class_name_one != self.label: # 如果不是当前的类别，就将其加入到data_img中
                                                  # 这样训练基于配准的FSAD来学习类别不可知的特征配准，使模型能够在不调整正常图像的情况下检测新类别的异常。
                data_img[class_name_one] = [] 
                img_dir = os.path.join(self.root, class_name_one, self.phase, 'good')
                img_types = sorted(os.listdir(img_dir)) # 获取当前类别的所有子类别：图片
                for img_type in img_types:
                    img_type_dir = os.path.join(img_dir, img_type) 
                    data_img[class_name_one].append(img_type_dir) 
                random.shuffle(data_img[class_name_one])

        query_dir, support_dir = [], []
        for class_name_one in data_img.keys(): # key: class_name like wood, zipper, value: each image path.
            for image_index in range(0, len(data_img[class_name_one]), self.batch): # 每次取batch个图片
                query_sub_dir = []
                support_sub_dir = []
                
                for batch_count in range(0, self.batch): # 遍历batch个图片
                    if image_index + batch_count >= len(data_img[class_name_one]): # 如果超出了类别的图片数量，就跳出循环
                        break
                    image_dir_one = data_img[class_name_one][image_index + batch_count] # 取出一个图片
                    query_sub_dir.append(image_dir_one) # 一共追加batch个图片
                    
                    support_dir_one = [] # 用于存放支持集的图片
                    for k in range(self.shot):
                        random_choose = random.randint(0, (len(data_img[class_name_one]) - 1)) # 随机选择一个图片
                        while data_img[class_name_one][random_choose] == image_dir_one: # 如果随机选择的图片和query图片一样，重新选择
                            random_choose = random.randint(0, (len(data_img[class_name_one]) - 1)) # 重新选择一个图片
                        support_dir_one.append(data_img[class_name_one][random_choose]) # 将随机选择的图片加入到support_dir_one中
                    support_sub_dir.append(support_dir_one) # 每张图（Batch）追加两个，一共追加2*batch个图片（每个query图片对应2个support图片）
                
                query_dir.append(query_sub_dir) # 将每一批=>query_sub_dir加入到query_dir中（不管类别了）
                support_dir.append(support_sub_dir)

        assert len(query_dir) == len(support_dir), 'number of query_dir and support_dir should be same'
        self.query_dir = query_dir
        self.support_dir = support_dir


class TestDatasets(Dataset):
    def __init__(self,
                 root: str,
                 label: str,
                 phase: str,
                 x_resize: int,
                 y_resize: int,
                 mask_resize: int,
                 transform_x: transforms.Compose,
                 transform_y: transforms.Compose,
                 transform_mask: transforms.Compose, # 添加了mask
                 shot=2):
        assert label in CLASS_NAMES, 'class_name: {}, should be in {}'.format(label, CLASS_NAMES)
        self.root = root   #'/home/aorus/Desktop/Code/datasets/MVTec/MVTec-AD'
        self.label = label # 'pcb3'
        self.phase = phase # 'test'
        self.x_resize = x_resize
        self.y_resize = y_resize
        self.mask_resize = mask_resize
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.transform_mask = transform_mask
        self.shot = shot
        # load dataset
        self.query_dir, self.support_dir, self.query_mask = self.load_dataset_folder()
        
    def __getitem__(self, idx):
        query_one, support_one, mask_one = self.query_dir[idx], self.support_dir[idx], self.query_mask[idx]
        
        query_img = Image.open(query_one).convert('RGB')
        query_img = self.transform_y(query_img)
        query_img = query_img.unsqueeze(dim=0) # Batch已经默认为1了，这只是保证与训练集一致
        noise_img = query_img # 测试时无噪音


        # 假设 support_imgs 是图像路径列表
        support_imgs = ['Start/000.png', 'Start/000.png', 'Start/000.png', 'Start/000.png', 'Start/000.png']
        support_img_tensors = []

        # 加载图像并应用转换
        for img_path in support_imgs:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform_x(img)
            support_img_tensors.append(img_tensor)

        # 将图像张量堆叠为一个单一的张量
        support_img_tensor = torch.stack(support_img_tensors, dim=0)  # 列表沿着第0维展开为tensor => (B, C, H, W)
        support_img = support_img_tensor.unsqueeze(dim=0) # Batch已经默认为1了，这只是保证与训练集一致
        
        if 'good' in mask_one:
            mask = torch.zeros([1, self.mask_resize, self.mask_resize]) # 单独修改
            label = 0 # label 代表有无异常
        else:
            mask = Image.open(mask_one)
            mask = self.transform_mask(mask)
            label = 1
        return query_img, noise_img, support_img, mask, label

    def __len__(self):
        return len(self.query_dir)

    def load_dataset_folder(self):
        query_dir, support_dir = [], []
        # 所有数据类
        data_img = {}
        img_dir = os.path.join(self.root, self.label, self.phase) # '/home/aorus/Desktop/Code/datasets/MVTec/MVTec-AD/pcb3/test'
        img_types = sorted(os.listdir(img_dir)) # 所有的类 ['anomaly', 'good']
        for img_type in img_types:
            data_img[img_type] = []
            img_type_dir = os.path.join(img_dir, img_type)
            img_num = sorted(os.listdir(img_type_dir))
            for img_one in img_num:
                img_dir_one = os.path.join(img_type_dir, img_one)
                data_img[img_type].append(img_dir_one)
        # good数据类
        data_train = [] # good 里所有的数据图片的列表
        img_dir_train = os.path.join(self.root, self.label, 'train', 'good') #'/home/aorus/Desktop/Code/datasets/MVTec/MVTec-AD/pcb3/train/good'
        img_num = sorted(os.listdir(img_dir_train)) #good 下所有的数据列表
        for img_one in img_num:
            img_dir_one = os.path.join(img_dir_train, img_one)
            data_train.append(img_dir_one)

        # gt_dir = os.path.join(self.root, self.label, 'ground_truth') #'/home/aorus/Desktop/Code/datasets/MVTec/MVTec-AD/pcb3/ground_truth'

        query_dir, support_dir, query_mask = [], [], []
        #query_dir   数据列表 /home/aorus/Desktop/Code/datasets/MVTec/MVTec-AD/pcb3/test/anomaly/000.png'
        #support_dir 支持集数据列表
        #query_mask  mask   /home/aorus/Desktop/Code/datasets/MVTec/MVTec-AD/pcb3/ground_truth/anomaly/000_mask.png'
        #dict_keys(['anomaly', 'good'])
        
        query_dir.append('Start/000.png')
        query_mask.append('Start/000.png')
        support_dir.append('Start/000.png')  # 原来是选取的5张
        # for img_type in data_img.keys():
        #     for image_dir_one in data_img[img_type]:
        #         support_dir_one = []
        #         query_dir.append(image_dir_one)
        #         query_mask_dir = image_dir_one.replace('test', 'ground_truth')
        #         query_mask_dir = query_mask_dir[:-4] + '_mask.png' #'/home/aorus/Desktop/Code/datasets/MVTec/MVTec-AD/pcb3/ground_truth/anomaly/000.png'
        #         query_mask.append(query_mask_dir)
        #         for k in range(self.shot):    # 随机选5张
        #             random_choose = random.randint(0, (len(data_train) - 1))   #随机从训练集中找到一张
        #             support_dir_one.append(data_train[random_choose])
        #         support_dir.append(support_dir_one)




        assert len(query_dir) == len(support_dir) == len(
            query_mask), 'number of query_dir and support_dir should be same'
        return query_dir, support_dir, query_mask

def choose_random_aug_image(image):
    aug_index = random.choice([1,2,3])
    coefficient = random.uniform(0.8,1.2)
    if aug_index == 1:
        img_aug = transforms.functional.adjust_brightness(image,coefficient)
    elif aug_index == 2:
        img_aug = transforms.functional.adjust_contrast(image,coefficient)
    elif aug_index == 3:
        img_aug = transforms.functional.adjust_saturation(image,coefficient)
    return img_aug

def add_noise_to_image(image, noise, size=256):
    # 确保img是PyTorch张量
    if not isinstance(image, torch.Tensor):
        raise ValueError("img must be a PyTorch tensor")
    else:
        # 假设img的形状是Bx3xHxW，我们需要保持这个形状不变
        C, H, W = image.shape
    # 确保img的尺寸是Bx3xHxW，其中H和W都是256
    if image.shape[1:] != (size, size):
        raise ValueError("img must be of size Bx3x256x256")
    # 转为numpy
    image_np = image.detach().cpu().numpy().copy() # 不会改变原tensor
    noise_images = np.zeros_like(image_np)  # 创建一个相同形状的数组来存储带噪声的图像

    # 生成随机噪声区域的大小和起始位置
    h_noise = np.random.randint(int(size//2), int(size-1))
    w_noise = np.random.randint(int(size//2), int(size-1))
    start_h_noise = np.random.randint(1, size - h_noise)
    start_w_noise = np.random.randint(1, size - w_noise)

    # 生成噪声
    noise_size = (h_noise, w_noise)
    # 假设simplexNoise.rand_3d_octaves能够处理多维张量
    simplex_noise = noise.rand_3d_octaves((image.shape[0], *noise_size), 6, 0.6)
    init_zero = np.zeros((C, W, H)) 
    init_zero[:,
                start_h_noise: start_h_noise + h_noise, 
                start_w_noise: start_w_noise + w_noise] = 0.8 * simplex_noise

    # 将噪声添加到当前批次的图像上
    noise_images = image_np + init_zero

    # 还原为tensor
    noise_images_tensor = torch.tensor(noise_images).to(dtype=torch.float).to(image.device)

    return noise_images_tensor
