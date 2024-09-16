import os
from PIL import Image
import random
import math
import numpy as np
import torch
import torch.nn.functional as F

from base.base_config import BaseConfig
from base.base_dataset import BaseDataset
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'tile', # 'screw' 无
    'toothbrush', 'transistor', 'wood', 'zipper'
]

class VISA_Dataset(BaseDataset):
    def __init__(self,
                 cfg: BaseConfig,
                 root: str,
                 size: int,
                 label: str,
                 shot: int = 2,
                 batch: int = 32, # 手动的
                 batch_size: int = 1, # loder
                 num_workers: int = 2):
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
        self.train_set = VISA_train(root=root, label=label, phase= "train",
                                     mask_resize = self.mask_resize, 
                                     transform_x = self.transform_x,
                                     transform_y = self.transform_y,
                                     shot=shot,
                                     batch=batch)
        self.test_set = VISA_test(root=root, label=label, phase= "train", # 此处暂时为train
                                   mask_resize = self.mask_resize, 
                                   transform_x = self.transform_x,
                                   transform_y = self.transform_y,
                                   transform_mask= self.transform_mask,
                                   shot=shot) # batch=1 默认
        super().__init__(cfg,
                         root=root,
                         train_set=self.train_set,
                         test_set=self.test_set,
                         batch_size=batch_size,
                         num_workers=num_workers)
        
class VISA_train(Dataset):
    def __init__(self,
                 root: str,
                 label: str,
                 phase: str, # 训练train，而不是test
                 mask_resize: int,
                 transform_x: transforms.Compose,
                 transform_y: transforms.Compose,
                 shot=2,
                 batch=32):
        assert label in CLASS_NAMES, 'class_name: {}, should be in {}'.format(label, CLASS_NAMES)
        self.root = root
        self.label = label
        self.phase = phase
        self.mask_resize = mask_resize
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.shot = shot
        self.batch = batch
        # load dataset
        self.query_dir, self.support_dir = self.load_dataset_folder()
        
    def choose_random_aug_image(self, image):
        aug_index = random.choice([1,2,3])
        coefficient = random.uniform(0.8,1.2)
        if aug_index == 1:
            img_aug = transforms.functional.adjust_brightness(image,coefficient)
        elif aug_index == 2:
            img_aug = transforms.functional.adjust_contrast(image,coefficient)
        elif aug_index == 3:
            img_aug = transforms.functional.adjust_saturation(image,coefficient)
        return img_aug

    def __getitem__(self, idx):
        query_list, support_list = self.query_dir[idx], self.support_dir[idx]
        query_img = None
        support_sub_img = None
        support_img = None

        for i in range(len(query_list)):
            # 查询集（1）
            image = Image.open(query_list[i]).convert('RGB')
            image = self.transform_y(image) #image_shape torch.Size([3, 224, 224])
            image = image.unsqueeze(dim=0) #image_shape torch.Size([1, 3, 224, 224])
            if query_img is None:
                query_img = image
            else:
                query_img = torch.cat([query_img, image],dim=0)
            # 支持集
            for k in range(self.shot):
                image = Image.open(support_list[i][k]).convert('RGB')
                image = self.transform_x(image)
                image = self.choose_random_aug_image(image) # 增强处理
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
        return query_img, support_img, mask

    def __len__(self):
        return len(self.query_dir)
    
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
    def load_dataset_folder(self):
        data_img = {}
        for class_name_one in CLASS_NAMES:
            if class_name_one != self.label:
                data_img[class_name_one] = []
                img_dir = os.path.join(self.root, class_name_one, self.phase, 'good')
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


class VISA_test(Dataset):
    def __init__(self,
                 root: str,
                 label: str,
                 phase: str,
                 mask_resize: int,
                 transform_x: transforms.Compose,
                 transform_y: transforms.Compose,
                 transform_mask: transforms.Compose, # 添加了mask
                 shot=2):
        assert label in CLASS_NAMES, 'class_name: {}, should be in {}'.format(label, CLASS_NAMES)
        self.root = root
        self.label = label
        self.phase = phase
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

        support_img = []
        for k in range(self.shot):
            support_img_one = Image.open(support_one[k]).convert('RGB')
            support_img_one = self.transform_x(support_img_one)
            support_img.append(support_img_one)

        if 'good' in mask_one:
            mask = torch.zeros([1, self.mask_resize, self.mask_resize]) # 单独修改
            y = 0
        else:
            mask = Image.open(mask_one)
            mask = self.transform_mask(mask)
            y = 1
        return query_img, support_img, mask, y

    def __len__(self):
        return len(self.query_dir)

    def load_dataset_folder(self):
        query_dir, support_dir = [], []
        data_img = {}
        img_dir = os.path.join(self.root, self.label, self.phase)
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            data_img[img_type] = []
            img_type_dir = os.path.join(img_dir, img_type)
            img_num = sorted(os.listdir(img_type_dir))
            for img_one in img_num:
                img_dir_one = os.path.join(img_type_dir, img_one)
                data_img[img_type].append(img_dir_one)
        img_dir_train = os.path.join(self.root, self.label, 'train', 'good')
        img_num = sorted(os.listdir(img_dir_train))

        data_train = []
        for img_one in img_num:
            img_dir_one = os.path.join(img_dir_train, img_one)
            data_train.append(img_dir_one)

        gt_dir = os.path.join(self.root, self.label, 'ground_truth')
        query_dir, support_dir, query_mask = [], [], []
        for img_type in data_img.keys():
            for image_dir_one in data_img[img_type]:
                support_dir_one = []
                query_dir.append(image_dir_one)
                query_mask_dir = image_dir_one.replace('test', 'ground_truth')
                query_mask_dir = query_mask_dir[:-4] + '.png'
                query_mask.append(query_mask_dir)
                for k in range(self.shot):
                    random_choose = random.randint(0, (len(data_train) - 1))
                    support_dir_one.append(data_train[random_choose])
                support_dir.append(support_dir_one)

        assert len(query_dir) == len(support_dir) == len(
            query_mask), 'number of query_dir and support_dir should be same'
        return query_dir, support_dir, query_mask
