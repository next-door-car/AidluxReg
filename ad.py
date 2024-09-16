import os
import json
import torch
import logging
import cv2
import numpy as np
from PIL import Image
from utils.misc import master_only
from data.datasets import create_dataset
from base.base_config import BaseConfig
from networks.ad_net import ADNet
from networks.pe_net import PENet
from base.base_dataset import BaseDataset
from networks.main import build_network, build_pretrain_encoder
from trainer.ad_trainer import ADTrainer
from trainer.pe_trainer import PETrainer
from loggers.logger import  get_root_logger

# 以下两句等价，但后者更简洁
# logger = get_root_logger()

class ImageProcessor:
    def __init__(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path
        self.image = cv2.imread(image_path)
        self.mask = cv2.imread(mask_path)
        self.processed_image = None

    def image_angles(self, image):
        kerne2 = np.ones((15, 15), np.uint8)
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kerne2)
        edges = cv2.Canny(closing, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
        line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        angles1 = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles1.append(angle1)

        if angles1:
            average_angle1 = np.mean(angles1)
            # print(f'Average Angle1: {average_angle1}')
        else:
            average_angle1 = 0
            # print("No lines detected with average_angle1.")

        return average_angle1
    
    def calculate_black_density(self, image, threshold=50):
        black_pixels = np.sum(image < threshold)
        total_pixels = image.size
        return black_pixels / total_pixels
    
    def show(self, image):
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_image(self):
        image_array = np.array(self.image)
        mask_array = np.array(self.mask)
        result_array = image_array - mask_array

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        lower_green = np.array([36, 25, 25])
        upper_green = np.array([86, 255, 255])

        hsv_image = cv2.cvtColor(result_array, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        combined_mask = cv2.bitwise_or(yellow_mask, green_mask)
        result_array[combined_mask > 0] = [255, 255, 255]
        result_array[(result_array > 100).all(axis=-1)] = [255, 255, 255]

        result_image = Image.fromarray(result_array)
        result_cv2 = np.array(result_image)
        median_blur_image = cv2.medianBlur(result_cv2, 5)
        median_blur_image = cv2.GaussianBlur(median_blur_image, (5, 5), 0)
        image = cv2.cvtColor(median_blur_image, cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        average_angle1 = self.image_angles(opening)
        median_image = cv2.medianBlur(opening, 5)
        kerne3 = np.ones((40, 40), np.uint8)
        dilated_image = cv2.dilate(median_image, kerne3, iterations=1)
        edges = cv2.Canny(dilated_image, 70, 200, apertureSize=5, L2gradient=False)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        line_image = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2BGR)
        angles = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)

        if angles:
            average_angle = np.mean(angles)
            # print(f'Average Angle: {average_angle}')
        else:
            average_angle = 0
            # print("No lines detected with average_angle.")
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        best_rect = None
        best_box = None

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            area = rect[1][0] * rect[1][1]
            if area > max_area:
                max_area = area
                best_rect = rect
                best_box = box

        # print("最大矩形面积：", max_area)

        if best_rect is not None:
            current_width, current_height = best_rect[1]
            # print("当前矩形宽高：", current_width, current_height)
            width1 = current_width
            height1 = current_height
            if height1 > width1:
                width1, height1 = height1, width1
            if width1 < 1024:
                width1 = 1024
            if height1 < 384:
                height1 = 384

            expanded_rect = ((best_rect[0][0], best_rect[0][1]), (current_width, current_height), best_rect[2])
            expanded_box = cv2.boxPoints(expanded_rect)
            expanded_box = np.intp(expanded_box)

            cv2.drawContours(line_image, [expanded_box], 0, (255, 0, 0), 2)

            # 提取旋转外接矩形
            center = best_rect[0]
            size = (int(width1), int(height1))
            weight_factor = 0.98  # 设定权重因子，范围可以根据需要调整
            if average_angle1 != 0:
                angle = weight_factor * average_angle1 + (1 - weight_factor) * average_angle
            else:
                angle = average_angle

            # 获取旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rows, cols = self.image.shape[:2]

            # 增加边界填充
            padX = int(rows * 0.5)
            padY = int(cols * 0.5)
            padded_image = cv2.copyMakeBorder(self.image, padY, padY, padX, padX, cv2.BORDER_REPLICATE)

            # 更新中心点坐标
            new_center = (center[0] + padX, center[1] + padY)

            # 重新计算旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(new_center, angle, 1.0)
            rotated_image = cv2.warpAffine(padded_image, rotation_matrix, (cols + 2 * padX, rows + 2 * padY), flags=cv2.INTER_CUBIC)

            # 提取旋转后的图像
            extracted_image = cv2.getRectSubPix(rotated_image, size, new_center)

            gray_extracted_image = cv2.cvtColor(extracted_image, cv2.COLOR_BGR2GRAY)
            width1 = int(width1)
            if width1 % 2 == 1:
                width1 += 1
            left_half = gray_extracted_image[:, :width1 // 2]
            right_half = gray_extracted_image[:, width1 // 2:]
            left_density = self.calculate_black_density(left_half)
            right_density = self.calculate_black_density(right_half)

            if left_density > right_density:
                extracted_image = cv2.rotate(extracted_image, cv2.ROTATE_180)
            self.processed_image = extracted_image
            
            return self.processed_image
        else:
            print("No contours found.")


logger = logging.getLogger('basicad')

class AD(object):
    """A class for the Deep SVDD method.

    Attributes:
        objective: A string specifying the Deep SVDD objective (either 'one-class' or 'soft-boundary').
        nu: Deep SVDD hyperparameter nu (must be 0 < nu <= 1).
        R: Hypersphere radius R.
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network.
        ae_net: The autoencoder network corresponding to for network weights pretraining.
        trainer: DeepSVDDTrainer to train a Deep SVDD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SVDD network.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
    """

    def __init__(self, cfg: BaseConfig):
        """Inits DeepSVDD with one of the two objectives and hyperparameter nu."""

        self.cfg = cfg
        
        self.dataset : BaseDataset
        
        self.ad_net : ADNet  # neural network \phi
        self.net_name : str
        self.pe_net : PENet
        self.pe_net_name : str
        
        self.ad_trainer : ADTrainer = None
        self.pe_trainer : PETrainer = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }
    
    # def create_dataset(self):
    #     dataset_config = self.cfg.get_config(['datasets'])
    #     dataset_args = dict(
    #         cfg=self.cfg,
    #         root=self.cfg.get_config(['datasets', 'dataset_path']),
    #         seed=self.cfg.get_config(['seed']),
    #         # 数据集配置
    #         size=self.cfg.get_config(['datasets', 'size']),
    #         label=self.cfg.get_config(['datasets', 'label']),
    #         shot=self.cfg.get_config(['datasets', 'shot']), 
    #         batch=self.cfg.get_config(['datasets', 'batch']),
    #         # 分布式配置
    #         dist=self.cfg.get_config(['model', 'dist']),
    #         num_gpu=self.cfg.get_config(['model', 'num_gpu']),
    #         rank=self.cfg.get_config(['dist_params', 'rank']),
    #         world_size=self.cfg.get_config(['dist_params', 'world_size']),
    #         # 数据集加载配置
    #         batch_size_per_gpu=self.cfg.get_config(['datasets', 'batch_size_per_gpu']), 
    #         num_worker_per_cpu=self.cfg.get_config(['datasets', 'num_worker_per_cpu']),
    #         dataset_enlarge_ratio=self.cfg.get_config(['datasets', 'dataset_enlarge_ratio'])
    #     )
    #     # 数据集
    #     self.dataset = create_dataset(dataset_config=dataset_config,
    #                                   **dataset_args) # 解包关键字参数
    #     # 数据集加载放置训练器中
    #     pass
    def create_dataset(self):
        dataset_config = self.cfg.get_config(['datasets'])
        dataset_args = dict(
            cfg=self.cfg,
            root=self.cfg.get_config(['datasets', 'dataset_path']),
            seed=self.cfg.get_config(['seed']),
            # 数据集配置
            size=self.cfg.get_config(['datasets', 'size']),
            label=self.cfg.get_config(['datasets', 'label']),
            shot=self.cfg.get_config(['datasets', 'shot']), 
            batch=self.cfg.get_config(['datasets', 'batch']),
            # 分布式配置
            dist=self.cfg.get_config(['model', 'dist']),
            num_gpu=self.cfg.get_config(['model', 'num_gpu']),
            rank=self.cfg.get_config(['dist_params', 'rank']),
            world_size=self.cfg.get_config(['dist_params', 'world_size']),
            # 数据集加载配置
            batch_size_per_gpu=self.cfg.get_config(['datasets', 'batch_size_per_gpu']), 
            num_worker_per_cpu=self.cfg.get_config(['datasets', 'num_worker_per_cpu']),
            dataset_enlarge_ratio=self.cfg.get_config(['datasets', 'dataset_enlarge_ratio'])
        )
        # 数据集
        self.dataset = create_dataset(dataset_config=dataset_config,
                                      **dataset_args) # 解包关键字参数
        # 数据集加载放置训练器中
        pass
    
    def load_supportset(self):
        support_save_path = self.cfg.get_config(['datasets', 'support_path']) + '/' + \
                            self.cfg.get_config(['datasets', 'dataset_name']) + '/' + \
                            self.cfg.get_config(['datasets', 'label']) + '/'
        if not os.path.exists(support_save_path):
            os.makedirs(support_save_path)
        support_save_name = os.path.join(support_save_path, '{}_{}_{}.pt'.format(self.cfg.get_config(['trainner', 'test_rounds']),
                                                                                 self.cfg.get_config(['datasets', 'batch']),
                                                                                 self.cfg.get_config(['datasets', 'shot'])))
        _, test_loader = self.dataset.loaders() # 训练的good图
        support_img_list = []
        for _, support_img, _, _ in test_loader:
            # support_img.shape = (Batch_size=1, Batch=1, K, C, H, W)
            support_img_list.append(support_img.squeeze(dim=0)) # 添加列表到新的列表
            if len(support_img_list) == self.cfg.get_config(['trainner', 'test_rounds']):
                break
        torch.save(support_img_list, support_save_name)
        # fixed_fewshot_list = torch.load(support_save_name) 
    
    def create_network(self, net_name):
        self.net_name = net_name
        self.ad_net = build_network(self.cfg, net_name)

    def train(self, device: str, n_epochs: int):

        self.ad_trainer = ADTrainer(cfg=self.cfg,
                                    net=self.ad_net,
                                    dataset=self.dataset,
                                    device=device,
                                    n_epochs=n_epochs)
        self.ad_trainer.print_network() # 打印训练器中的网络
        resume_path = self.ad_trainer.cfg.get_config(['path', 'resume_state'])
        strict_path = self.ad_trainer.cfg.get_config(['path', 'strict_load'])
        pretrain_path = self.ad_trainer.cfg.get_config(['path', 'pretrain_load'])
        if resume_path is not None:
            # 恢复训练
            device_id = torch.cuda.current_device()
            resume_network_path = self.ad_trainer.check_resume(resume_path)
            resume_network_state = torch.load(
                resume_path,
                map_location=lambda storage, loc: storage.cuda(device_id))
            # map_location 参数是一个函数，它用于修改加载模型状态时的存储位置。
            # lambda storage, loc: storage.cuda(device_id) 是一个匿名函数（lambda 函数）
            # 它接受两个参数：storage（模型状态的存储对象）和 loc（原本的存储位置）
            # 返回一个新的存储对象，该对象已经被移动到了当前进程的 GPU 设备上（通过 .cuda(device_id) 实现）。
            self.ad_trainer.resume_training(resume_network_state)
            self.ad_trainer.load_network(resume_network_path, strict=True) # 严格加载
        else:
            # 是否有网络预训练
            if strict_path is not None and pretrain_path is not None:
                raise NotImplementedError(
                    f'strict_path {strict_path} and pretrain_path {pretrain_path} only one.')
            elif strict_path is not None:
                self.ad_trainer.load_network(load_path=strict_path, strict=True)
            elif pretrain_path is not None:
                self.ad_trainer.load_network(load_path=pretrain_path, strict=False)
            else:
                print('no pre network load')
        
        # Get the model
        self.ad_net = self.ad_trainer.train()
        self.results['train_time'] = self.ad_trainer.train_time

    def test(self, device: str, n_epochs: int , origin_picture , CL_picture):

        if self.ad_trainer is None:
            self.ad_trainer = ADTrainer(cfg=self.cfg,
                                        net=self.ad_net,
                                        dataset=self.dataset,
                                        device=device,
                                        n_epochs=n_epochs)
        # self.ad_trainer.print_network() # 打印训练器中的网络
        strict_path = self.ad_trainer.cfg.get_config(['path', 'strict_load'])
        pretrain_path = self.ad_trainer.cfg.get_config(['path', 'pretrain_load'])
        if strict_path is not None:
            self.ad_trainer.load_network(load_path=strict_path, strict=True)
        else:
            print('no pre network load')
        self.ad_trainer.test(origin_picture , CL_picture)

    def pretrain(self, device: str, n_epochs: int):

        self.pe_net = build_pretrain_encoder(self.cfg, net_name='PENet')
        self.pe_trainer = PETrainer(cfg=self.cfg,
                                    net=self.pe_net,
                                    dataset=self.dataset,
                                    n_epochs=n_epochs,
                                    device=device)
        self.pe_net = self.pe_trainer.train() # 训练+保存
        # self.pe_trainer.test(dataset, self.pretrain_encoder)
        self.init_network_weights_from_pretrain()

    def init_network_weights_from_pretrain(self):
        """
        Initialize the Deep SVDD network weights from the encoder weights of the pretraining autoencoder.
        方法，用于将预训练自动编码器的权重初始化到 Deep SVDD 网络中
        """
        ad_net_dict = self.ad_net.state_dict()
        pe_net_dict = self.pe_net.state_dict()

        # Filter out decoder network keys
        pe_net_dict = {k: v for k, v in pe_net_dict.items() if k in ad_net_dict}
        # Overwrite values in the existing state_dict
        ad_net_dict.update(pe_net_dict)
        # Load the new state_dict
        self.ad_net.load_state_dict(ad_net_dict)

    @master_only
    def save_model(self, save_pe=True):
        """
        保存最终的模型，不能用于加载网络的训练
        save_ae: 布尔值，表示是否同时保存自动编码器模型。
        """
        model_name = self.cfg.get_config(['model', 'name'])
        save_name = f'{model_name}.pth'
        save_path = os.path.join(self.cfg.get_config(['path', 'models']), 
                                 save_name)
        net_dict = self.ad_net.state_dict()
        pe_net_dict = self.pe_net.state_dict() if save_pe else None
        logger.info('Saving last models.')
        torch.save({'net_dict': net_dict,
                    'ae_net_dict': pe_net_dict}, save_path)
    
    @master_only
    def save_results(self, export_json):
        """
        Save results dict to a JSON-file.
        方法，将实验结果保存为 JSON 文件。
				export_json: 保存结果的 JSON 文件路径。 
        """
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
