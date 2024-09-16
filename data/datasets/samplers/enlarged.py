import math
import torch
from torch.utils.data.sampler import Sampler


class EnlargedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        world_size (int | None): Number of processes participating in
            the training.  
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
            ratio参数的作用是允许你根据需要选择处理数据集的一部分，而不是全部
    """

    def __init__(self, 
                 dataset, 
                 rank,
                 world_size, 
                 ratio=1):
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.world_size) # math.ceil函数向上取整到最近的整数来实现的
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
