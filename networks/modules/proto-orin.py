''' Layers
    This file contains various layers for the BigGAN models.
'''
import functools
from tkinter import X
import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.autograd import Variable
import faiss

from sklearn.cluster import KMeans

def hard_shrink_relu(x, lambd=0.0, epsilon=1e-12):
    ''' Hard Shrinking '''
    return (F.relu(x-lambd) * x) / (torch.abs(x-lambd) + epsilon)

class ProtoMemory(nn.Module):
    """concept attention"""

    def __init__(self, 
                 ch, feat, 
                 num_k, pool_size_per_cluster,
                 warmup_total_iter=100,
                 which_conv=nn.Conv2d,
                 cp_momentum=1, 
                 cp_phi_momentum=0.6, 
                 device='cuda', 
                 use_sa=True):
        super(ProtoMemory, self).__init__()
        self.myid = "atten_concept_prototypes"
        self.device = device 
        self.ch = ch  # input channel
        self.feat = feat # feature dim channel
        self.num_k = num_k # 每个cluster的prototype数量
        self.pool_size_per_cluster = pool_size_per_cluster # 这是每个cluster的pool size
        self.total_pool_size = self.num_k * self.pool_size_per_cluster # total pool size
        # concept pool is arranged as memory cell, i.e. linearly arranged as a 2D tensor, use get_cluster_ptr to get starting pointer for each cluster
        # 这样，concept_pool 和 concept_proto 就会被注册为模型的缓冲区，但不会被优化器更新，因为它们不是 nn.Parameter。
        self.register_buffer('concept_pool', torch.rand(self.feat, self.total_pool_size))
        self.register_buffer('concept_proto',  torch.rand(self.feat, self.num_k))
        
        # states that indicating the warmup
        self.register_buffer('warmup_iter_counter', torch.FloatTensor([0.]))
        self.warmup_total_iter = warmup_total_iter # 这是warmup的总iter
        self.register_buffer('pool_structured', torch.FloatTensor([0.]))  
        ## 0 means pool is un clustered, 1 mean pool is structured as clusters arrays
        
        # 定义两个全连接层
        self.hidden = self.feat*2
        self.fc1 = nn.Linear(self.num_k, self.hidden)  # 第一个全连接层的大小可以根据需要调整
        self.fc2 = nn.Linear(self.hidden, self.num_k)  # 第二个全连接层输出大小与输入相同
        
        # register attention module
        self.which_conv = which_conv # 这是一个卷积层
        self.theta = self.which_conv(
            self.ch, self.feat, kernel_size=1, padding=0, bias=False) # 生成查询（query）特征。

        self.phi = self.which_conv(
            self.ch, self.feat, kernel_size=1, padding=0, bias=False) # 生成键（key）特征，与查询一起用于计算注意力得分。

        self.phi_k = [self.which_conv(
            self.ch, self.feat, kernel_size=1, padding=0,
            bias=False).cuda()] # phi_k 代表原型单元，是个列表
        # using list to prevent pytorch consider phi_k as a parameter to optimize

        for param_phi, param_phi_k in zip(self.phi.parameters(), self.phi_k[0].parameters()):
            param_phi_k.data.copy_(param_phi.data)  # initialize
            param_phi_k.requires_grad = False  # not update by gradient

        self.g = self.which_conv(
            self.ch, self.feat, kernel_size=1, padding=0, bias=False) # 生成用于计算注意力权重的得分特征。
        self.o = self.which_conv(
            self.feat, self.ch, kernel_size=1, padding=0, bias=False) # 将加权后的注意力特征映射回原始特征维度，生成最终的输出。
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

        self.norm_layer = nn.BatchNorm2d(num_features=ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # self.momentum
        self.cp_momentum = cp_momentum
        self.cp_phi_momentum = cp_phi_momentum

        # self attention
        self.use_sa = use_sa

        # reclustering
        self.re_clustering_iters = 100
        self.clustering_counter = 0

    @torch.no_grad()
    def forward_update_pool(self, activation, cluster_num, momentum=None):
        """update activation into the concept pool after warmup in each forward pass
        activation: [m, c]
        cluster_num: [m, ]
        momentum: None or a float scalar
        """

        if not momentum:
            momentum = 1.

        # generate update index
        assert cluster_num.max() < self.num_k
        # index the starting pointer of each cluster add a rand num
        # 生成更新索引
        index = cluster_num * self.pool_size_per_cluster + \
                torch.randint(self.pool_size_per_cluster, size=(cluster_num.shape[0],)).to(self.device)

        # adding momentum to activation
        # 动量更新：接下来，代码使用动量（momentum）来更新概念池。
        # momentum 是一个介于 0 到 1 之间的浮点数，它控制着新激活（activation）与概念池中原有内容的混合比例。
        # activation 是传入该方法的激活张量，其转置（.T）后用于更新。
        self.concept_pool[:, index] = (1. - momentum) * self.concept_pool[:,index].clone() + \
                                            momentum * activation.detach().T # 
                                            
    @torch.no_grad()
    def computate_prototypes(self):
        """compute prototypes based on current pool"""
        assert not self._get_warmup_state(), f"still in warm up state {self.warmup_state}, computing prototypes is forbidden"
        self.concept_proto = self.concept_pool.detach().clone().reshape(self.feat, self.num_k,
                                                                        self.pool_size_per_cluster).mean(2) # shape [c, k]

    #############################
    # Initialization and warmup #
    #############################
    @torch.no_grad()
    def pool_kmean_init_gpu(self, seed=0, gpu_num=0, temperature=1):
        """TODO: clear up
        perform kmeans for cluster concept pool initialization
        Args:
            x: data to be clustered
        """

        print('performing kmeans clustering')
        results = {'im2cluster': [], 'centroids': [], 'density': []}
        x = self.concept_pool.clone().cpu().numpy().T
        x = np.ascontiguousarray(x) # 内存连续
        num_cluster = self.num_k
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k) # faiss 聚类
        clus.verbose = True
        clus.niter = 100
        clus.nredo = 10
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources() # 创建一个 GPU 资源对象，用于管理和分配 GPU 相关的资源。
        cfg = faiss.GpuIndexFlatConfig() # 创建一个 GPU 索引配置对象，用于设置 GPU 索引的参数。
        cfg.useFloat16 = False # 设置配置，指定不使用半精度浮点数（float16），即使用标准的单精度浮点数（float32）。
        cfg.device = gpu_num # 设置 GPU 设备编号
        index = faiss.GpuIndexFlatL2(res, d, cfg) # 创建一个基于 GPU 的平坦 L2 距离索引，d 是向量的维度，L2 表示使用欧几里得距离（L2范数）来衡量距离。

        clus.train(x, index) # 使用训练数据 x 来训练聚类对象 clus，这里 index 用于加速训练过程中的搜索。

        D, I = index.search(x, 1)  # 在索引中搜索每个样本，找到最近的簇的距离 D 和分配 I。
        im2cluster = [int(n[0]) for n in I] # 将搜索结果中的簇分配转换为整数列表。

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d) # 获取聚类中心，并将其转换为 NumPy 数组。

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(k)] # 为每个簇收集样本到簇中心的距离。
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        # concentration estimation (phi)
        # 计算每个簇的密度估计，这里使用了一种启发式方法，基于簇内样本到中心的平均距离和样本数量
        density = np.zeros(k) 
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

        # if cluster only has one point, use the max to estimate its concentration
        # 对密度进行调整，确保没有极端值。
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10),
                               np.percentile(density, 90))  # clamp extreme values for stability
        print(density.mean())
        # 根据温度参数调整密度，使其平均值为温度值。
        density = temperature * density / density.mean()  # scale the mean to temperature

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids)
        centroids = nn.functional.normalize(centroids, p=2, dim=1)
        # 转换为 PyTorch 张量，以便后续操作。
        im2cluster = torch.LongTensor(im2cluster)
        density = torch.Tensor(density)

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)

        del cfg, res, index, clus

        # rearrange
        self.structure_memory_bank(results)
        print("Finish kmean init...")
        del results

    @torch.no_grad()
    def pool_kmean_init(self, seed=0, gpu_num=0, temperature=1):
        """TODO: clear up
        perform kmeans for cluster concept pool initialization
        Args:
            x: data to be clustered
        """

        print('performing kmeans clustering')
        results = {'im2cluster': [], 'centroids': [], 'density': []}
        x = self.concept_pool.clone().cpu().numpy().T # shape [total_pool_size, feat]
        x = np.ascontiguousarray(x) # 确保内存连续
        num_cluster = self.num_k

        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(x) # 聚类
        # n_clusters=num_cluster: 这是 KMeans 类的一个参数，指定了要形成的聚类数量。
        # 用于对数据集 x 进行拟合（训练）。x 应该是一个二维数组，其中每一行代表一个数据点，每一列代表一个特征。
        # random_state 可以保证每次运行代码时，随机数生成器产生的随机数序列是相同的，这有助于结果的可复现性。

        centroids = torch.Tensor(kmeans.cluster_centers_) # shape [n_clusters, c]
        centroids = nn.functional.normalize(centroids, p=2, dim=1)
        im2cluster = torch.LongTensor(kmeans.labels_) # [total_pool_size,] 聚类标签

        results['centroids'].append(centroids) # [num_k, feature_dim] 聚类中心
        results['im2cluster'].append(im2cluster) # [total_pool_size,] 聚类标签

        # rearrange
        self.structure_memory_bank(results)
        print("Finish kmean init...")

    @torch.no_grad()
    def structure_memory_bank(self, cluster_results):
        """make memory bank structured """
        # 0 代表最新的聚类结果
        centeriod = cluster_results['centroids'][0]  # [num_k, feature_dim]
        cluster_assignment = cluster_results['im2cluster'][0]  # [total_pool_size,]

        mem_index = torch.zeros(self.total_pool_size).long()  # array of memory index that contains instructions of how to rearange the memory into structured clusters
        memory_states = torch.zeros(self.num_k, ).long()  # 0 indicate the cluster has not finished structured
        memory_cluster_insert_ptr = torch.zeros(self.num_k, ).long()  # ptr to each cluster block

        # loop through every cluster assignment to populate the concept pool for each cluster seperately
        # 填充记忆
        for idx, i in enumerate(cluster_assignment):
            cluster_num = i
            if memory_states[cluster_num] == 0:
                # manipulating the index for populating memory
                mem_index[cluster_num * self.pool_size_per_cluster + memory_cluster_insert_ptr[cluster_num]] = idx
                memory_cluster_insert_ptr[cluster_num] += 1 # 依次加入
                if memory_cluster_insert_ptr[cluster_num] == self.pool_size_per_cluster:
                    memory_states[cluster_num] = 1 - memory_states[cluster_num] # 置1
            else:
                # check if the ptr for this class is set to the last point
                assert memory_cluster_insert_ptr[cluster_num] == self.pool_size_per_cluster # 断言当前族是否填充完

        # 检查并处理未完全填充的聚类
        # what if some cluster didn't get populated enough? -- replicate
        not_fill_cluster = torch.where(memory_states == 0)[0]
        #print(f"memory_states {memory_states}")
        #print(f"memory_cluster_insert_ptr {memory_cluster_insert_ptr}")
        for unfill_cluster in not_fill_cluster:
            # 未填充的族
            cluster_ptr = memory_cluster_insert_ptr[unfill_cluster] # 当前填充的位置
            assert cluster_ptr != 0, f"cluster_ptr {cluster_ptr} is zero!!!" # 必须至少得有一个
            # 已经填充的部分
            existed_index = mem_index[
                            unfill_cluster * self.pool_size_per_cluster: unfill_cluster * self.pool_size_per_cluster + cluster_ptr]
            # print(f"existed_index {existed_index}")
            # print(f"cluster_ptr {cluster_ptr}")
            # print(f"(self.pool_size_per_cluster {self.pool_size_per_cluster}")
            # 复制的次数 
            replicate_times = (self.pool_size_per_cluster // cluster_ptr) + 1  # with more replicate and cutoff
            # print(f"replicate_times {replicate_times}")
            replicated_index = torch.cat([existed_index for _ in range(replicate_times)]) # 复制
            # print(f"replicated_index {replicated_index}")
            # permutate the replicate and select pool_size_per_cluster num of index
            replicated_index = replicated_index[
                torch.randperm(replicated_index.shape[0])][
                :self.pool_size_per_cluster]  # [pool_size_per_cluster, ] # 复制后去除了多余的部分
            # put it back
            assert replicated_index.shape[0] == self.pool_size_per_cluster, f"replicated_index ({replicated_index.shape}) should has the same len as pool_size_per_cluster ({self.pool_size_per_cluster})"
            # 将复制好的族 放入
            mem_index[unfill_cluster * self.pool_size_per_cluster: (unfill_cluster + 1) * self.pool_size_per_cluster] = replicated_index
            # update ptr
            memory_cluster_insert_ptr[unfill_cluster] = self.pool_size_per_cluster
            # update state
            memory_states[unfill_cluster] = 1

        assert (memory_states == 0).sum() == 0, f"memory_states has zeros: {memory_states}"
        assert (memory_cluster_insert_ptr != self.pool_size_per_cluster).sum() == 0, f"memory_cluster_insert_ptr didn't match with pool_size_per_cluster: {memory_cluster_insert_ptr}"

        # 更新记忆
        # update the real pool
        self._update_pool(torch.arange(mem_index.shape[0]), self.concept_pool[:, mem_index])
        # initialize the prototype
        self._update_prototypes(torch.arange(self.num_k), centeriod.T.cuda())
        #print(f"Concept pool updated by kmeans clusters...")

    def warmup_sampling(self, x):
        """
        linearly sample input x to make it
        x: [n, c, h, w]"""
        n, c, h, w = x.shape
        # 确保warmup_state为1，已经启动了warmup，再采样
        assert self._get_warmup_state(), "calling warmup sampling when warmup state is 0"

        # evenly distributed across space
        # self.total_pool_size = self.num_k * self.pool_size_per_cluster # total_pool_size
        sample_per_instance = max(int(self.total_pool_size / n), 1) # 分成n个实例，每个实例有sample_per_instance个样本

        # sample index
        index = torch.randint(h * w, size=(n, 1, sample_per_instance)).repeat(1, c, 1).to(self.device) 
        # (n, c, sample_per_instance) 通道1复制c次，其中包含了从 0 到 h * w - 1 之间的随机整数。
        
        sampled_columns = torch.gather(x.reshape(n, c, h * w), 2, index) # n, c, sample_per_instance
        # 第 2 维（即最后一个维度，像素索引）中选择元素。

        sampled_columns = torch.transpose(sampled_columns, 1, 0).reshape(c,-1).contiguous() # c, n * sample_per_instance
        # transpose 函数调用会转置 sampled_columns 张量
        # contiguous 用于确保张量在内存中是连续的。

        # calculate percentage to populate into pool, as the later the better, use linear intepolation from 1% to 50% according to self.warmup_iter_couunter
        percentage = (self.warmup_iter_counter + 1) / self.warmup_total_iter * 0.5  # max percent is 50%
        # 计算一个百分比值，该值从 1% 线性增加到 50%，这个百分比基于当前的预热迭代次数 
        
        print(f"percentage {percentage.item()}")
        sample_column_num = max(1, int(percentage * sampled_columns.shape[1])) # 根据计算出的百分比，确定最终要采样的列数 sample_column_num。
        sampled_columns_idx = torch.randint(sampled_columns.shape[1], size=(sample_column_num,)) # 随机选择 sample_column_num 个列索引
        sampled_columns = sampled_columns[:, sampled_columns_idx]  # [c, sample_column_num] 采样列数

        # random select pool idx to update
        update_idx = torch.randperm(self.concept_pool.shape[1])[:sample_column_num] # 从随机排列中选择前 sample_column_num 个元素。
        self._update_pool(update_idx, sampled_columns) # 更新概念池

        # update number
        # print(f"before self.warmup_iter_counter {self.warmup_iter_counter}")
        self.warmup_iter_counter += 1 # 更新预热迭代次数
        # print(f"after self.warmup_iter_counter {self.warmup_iter_counter}")

    def gumbel_softmax_sampling(self, logits, temperature=1.0, hard=False, dim=-1):
        # Step 1: 为每个logit生成Gumbel噪声
        gumbel_noise = -torch.empty_like(logits).exponential_().log()  # 从标准Gumbel分布中采样
        # Step 2: 将Gumbel噪声添加到logits上
        logits_with_noise = logits + gumbel_noise
        # Step 3: 应用softmax函数
        soft_probs = F.softmax(logits_with_noise / temperature, dim=dim)
        # Step 4: 可选：硬Gumbel-Softmax
        if hard:
            # 使用one-hot编码
            _, idx = soft_probs.max(dim=dim, keepdim=True)
            gumbel_softmax_sample = torch.zeros_like(logits).scatter_(dim, idx, 1.0)
            return gumbel_softmax_sample
        else:
            # 返回软概率
            return soft_probs

    #############################
    #       Forward logic       #
    #############################
    def forward(self, x, evaluation=False):
        # warmup
        if self._get_warmup_state(): # 没有达到warmup状态
            # 映射到低维
            theta = self.theta(x)  # [n, c, h, w]
            phi = self.phi(x)  # [n, c, h, w]
            g = self.g(x)  # [n, c, h, w]
            n, c, h, w = theta.shape
            # if still in warmup, skip attention
            self.warmup_sampling(phi) # 随机采样
            self._check_warmup_state() # 检查是否需要将warup_state设置为0; 关闭预热状态时，触发k-means init进行集群

            # 注意力机制
            # normal self attention
            theta = theta.view(-1, self.feat, x.shape[2] * x.shape[3]) # 查询（变换）
            phi = phi.view(-1, self.feat, x.shape[2] * x.shape[3]) # key
            g = g.view(-1, self.feat, x.shape[2] * x.shape[3]) # 得分
            # Matmul and softmax to get attention maps
            # 矩阵乘法（torch.bmm）
            beta = F.softmax(torch.bmm(theta.transpose(1, 2).contiguous(), phi), -1)
            # Attention map times g path
            # self.o 输出调制
            o = self.o(torch.bmm(g, beta.transpose(1, 2).contiguous()).view(-1, self.feat, x.shape[2], x.shape[3]))
            # gamma 也是可以学习的参数，用于调制参数
            return self.norm_layer(o) # 注意力得分，得到空间上下文调制张量
        else:
            # 达到warmup状态
            # transform into low dimension
            theta = self.theta(x)  # [n, c, h, w]
            phi = self.phi(x)
            g = self.g(x)  # [n, c, h, w]
            n, c, h, w = theta.shape
            
            # attend to concepts
            ## selecting cooresponding prototypes -> [n, h, w]
            theta = torch.transpose(torch.transpose(theta, 0, 1).reshape(c, n * h * w), 0, 1).contiguous()  # n * h * w, c
            phi = torch.transpose(torch.transpose(phi, 0, 1).reshape(c, n * h * w), 0, 1).contiguous()  # n * h * w, c
            g = torch.transpose(torch.transpose(g, 0, 1).reshape(c, n * h * w), 0, 1).contiguous()  # n * h * w, c
            # ---------------
            with torch.no_grad():
                # 概念记忆
                concept_proto_T = self.concept_proto.permute(1, 0) # shape = (num_k, feature_dim(c))
                # 概念原型注意力计算
                theta_atten_proto = torch.matmul(theta, self.concept_proto.detach().clone())  # n * h * w, num_k
                ## self.concept_proto.shape = (feature_dim(c), num_k)
                ## 使用 theta 和概念原型 self.concept_proto 计算注意力，得到 cluster_affinity
                ## 这是每个数据点属于每个概念原型的概率分布。
                cluster_affinity = F.softmax(theta_atten_proto, dim=1)  # n * h * w, num_k
                ## print(f"cluster_affinity.max(1) {cluster_affinity.max(1)}")
            self.shrink_thres = 0.0025
            if self.shrink_thres > 0:
                # hard shrink
                similarity_clusters = hard_shrink_relu(cluster_affinity, self.shrink_thres) # 作用是避免过拟合
                # re-normalize
                similarity_clusters = F.normalize(similarity_clusters, p=1, dim=1)    # [N, M]
                concept_residuals = torch.mm(similarity_clusters, concept_proto_T).reshape(n, h*w, c) # [N, K] x [K, C] = [N, C]
            else:
                # gumbel shrink
                average_cluster_affinity = torch.mean(cluster_affinity, dim=0)  # [k] 计算每个k下的平均
                activate_cluster_affinity = self.fc2(F.relu(self.fc1(average_cluster_affinity))) # fc1 => relu => fc2
                gumbel_cluster_weight = self.gumbel_softmax_sampling(activate_cluster_affinity) # shape [k]
                # 内存调制
                similarity_clusters = cluster_affinity * gumbel_cluster_weight # shape [N, k]
                concept_residuals = torch.mm(similarity_clusters, concept_proto_T).reshape(n, h*w, c) # [N, K] x [K, C] = [N, C]
            
            # dot product with context 
            # 空间上下文权重（相似度） = 输入特征 theta 和 phi 之间的交互
            similarity_context = torch.bmm(theta.reshape(n, h*w, c), torch.transpose(phi.reshape(n, h*w, c), 1, 2))  # [n, h*w, h*w]
            context_residuals = torch.bmm(similarity_context, g.reshape(n, h*w, c))  # [n, h*w, c]
            
            beta_residual = concept_residuals + context_residuals
            # integrate context residual with pool residual
            beta_residual = torch.transpose(beta_residual, 1, 2).reshape(n, c, h, w).contiguous()
            # print(f"beta_residual {beta_residual.shape}")
            o = self.o(beta_residual)  # n, h, w, c
            
            # update pool
            with torch.no_grad():
                phi_k = self.phi_k[0](x)  # [n, c, h, w]
                phi_k = torch.transpose(torch.transpose(phi_k, 0, 1).reshape(c, n * h * w), 0, 1).contiguous()  # n * h * w, c
                phi_k_atten_proto = torch.matmul(phi_k, self.concept_proto.detach().clone())  # n * h * w, num_k
                phi_k_cluster_affinity = F.softmax(phi_k_atten_proto, dim=1)  # n * h * w, num_k
                # print(f"cluster_affinity.max(1) {cluster_affinity.max(1)}")
                cluster_assignment_phi_k = phi_k_cluster_affinity.max(1)[1] # [n * h * w, ]

                # update pool first to allow contextual information
                # should use the lambda to update concept pool momentumlly
                self.forward_update_pool(phi_k, cluster_assignment_phi_k, momentum=self.cp_momentum) # 混合更新
                # update prototypes
                self.computate_prototypes() # 重新计算均值，族的均值

                # update phi_k
                for param_q, param_k in zip(self.phi.parameters(), self.phi_k[0].parameters()):
                    # 对每一对卷积的参数，执行动量更新
                    param_k.data = param_k.data * self.cp_phi_momentum + param_q.data * (1. - self.cp_phi_momentum)

                # perform clustering again and re-assign the prototypes
                self.clustering_counter += 1
                self.clustering_counter = self.clustering_counter % self.re_clustering_iters # 重新集群
                # print(f"self.clustering_counter {self.clustering_counter}; self.re_clustering_iters {self.re_clustering_iters}")
                if self.clustering_counter == 0:
                    self.pool_kmean_init()
                    
            if evaluation:
                return o, cluster_affinity

            return self.norm_layer(o)
        
    #############################
    #       Pool operation      #
    #############################
    @torch.no_grad()
    def _update_pool(self, index, content):
        """update concept pool according to the content
        index: [m, ]
        content: [c, m]
        """
        assert len(index.shape) == 1
        assert content.shape[1] == index.shape[0] # 列数
        assert content.shape[0] == self.feat # c

        # print("Updating concept pool...")
        self.concept_pool[:, index] = content.clone()

    @torch.no_grad()
    def _update_prototypes(self, index, content):
        assert len(index.shape) == 1
        assert content.shape[1] == index.shape[0]
        assert content.shape[0] == self.feat
        # print("Updating prototypes...")
        self.concept_proto[:, index] = content.clone()
        
    def _check_warmup_state(self):
        """check if need to switch warup_state to 0; when turn off warmup state, trigger k-means init for clustering"""
        # assert self._get_warmup_state(), "Calling _check_warmup_state when self.warmup_state is 0 (0 means not in warmup state)"

        if self.warmup_iter_counter == self.warmup_total_iter:
            # trigger kmean concept pool init
            self.pool_kmean_init() # 聚类

    def _get_warmup_state(self):
        '''
        目的是避免warmup的时候，因为warmup_iter_counter还没有被初始化，导致报错
        '''
        # print(f"NOW ---- self.warmup_iter_counter {self.warmup_iter_counter}")
        return self.warmup_iter_counter.cpu() <= self.warmup_total_iter
    
    #############################
    #     Helper  functions     #
    #############################

    def _get_cluster_num_index(self, idx):
        assert idx < self.total_pool_size
        return idx // self.pool_size_per_cluster

    def _get_cluster_ptr(self, cluster_num):
        """get starting pointer for cluster_num"""
        assert cluster_num < self.num_k, f"cluster_num {cluster_num} out of bound (totally has {self.num_k} clusters)"
        return self.pool_size_per_cluster * cluster_num