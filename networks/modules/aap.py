from cProfile import label
import numpy as np
# from sklearn.datasets.samples_generator import make_blobs
import matplotlib.cm as cm # 颜色
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

def cplot(datas,labels,str_title=""):
    plt.cla()
    index_center = np.unique(labels).tolist()

    colors={}
    # 使用 ScalarMappable 来获取颜色
    cmap = cm.ScalarMappable(cmap='Spectral')  # 获取颜色映射
    for i, each in zip(index_center, np.linspace(0, 1,len(index_center))):
        colors[i] = cmap.to_rgba(each)

    N,D = np.shape(datas)
    for i in range(N):
        i_center = labels[i]
        center = datas[i_center]
        data = datas[i]
        
        color = colors[i_center]
        plt.plot([center[0],data[0]],[center[1],data[1]],color=color)
    plt.title(str_title)

class AdaptAffinityPropagation:
    def __init__(self, 
                 maxiter=200, iteration=5, # 200
                 dampfac=0.75, 
                 pmin=-50, pmax=50):
        self.maxiter = maxiter
        self.iteration = iteration
        self.dampfac = dampfac
        self.dampfac_init = 0.7
        self.dampfac_inc = 0.02
        self.dampfac_dec = 0.02
        self.message_thresh = 1e-5
        self.local_thresh = 10
        self.pmin = pmin
        self.pmax = pmax
        self.Cn_history = []
        
    def compute_S_init(self, datas, preference):
        # ... 实现 compute_S_init 函数 ...
        N,D = np.shape(datas)
        tile_x = np.tile(np.expand_dims(datas,1),[1,N,1]) # N, N,D
        tile_y = np.tile(np.expand_dims(datas,0),[N,1,1]) # N, N,D
        S = -np.sum((tile_x-tile_y)**2, axis=-1)
        if type(preference) == np.ndarray:
            m = preference # m.shape = (N,)
        elif isinstance(preference, (int, float)):  # 检查 preference 是否为整数或浮点数
            m = preference
        else:
            # 为了加速
            indices = np.where(~np.eye(S.shape[0],dtype=bool))
            if preference == "median":
                m = np.median(S[indices])
            elif preference == "min":
                m = np.min(S[indices])
        np.fill_diagonal(S, m)
        return S
    
    # def compute_R(self, S, R, A, dampfac):
    #     # ... 实现 compute_R 函数 ...
    #     # to_max = A + S
    #     to_max = A + R
    #     N = np.shape(to_max)[0]        
    #     max_AS = np.zeros_like(S)
    #     for i in range(N):
    #         for k in range(N):
    #             if not i ==k:
    #                 temp = to_max[i,:].copy()
    #                 temp[k] = -np.inf
    #                 max_AS[i,k] = max(temp)
    #             else:
    #                 temp = S[i,:].copy()
    #                 temp[k] = -np.inf
    #                 max_AS[i,k] = max(temp)

    #     return (1-dampfac) * (S - max_AS) + dampfac * R
    
    def compute_R(self, S, R, A, dampfac):
        # to_max = A + S
        to_max = A + R
        N = to_max.shape[0]
        # 使用 NumPy 的广播和索引特性来更新 max_AS
        # 将对角线元素设置为 -np.inf，以便在求 max 时排除自身
        np.fill_diagonal(to_max, -np.inf)
        # 计算每一行的最大值，得到 max_AS，排除了对角线上的 -np.inf
        max_AS = np.max(to_max, axis=1, keepdims=True)
        # 恢复对角线上的值，使其等于 S 的对角线值
        np.fill_diagonal(max_AS, np.diag(S))
        # 计算 R 的更新值
        R_updated = (1 - dampfac) * (S - max_AS) + dampfac * R
        return R_updated

    # def compute_A(self, R, A, dampfac):
    #     # ... 实现 compute_A 函数 ...
    #     max_R = np.zeros_like(R)
    #     N = np.shape(max_R)[0]
    #     for i in range(N):
    #         for k in range(N):
    #             max_R[i,k] = np.max([0,R[i,k]])
    #     min_A = np.zeros_like(A)
    #     for i in range(N):
    #         for k in range(N):
    #             if not i == k:
    #                 temp = max_R[:,k].copy()
    #                 temp[i] =0
    #                 min_A[i,k] = np.min([0,R[k,k]+np.sum(temp)])
    #             else:
    #                 temp = max_R[:,k].copy()
    #                 temp[k] =0
    #                 min_A[i,k] = np.sum(temp)
    #     return (1-dampfac)*min_A + dampfac*A
    
    def compute_A(self, R, A, dampfac):
        N = R.shape[0] # 获取矩阵的大小
        # 初始化 max_R，仅保留 R 中非负值
        max_R = np.maximum(0, R)
        # 计算每一列的和，存储到 min_A 中，跳过对角线元素
        # 使用广播将 R[k, k] 扩展到合适大小并计算
        min_A = np.zeros_like(A)
        for k in range(N):
            col_sum = np.sum(max_R[:, k])
            min_A[:, k] = np.where(R[k, k] + col_sum > 0, 0, R[k, k] + col_sum)
        # 将对角线元素设置为 max_R 对角线元素的和
        np.fill_diagonal(min_A, np.diag(R) + np.sum(max_R, axis=0))
        # 应用阻尼因子 dampfac 更新 A
        A_updated = (1 - dampfac) * min_A + dampfac * A
        return A_updated

    # def update_p(self, pn, Ct, Cn):
    #     # ... 实现 update_p 函数 ...
    #     """
    #     根据当前聚类数 Cn 和目标聚类数 Ct 动态更新偏好参数 p。
    #     :param pn: 当前的偏好参数 p
    #     :param Ct: 目标聚类数
    #     :param Cn: 当前迭代得到的聚类数
    #     :return: 更新后的偏好参数 p
    #     """
    #     ps = pn # 首端
    #     pe = pn # 末端
    #     if Cn > Ct:
    #         # 聚类数大于目标聚类数(P太大，太多中心)
    #         pe = pn # 调小p,使得每个点都聚类到同一个簇
    #         ps = (pe + self.pmin) / 2.0
    #     elif Cn < Ct:
    #         # 聚类数小于目标聚类数
    #         ps = pn # 调大p,使得每个点都聚类到同一个簇
    #         pe = (pn + self.pmax) / 2.0
    #     pn_new = (pe + ps) / 2.0 
    #     pn_new = max(self.pmin, min(pn_new, self.pmax))  # 确保 pn_next 在 [pmin, pmax] 范围内
    #     return pn_new
    
    def update_p(self, pn, Ct, Cn):
        # ... 实现 update_p 函数 ...
        """
        根据当前聚类数 Cn 和目标聚类数 Ct 动态更新偏好参数 p。
        :param pn: 当前的偏好参数 p
        :param Ct: 目标聚类数
        :param Cn: 当前迭代得到的聚类数
        :return: 更新后的偏好参数 p
        """
        # 默认
        ps = pn # 首端
        pe = pn # 末端
        # 假设 self.Cn_history 保存了之前迭代的 Cn 值
        self.Cn_history.append(Cn)
        # 限制历史长度，比如保持最近的10个值
        if len(self.Cn_history) > 10:
            self.Cn_history.pop(0)
        # 计算移动平均值
        moving_average = sum(self.Cn_history) / len(self.Cn_history)
        # 计算波动度
        volatility = abs(Cn - moving_average) / (moving_average+1e-3) 
        if Cn > Ct:
            # 聚类数大于目标聚类数(P太大，太多中心)
            pe = pn # 调小p,使得每个点都聚类到同一个簇
            ps = (pe + self.pmin) / 2.0
        elif Cn < Ct:
            # 聚类数小于目标聚类数
            ps = pn # 调大p,使得每个点都聚类到同一个簇
            pe = (ps + self.pmax) / 2.0
        change = (pe + ps) / 2.0 - pn
        pn_new = (pn + volatility * change) if volatility != 0.0 else (pn + change) # 指数平滑更新 
        pn_new = max(self.pmin, min(pn_new, self.pmax))  # 确保 pn_next 在 [pmin, pmax] 范围内
        return pn_new

    def affinity_prop(self, datas, target=None, display=False):
        cluster_count_cur = None 
        cluster_count_prev = None
        local_thresh = 10 # 判断聚类结果是否多轮不变
        message_thresh = 1e-5 # 判断更新前后 R+A是否有显著变化  
        if target is None:
            # 自动聚类
            preference='median'
        else:
            # 指定聚类
            print(f"affinity_prop target: {target}")
            preference = (self.pmin + self.pmax) / 2.0  # 初始 p 参数
        print("first")
        S = self.compute_S_init(datas, preference) # 计算S
        S = S+1e-12*np.random.normal(size=S.shape) * (np.max(S)-np.min(S)) # 避免S为0，归一化
        ## 加上较小的值防止震荡
        # A 和 R 的初始化
        A = np.zeros_like(S) # 可用矩阵
        R = np.zeros_like(S) # 责任矩阵

        i = 0
        converged = False
        converged_count = 0
        while i < self.maxiter:
            print("start iteration %d ..." %i)
            # 第一次不计入
            if i > 0:
                E_old = R+A # R+A 代表聚类中心数量
                labels_old = np.argmax(E_old, axis=1) # 含义是每个样本对应的类别
                cluster_count_prev = len(np.unique(labels_old)) # 更新前一次聚类数
            # 更新 R 和 A
            print("updating R and A in iteration %d")
            R = self.compute_R(S,R,A,self.dampfac)
            A = self.compute_A(R,A,self.dampfac)
            print("updated R and A in iteration %d")
            E_new = R+A
            labels_cur = np.argmax(E_new, axis=1) # 计算当前的聚类中心数量
            cluster_count_cur = len(np.unique(labels_cur)) # 计算当前的聚类中心数量
            
            # 判断是否收敛 => 更新前后 label 是否一致
            if i > 0:
                if np.all(labels_cur == labels_old):
                    converged_count += 1
                else:
                    converged_count = 0
                if (message_thresh != 0 and np.allclose(E_old, E_new, atol=message_thresh)) or \
                   (local_thresh != 0 and converged_count > local_thresh):
                    # np.allclose 比较两个数组是否在某个容忍度内完全相等
                    converged = True # 收敛
                    print("第 %d 轮后收敛."%(i))
                else:
                    converged = False # 未收敛
            
            if target is None:
                if converged:
                    # 自动聚类=>收敛直接返回
                    print("第 %d 轮迭代okokokok" %(i))
                    break
                else:
                    # 如果是第一次迭代，不进行震荡检测
                    if cluster_count_prev is not None:
                        # 计算聚类数的增长方向 dir
                        dir = cluster_count_cur - cluster_count_prev
                        if dir > 0:
                            # 如果 dir > 0，增加阻尼因子以消除震荡
                            self.dampfac += self.dampfac_inc
                            self.dampfac = min(self.dampfac, 0.9)  # 确保阻尼因子不会过大
                        elif dir < 0:
                            # 如果 dir < 0，减小阻尼因子以加速收敛
                            self.dampfac -= self.dampfac_dec
                            self.dampfac = max(self.dampfac, 0.5)  # 确保阻尼因子不会过小
                    print("dampfac:%f" %(self.dampfac))
                    print("第 %d 轮迭代结束" %(i))
            else:
                # 指定聚类
                # 大小周期判断
                if i % self.iteration == 0:
                    # 大周期判断
                    if converged:
                        # 收敛=>重置阻尼因子
                        dampfac = self.dampfac_init # dampfac不用记忆，通过重置dampfac_init为最小值来加速算法。
                        if target == cluster_count_cur:
                            # 聚类结果与目标一致
                            print("第 %d 轮迭代okokokok" %(i))
                            break
                        else:
                            # 不一致
                            preference = self.update_p(preference, 
                                                       target, cluster_count_cur)
                            print("big iter preference:%f" %(preference))
                            S = self.compute_S_init(datas, preference) # 计算S
                            S = S+1e-12*np.random.normal(size=S.shape) * (np.max(S)-np.min(S)) # 避免S为0，归一化
                            ## 加上较小的值防止震荡
                    # 震荡检测
                    # 如果是第一次迭代，不进行震荡检测
                    if cluster_count_prev is not None:
                        # 计算聚类数的增长方向 dir
                        dir = (target - cluster_count_cur + 1e-3) * \
                              (cluster_count_cur - cluster_count_prev + 1e-3)
                        if dir > 0:
                            # 如果 dir > 0，代表同向在衰减了，增加阻尼因子以消除震荡
                            self.dampfac += self.dampfac_inc
                            self.dampfac = min(self.dampfac, 0.9)  # 确保阻尼因子不会过大
                        elif dir < 0:
                            # 如果 dir < 0，减小阻尼因子以加速收敛
                            self.dampfac -= self.dampfac_dec
                            self.dampfac = max(self.dampfac, 0.5)  # 确保阻尼因子不会过小
                        print("big iter dampfac:%f" %(self.dampfac))
                else:
                    # 小周期判断(加速)
                    if converged:
                        # 收敛=>重置阻尼因子
                        self.dampfac = self.dampfac_init
                    else:
                        # 如果当前聚类数与目标聚类数不符，更新 preference
                        if target != cluster_count_cur:
                            preference = self.update_p(preference, 
                                                       target, cluster_count_cur)
                            print("small iter preference:%f" %(preference))
                            S = self.compute_S_init(datas, preference) # 计算S
                            S = S+1e-12*np.random.normal(size=S.shape) * (np.max(S)-np.min(S)) # 避免S为0，归一化
                            ## 加上较小的值防止震荡
            i = i+1
            print("第 %d 轮迭代结束"%(i))

            if display:
                plt.ion()
                E = R+A # Pseudomarginals
                labels = np.argmax(E, axis=1)
                N_cluster = len(np.unique(labels).tolist())
                str_title = 'epoch %d N_cluster%d'%(i,N_cluster)
                cplot(datas,labels,str_title=str_title)
                plt.pause(0.1)
                plt.ioff()


        E = R+A # Pseudomarginals E是最终聚类中心数量
        labels = np.argmax(E, axis=1) # argmax代表
        # 根据索引对唯一值进行排序然后再转为list
        exemplars = np.unique(labels) # 聚类中心 unique 是返回唯一值的array
        sorted_exemplars = exemplars[np.argsort(exemplars)] # 从小到大排序
        sorted_exemplars_list = sorted_exemplars.tolist()
        # 创建一个映射，将每个唯一标签映射到一个从0开始的连续整数
        label_mapping = {original_label: new_label 
                         for new_label, original_label in enumerate(sorted_exemplars_list)}
        maps = np.array([label_mapping[label.item()] for label in labels])
        centers = datas[exemplars]

        return labels, exemplars, maps, centers


if __name__ == "__main__":
    ap = AdaptAffinityPropagation()
    a = np.random.multivariate_normal([3,3], [[.5,0],[0,.5]],50)
    b = np.random.multivariate_normal([0,0], [[0.5,0],[0,0.5]], 50)
    c = np.random.multivariate_normal([3,0], [[0.5,0],[0,0.5]], 50)
    d = np.random.multivariate_normal([0,3], [[0.5,0],[0,0.5]], 50)
    data = np.r_[a,b,c,d]
    labels, exemplars, maps, centers = ap.affinity_prop(data,target=5,display=True)
    print(labels)
  
    cplot(data,labels)
    plt.show()


