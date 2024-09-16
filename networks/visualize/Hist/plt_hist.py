import torch
import matplotlib.pyplot as plt
cluster_affinity = torch.rand(256, 600)

# 直接绘制直方图
sorted_affinity, _ = torch.sort(cluster_affinity[15], descending=True)
# 绘制直方图
plt.hist(sorted_affinity.detach().cpu().numpy(), bins=100)
# 添加标签和标题
plt.title('Histogram of Sorted affinity')
plt.xlabel('Sorted affinity')
plt.ylabel('Frequency')
# 显示图形
plt.show()


# 通过计算
sorted_bins = 200
sorted_affinity, _ = torch.sort(cluster_affinity, dim=1, descending=False)  # 对每个特征的相似度进行降序排列（沿着列）
histograms = torch.stack([torch.histc(sorted_affinity[i], bins=sorted_bins, 
                                      min=sorted_affinity[i].min(), max=sorted_affinity[i].max())
                         for i in range(sorted_affinity.size(0))])  # 计算每个特征的相似度直方图
# 绘制直方图
# 创建x轴的值（区间的中心）
x = torch.linspace(sorted_affinity[10].min(), sorted_affinity[10].max(), len(histograms[10]))
y = histograms[10]
plt.bar(x, histograms[10].detach().cpu().numpy(), width=(x[1]-x[0]))
# 设置标题和轴标签
plt.title('Histogram of Sorted affinity')
plt.xlabel('Sorted affinity')
plt.ylabel('Frequency')
# 显示图形
plt.show()
# 累积
cumulative_histograms = histograms.cumsum(dim=1)  # 计算每行累积直方图（沿列）
# 计算相对频率
relative_frequency = cumulative_histograms / cumulative_histograms[:, -1].unsqueeze(1)  # 计算相对频率
# 绘制线图
# 创建x轴的值（区间的中心）
x = torch.linspace(sorted_affinity[10].min(), sorted_affinity[10].max(), len(relative_frequency[10]))
y = relative_frequency[10]
plt.plot(x, relative_frequency[10].detach().cpu().numpy())
# 设置标题和轴标签
plt.title('Relative Frequency')
plt.xlabel('Sorted affinity')
plt.ylabel('Relative Frequency')
# 显示图形
plt.show()

shrink_thres_percentage = 0.2  # 阈值的百分比
threshold_indices = (relative_frequency > shrink_thres_percentage).long().argmax(dim=1)  # 转为int64，并找到每行的阈值索引
threshold_index = threshold_indices * (sorted_affinity.size(1) // sorted_bins)
threshold = sorted_affinity.gather(1, threshold_index.unsqueeze(1))  # 根据阈值索引获取阈值

index = torch.where(cluster_affinity[15]>threshold)[0]


