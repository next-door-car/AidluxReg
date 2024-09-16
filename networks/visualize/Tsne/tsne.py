import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 假设 X 是形状为 (n, c, h, w) 的张量
# 首先将其展平为形状为 (n, c*h*w) 的矩阵
x = x.reshape(-1, -1)

# 应用t-SNE
tsne = TSNE(n_components=2, random_state=0)
## n_components 将维的维度
x_tsne = tsne.fit_transform(x)

# 可视化
plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
plt.show()


from sklearn.cluster import AffinityPropagation
ap = AffinityPropagation(damping=0.9, preference=-20.0, random_state=0).fit(x) # 拟合数据并预测聚类标签
predict_label = ap.predict(x)
# 应用t-SNE
tsne = TSNE(n_components=2, random_state=0)
## n_components 将维的维度
x_tsne = tsne.fit_transform(x)
plt.scatter(x_tsne[:,0],x_tsne[:,1],c=predict_label,cmap="viridis")
plt.colorbar()
plt.show()