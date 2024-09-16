from numpy import float32
import torch
import matplotlib.pyplot as plt

# 假设 your_tensor 是一个形状为 [44, 64, 56, 56] 的张量在gpu上参与梯度计算
your_tensor = torch.rand(44, 64, 56, 56)  # 示例张量，实际情况请用你的张量替换

# 核心
# img_numpy = img_tensor.detach().cpu().numpy().astype(float32) # 分离梯度放在cpu转为numpy

#%
# 法一：用tensor对起处理后再转换显示
img_tensor = your_tensor.mean(dim=1)[0]  # 这里假设要显示第一个样本(torch)
img_tensor = your_tensor[b, c] # [b][c]
img_numpy = img_tensor.detach().cpu().numpy().astype() # 分离梯度放在cpu转为numpy
# 法二：转换为numpy后再处理直接显示

# 使用 plt 显示图像
plt.imshow(img_numpy, cmap='hot') # 热力图camp='hot'
plt.colorbar()  # 可选，为图像添加颜色条以更好地理解像素值的分布
plt.show()
#%

