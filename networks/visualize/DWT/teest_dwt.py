import pywt
import numpy as np
import cv2
from matplotlib import pyplot as plt

# 读取图像
image = cv2.imread('in_pre.png', cv2.IMREAD_GRAYSCALE)

# 确保图像是二维的
if len(image.shape) == 3:
    image = image[:, :, 0]

# 执行DWT变换
coeffs = pywt.dwtn(image, 'haar')
# 二维小波分解
# coeffs = pywt.wavedec2(image, 'haar', level=1) # 一级

# 获取低频系数（近似分量）
LL = coeffs['aa']  # 第一个元素是LL

# 获取高频系数（细节分量）
LH = coeffs['ad']  # 第二个元素是LH
HL = coeffs['da']  # 第三个元素是HL
HH = coeffs['dd']  # 第四个元素是HH

# 可视化低频和高频系数
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(LL, cmap='gray'), plt.title('Approximation Coefficients (Low Frequency)')
plt.subplot(122), plt.imshow(LH, cmap='gray'), plt.title('Detail Coefficients (High Frequency)')
plt.show()
print('ok')