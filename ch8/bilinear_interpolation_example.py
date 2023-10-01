import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, RectBivariateSpline

def my_bilinear_interpolation(data, x, y):
    (xLimit, yLimit) = data.shape
    if x==np.floor(x) and y==np.floor(y):
        return data[int(y)][int(x)]

    xLow = int(np.floor(x))
    yLow = int(np.floor(y))
    xHight = min(xLimit-1, int(xLow+1))
    yHight = min(yLimit-1, int(yLow+1))

    return (1+yLow-y)*(1+xLow-x)*data[yLow][xLow] \
        + (1+yLow-y)*(x-xLow)*data[yLow][xHight] \
        + (1-x+xLow)*(y-yLow)*data[yHight][xLow] \
        + (x-xLow)*(y-yLow)*data[yHight][xHight]

def my_bicubic_interpolation(data, x, y):
    return 0

# 創建一個簡單的2D數據
x = np.arange(0, 4)
y = np.arange(0, 4)
data = np.array([[1, 2, 1, 3],
                 [2, 4, 3, 1],
                 [3, 1, 2, 2],
                 [1, 3, 4, 4]])

# 定義插值函數 - 雙線性插值
interp_linear = interp2d(x, y, data, kind='linear')
# 定義插值函數 - 雙立方插值
# interp_cubic = interp2d(x, y, data, kind='cubic')
interp_cubic = RectBivariateSpline(x, y, data)

# 定義要進行插值的新坐標
new_x = np.arange(0, 3.5, 0.5)
new_y = np.arange(0, 3.5, 0.5)

# 使用雙線性插值和雙立方插值獲取新數據
new_data_linear = interp_linear(new_x, new_y)
new_data_cubic = interp_cubic(new_x, new_y)

# my BiLinear Interpolation Method
lui_new_data = np.zeros([len(new_x), len(new_y)])
for j in range(len(new_y)):
    for i in range(len(new_x)):
        lui_new_data[j][i] = my_bilinear_interpolation(data, new_x[i], new_y[j])

# my Bicubic Interpolation Method
lui_new_data = np.zeros([len(new_x), len(new_y)])
for j in range(len(new_y)):
    for i in range(len(new_x)):
        lui_new_data[j][i] = my_bicubic_interpolation(data, new_x[i], new_y[j])

compare_answer = np.zeros([len(new_x), len(new_y)])
for j in range(len(new_y)):
    for i in range(len(new_x)):
        if lui_new_data[j][i] == new_data_linear[j][i]:
            compare_answer[j][i] = True
        else:
            compare_answer[j][i] = False
print(compare_answer)

# 繪製原始數據和插值結果
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(data, cmap='viridis', origin='upper', extent=(0, 4, 0, 4))
plt.title('Original')
plt.subplot(1, 3, 2)
plt.imshow(new_data_linear, cmap='viridis', origin='upper', extent=(0, 3, 0, 3))
plt.title('Bilinear')
plt.subplot(1, 3, 3)
# plt.imshow(new_data_cubic, cmap='viridis', origin='upper', extent=(0, 3, 0, 3))
plt.imshow(lui_new_data, cmap='viridis', origin='upper', extent=(0, 3, 0, 3))
plt.title('Cubilinear')
plt.tight_layout()
plt.show()
