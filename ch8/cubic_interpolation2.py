import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 給定已知數據點
x_data = np.array([0, 1, 2, 3])
y_data = np.array([1, 2, 0, 1])

# 使用scipy的CubicSpline進行三次插值
cs = CubicSpline(x_data, y_data)

# 計算插值結果
x_new = np.linspace(0, 3, 100)
y_new = cs(x_new)

# 繪製圖表
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='red', label='Data Points')
plt.plot(x_new, y_new, label='Cubic Spline Interpolation')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('1D Cubic Interpolation Example')
plt.grid(True)
plt.show()