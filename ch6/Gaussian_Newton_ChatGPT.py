# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 23:39:08 2023

@author: Lui
"""
import numpy as np
import time
import matplotlib.pyplot as plt

# 初始化參數
ar, br, cr = 1.0, 2.0, 1.0  # 真實參數值
ae, be, ce = 2.0, 3.0, 0.0  # 估計參數值
N = 100  # 數據點
w_sigma = 1.0  # 噪聲Sigma值
inv_sigma = 1.0 / w_sigma

# 生成數據
np.random.seed(0)
x_data = np.array([i/100.0 for i in range(N)])
y_data = np.exp(ar * x_data**2 + br * x_data + cr) + np.random.normal(0, w_sigma, N)

# 開始Gauss-Newton迭代
iterations = 100  # 迭代次數
cost, lastCost = 0, 0  # 本次迭代的cost和上一次迭代的cost

t1 = time.time()
for iter in range(iterations):
    H = np.zeros((3, 3))
    b = np.zeros(3)
    cost = 0
    
    for i in range(N):
        xi, yi = x_data[i], y_data[i]
        error = yi - np.exp(ae * xi**2 + be * xi + ce)
        J = np.array([
            -xi**2 * np.exp(ae * xi**2 + be * xi + ce),
            -xi * np.exp(ae * xi**2 + be * xi + ce),
            -np.exp(ae * xi**2 + be * xi + ce)
        ])
        
        H += inv_sigma * inv_sigma * J[:, None] @ J[None, :]
        b += -inv_sigma * inv_sigma * error * J
        cost += error**2

    # 求解線性方程
    dx = np.linalg.solve(H, b)
    
    if np.isnan(dx).any():
        print("result is nan!")
        break
    
    if iter > 0 and cost >= lastCost:
        print(f"cost: {cost} >= last cost: {lastCost}, break.")
        break

    ae, be, ce = ae+dx[0], be+dx[1], ce+dx[2]
    lastCost = cost

    print(f"total cost: {cost}, update: {dx}, estimated params: {ae}, {be}, {ce}")

t2 = time.time()
print(f"solve time cost = {t2 - t1} seconds.")
print(f"estimated abc = {ae}, {be}, {ce}")

# for i in range(len(x_data)):
#     xi = x_data[i]
#     y_est = np.exp(ae*xi*xi + be*xi +c)
y_est = np.exp(ae * x_data**2 + be * x_data + ce)

plt.plot(x_data, y_data)
plt.plot(x_data, y_est)
plt.show()