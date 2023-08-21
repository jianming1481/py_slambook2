# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 21:33:42 2023

@author: Lui
"""

import numpy as np              # Matrix Computation
import matplotlib.pyplot as plt # Visualization
import math                     # Exponential
import random                   # Gaussian Noise

ar = 1.0
br = 2.0
cr = 1.0

ae = 2.0
be = -1.0
ce = 5.0
    
N = 100 # number of data
sigma = 1 # noise std
inv_sigma = 1/sigma

x_data = []
y_data = []

def target_func(a, b, c, x):
    return math.exp(a*x*x + b*x + c)+random.gauss(0, sigma*sigma)

for i in range(N):
    x = i/100
    x_data.append(x)
    y_data.append(target_func(ar, br, cr, x))
    
H = np.zeros((3, 3)) # Hessian J^T x W^{-1} x J in Gaussian-Newton
b = np.zeros(3) # bias

cost = 0

for i in range(N):
    xi = x_data[i]
    yi = y_data[i]
    error = yi - target_func(ae, be, ce, xi)
    J = np.zeros(3)
    J[0] = -xi*xi*math.exp(ae*xi*xi+be*xi+ce) # de/da
    J[1] = -xi*math.exp(ae*xi*xi+be*xi+ce)    # de/db
    J[2] = -1*math.exp(ae*xi*xi+be*xi+ce)     # de/dc
    
    H = H + inv_sigma*inv_sigma** J[:, None] @ J[None, :]
    b = b + -inv_sigma*inv_sigma*error*J
    cost = cost + error*error
    
    # Solve Linear Equation Hx=b
    dx = np.linalg.solve(H, b)

    ae = ae+dx[0]
    be = be+dx[1]
    ce = ce+dx[2]
    
print('ae:%f, be:%f, ce:%f'%(ae, be, ce))
# plt.plot(x_data, y_data)
# plt.show()

