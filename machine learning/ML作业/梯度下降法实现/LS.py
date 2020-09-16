# ML 第一次作业
# Author： 韩轶凡    Student ID： 1809401053
# 最小二乘法解回归问题 filename:LS.py Version: 1.0
# Lastest Edit:2020/03/08

import numpy as np
import random

# 读取input数据X
with open("x.txt", "r") as f:
    a = f.readlines()
    for i in range(len(a)):
        a[i] = a[i].split(" ")[0:10:]
        for j in range(len(a[i])):
            a[i][j] = float(a[i][j])

# 转换为矩阵格式
x = np.array(a)

# 读取output数据Y
with open("y.txt", "r") as f:
    b = [line for line in f]
for i in range(len(b)):
    b[i] = float( str( b[i][:-1:] ) )

# 转换为矩阵
y = np.array(b)

# 根据公式求出 w
Q = np.dot(x.T,x)
A = np.dot(np.linalg.inv(Q),x.T)
Z = np.dot(A,y)
print(Z)