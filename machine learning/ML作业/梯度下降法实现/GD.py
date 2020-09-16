# ML 第一次作业
# Author： 韩轶凡    Student ID： 1809401053
# 梯度下降法解回归问题 filename:GD.py Version: 5.0
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

# 初始化W矩阵
w = [ random.randint(-10,10) for i in range(10) ]
w = np.array(w)

# 设置步长，初始化b矩阵
lamb = 0.0001
B = [1 for i in range(300)]


# 损失函数计算
def loss_function(X, Y, W, b):
    sum = 0
    re = Y-b-np.dot(X,W)
    for i in re:
        sum += i**2
    return sum


# 参数w的更新项计算
def gradientW(X, Y, W, b):
    delta = np.dot(X, W)+b-Y
    return np.dot(delta, X)


# 参数b的更新项计算
def gradientB(X, Y, W, b):
    return Y-b-np.dot(X, W)


# 梯度下降法主循环
iter_number = 0
while loss_function(x,y,w,B) > 0.001:
    deltaW = (lamb) * gradientW(x, y, w, B)
    deltaB = (lamb) * gradientB(x, y, w, B)
    w = w - deltaW
    B = B - deltaB
    iter_number += 1
    if iter_number % 10000 == 0:
        print("第"+str(iter_number)+"次迭代结果为")
        print(w)
else:
    print(w)




