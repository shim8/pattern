#!usr/bin/env python
# -*- coding: utf-8 -*-

# モジュールのインポート
import pandas as pd
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# データセットの読み込み
data = pd.read_csv("Auto_MPG_DataSet.csv").values.tolist()

mpg_train = 0
cylinders_train = 1
displacement_train = 2
horsepower_train = 3
weight_train = 4
acceleration_train = 5
modelyear_train = 6
origin_train = 7
carname_train = 8

#説明変数の定義
explanatory1 = [[0 for n in range(3)] for m in range(len(data))]
explanatory2 = [[0 for n in range(5)] for m in range(len(data))]
mpg = []
for i in range(len(data)):
    mpg.append(data[i][mpg_train])
    explanatory1[i][0] = 1
    explanatory1[i][1] = data[i][horsepower_train]
    explanatory1[i][2] = data[i][weight_train]
    explanatory2[i][0] = 1
    explanatory2[i][1] = data[i][displacement_train]
    explanatory2[i][2] = data[i][horsepower_train]
    explanatory2[i][3] = data[i][weight_train]
    explanatory2[i][4] = data[i][acceleration_train]

#擬似逆行列の計算
pinv1 = np.linalg.pinv(explanatory1)
pinv2 = np.linalg.pinv(explanatory2)
#wの算出
w1 = np.dot(pinv1, mpg)
w2 = np.dot(pinv2, mpg)

#可視化のための配列定義
displacement = []
horsepower = []
weight = []
acceleration = []
for i in range(len(data)):
    displacement.append(data[i][displacement_train])
    horsepower.append(data[i][horsepower_train])
    weight.append(data[i][weight_train])
    acceleration.append(data[i][acceleration_train])

#平面の3D可視化
num = 20
W1_0 = [[w1[0] for n in range(num)] for m in range(num)]
W2_0 = [[w2[0] for n in range(num)] for m in range(num)]
x = np.linspace(np.amin(horsepower), np.amax(horsepower), num)
y = np.linspace(np.amin(weight), np.amax(weight), num)
X, Y = np.meshgrid(x, y)
Z1 = W1_0 + w1[1] * X + w1[2] * Y
p = np.linspace(np.amin(displacement), np.amax(displacement), num)
q = np.linspace(np.amin(acceleration), np.amax(acceleration), num)
P, Q = np.meshgrid(p, q)
Z2_xy = w2[2] * X + w2[3] * Y
Z2_pq = w2[1] * P + w2[4] * Q
ave = np.average(Z2_pq)
Z2_pq_ave = [[ave for n in range(num)] for m in range(num)]
Z2 = W2_0 + Z2_xy + Z2_pq_ave

#グラフの描写
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(horsepower, weight, mpg)
#ax.plot_wireframe(X, Y, Z1)
ax.plot_wireframe(X, Y, Z2)
ax.set_xlabel('horsepower')
ax.set_ylabel('weight')
ax.set_zlabel('mpg')
ax.grid(True)
plt.show()
