#!/usr/bin/env python
# -*- coding: utf-8 -*

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# データセットの読み込み
data = pd.read_csv("iris.csv").values.tolist()
# クラスの初期化
for i in range(len(data)):
  data[i][4] = 0

# 変数の定義
Threshold = 10 ** -30 # 閾値
X = 0 # X軸のパラメータ
Y = 3 # Y軸のパラメータ

# 距離を返す関数の定義
def d(x, y):
  return (x - y) **2

# クラスタの数を入力
print('Input the number of cluster. (2 to 5)')
k = input('>> ')
k = int(k)

centroid = [[0 for n in range(4)] for m in range(k)] # セントロイドを格納する配列

# セントロイドの初期値を乱数で与える
for j in range(k):
    for l in range(4):
      centroid[j][l] = random.uniform(np.amin(data, axis=0)[l], np.amax(data, axis=0)[l])

# セントロイドが収束するまでループ
while True:
  distance = [[0 for n in range(k)] for m in range(len(data))] # ノルムを格納する配列
  SUM = [[0 for n in range(5)] for m in range(k)] # クラスごとの各パラメータの合計
  delta_centroid = [0 for n in range(k)] # 新しいセントロイドと古いセントロイドとのノルム
  
  for i in range(len(data)):
    for j in range(k):
      for l in range(4):
        distance[i][j] += d(data[i][l], centroid[j][l])
    data[i][4] = np.argmin(distance, axis=1)[i]
    SUM[data[i][4]][4] += 1
    for l in range(4):
      SUM[data[i][4]][l] += data[i][l]

  hantei = 0 # 要素数ゼロのクラスがあるか否かの判定変数
  
  for j in range(k):
    if SUM[j][4] == 0:
      hantei = 1
      break
    
  if hantei == 0:
    for j in range(k):
      for l in range(4):
        delta_centroid[j] += d(SUM[j][l]/SUM[j][4], centroid[j][l])

    for j in range(k):
      for l in range(4):
        centroid[j][l] = SUM[j][l] / SUM[j][4]
    
    if max(delta_centroid) < Threshold:
      break
    
  elif hantei == 1:
    for j in range(k):
      for l in range(4):
        centroid[j][l] = random.uniform(np.amin(data, axis=0)[l], np.amax(data, axis=0)[l])

# 散布図の描写
data0_x = []
data0_y = []
data1_x = []
data1_y = []
if k >= 3:
  data2_x = []
  data2_y = []
if k >= 4:
  data3_x = []
  data3_y = []
if k >= 5:
  data4_x = []
  data4_y = []

for i in range(len(data)):
  if data[i][4] == 0:
    data0_x.append(data[i][X])
    data0_y.append(data[i][Y])
  elif data[i][4] == 1:
    data1_x.append(data[i][X])
    data1_y.append(data[i][Y])
  elif data[i][4] == 2:
    data2_x.append(data[i][X])
    data2_y.append(data[i][Y])
  elif data[i][4] == 3:
    data3_x.append(data[i][X])
    data3_y.append(data[i][Y])
  elif data[i][4] == 4:
    data4_x.append(data[i][X])
    data4_y.append(data[i][Y])

plt.scatter(data0_x, data0_y, c='blue', label='Class1')
plt.scatter(centroid[0][X], centroid[0][Y], s=150, c='blue', marker='*', label='Centroid of Class1')
plt.scatter(data1_x, data1_y, c='red', label='Class2')
plt.scatter(centroid[1][X], centroid[1][Y], s=150, c='red', marker='*', label='Centroid of Class2')
if k >= 3:
  plt.scatter(data2_x, data2_y, c='yellow', label='Class3')
  plt.scatter(centroid[2][X], centroid[2][Y], s=150, c='yellow', marker='*', label='Centroid of Class3')
if k >= 4:
  plt.scatter(data3_x, data3_y, c='green', label='Class4')
  plt.scatter(centroid[3][X], centroid[3][Y], s=150, c='green', marker='*', label='Centroid of Class4')
if k >= 5:
  plt.scatter(data4_x, data4_y, c='purple', label='Class5')
  plt.scatter(centroid[4][X], centroid[4][Y], s=150, c='purple', marker='*', label='Centroid of Class4')

plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend(loc='lower right')
plt.show()
