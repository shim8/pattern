#!/usr/bin/env python3
# -*- coding: utf-8 -*

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# データセットの読み込み
data = pd.read_csv("iris.csv").values.tolist()

rate = [] # 識別率を格納する配列

for k in range(1, 31): # kを1から30まで変化

    test = [] # テストデータの分類結果を格納する配列
    match = 0 # 正しく分類したデータ数

    for i in range(len(data)): # テストデータを変化

        iris_class = [0, 0, 0]
        distance = [[0 for n in range(2)] for m in range(150)] # テストデータと訓練データとのノルム

        for j in range(len(data)):
            distance[j][1] = j
            for l in range(4):
                distance[j][0] += (data[i][l] - data[j][l]) ** 2

        distance.sort() # ノルムを昇順にソート

        # ノルムの小さい順にk個のクラスを記録
        for j in range(1, k+1):
            if data[distance[j][1]][4] == 'Iris-setosa':
                iris_class[0] += 1
            elif data[distance[j][1]][4] == 'Iris-versicolor':
                iris_class[1] += 1
            elif data[distance[j][1]][4] == 'Iris-virginica':
                iris_class[2] += 1

        # 最頻クラスの判定
        if np.argmax(iris_class) == 0:
            test.append('Iris-setosa')
        elif np.argmax(iris_class) == 1:
            test.append('Iris-versicolor')
        elif np.argmax(iris_class) == 2:
            test.append('Iris-virginica')

        # 正誤判定
        if test[i] == data[i][4]:
            match += 1

    # 識別率の計算
    rate.append(match / len(data) * 1.0)

# x座標の定義
x = np.array(range(1, 31))

# グラフの表示
plt.figure()
plt.subplot(1, 1 ,1)
plt.plot(x, rate, marker='o', markersize=5, label='Identification rate')
plt.xlim(1, 30)
plt.ylim(min(rate)-0.01, max(rate)+0.01)
plt.legend()
plt.grid(True)
plt.show()
