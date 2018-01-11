#! usr/bin/env python
# -*- coding: utf-8 -*-

import os
import struct
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
import matplotlib.pyplot as plt

def load_mnist(path, kind):
    """MNISTデータをpathからロード"""
    # 引数に指定したパスを結合（ラベルや画像のパスを作成）
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    # ファイルを読み込む：
    # 引数にファイル、モードを指定（rbは読み込みのバイナリモード）
    with open(labels_path, 'rb') as lbpath:
        # バイナリを文字列に変換：unpack関数の引数にフォーマット、8バイト分のバイナリデータを指定してマジックナンバー、アイテムの個数を読み込む
        magic, n = struct.unpack('>II', lbpath.read(8))
        # ファイルからラベルを読み込み配列を構築：fromfile関数の引数にファイル、配列のデータ形式を指定
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        # 画像ピクセル情報の配列サイズを変更
        # (行数：ラベルのサイズ、列数：特徴量の個数)
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = images / 255.0

    return images, labels

def calc_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-alpha * x))

def sigmoid(x):
    sig = np.where(x < -Threshold, 0, x)
    sig = np.where(sig != 0, calc_sigmoid(sig), sig)
    return sig

def dsigmoid(x):
    return alpha * sigmoid(x) * (1.0 - sigmoid(x))

def ReLU(x):
    return np.where(x > 0, x, 0)

def dReLU(x):
    return np.where(x > 0, 1.0, 0.0)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.power(tanh(x), 2)

def softmax(x):
    exp = np.exp(x)
    sums = np.sum(exp, axis=1)
    sums = sums[:, np.newaxis]
    return exp / sums

def init_w(x):
    w = np.random.randn(len(x), len(x[0])) / np.sqrt(len(x))
    return w

def update_w(w, x, delta):
    return w - eta * np.dot(x.T, delta) / len(x)

def add_bias(x):
    return np.hstack((np.ones((len(x), 1)), x))

def part_calc(x_in, w, func):
    x_in = add_bias(x_in)
    u = np.dot(x_in, w)
    x_out = func(u)
    return x_in, u, x_out

def calc_delta_mid(w, delta, u, dfunc):
    w = np.delete(w, 0, 0)
    w_t = w.T
    return np.dot(delta, w_t) * dfunc(u)

def calc_delta_out(i, x_out, y_train, u, dfunc):
    y_train = y_train[batch*i:batch*(i+1)]
    y_hat = softmax(x_out)
    return (y_hat - y_train) * dfunc(u)

def train(i, x_train, y_train, w_mid1, w_mid2, w_out, func, dfunc):
    x_in = x_train[batch*i:batch*(i+1)]
    x_in, u_mid1, x_mid1 = part_calc(x_in, w_mid1, func)
    x_mid1, u_mid2, x_mid2 = part_calc(x_mid1, w_mid2, func)
    x_mid2, u_out, x_out = part_calc(x_mid2, w_out, func)
    delta_out = calc_delta_out(i, x_out, y_train, u_out, dfunc)
    delta_mid2 = calc_delta_mid(w_out, delta_out, u_mid2, dfunc)
    delta_mid1 = calc_delta_mid(w_mid2, delta_mid2, u_mid1, dfunc)
    w_mid1_new = update_w(w_mid1, x_in, delta_mid1)
    w_mid2_new = update_w(w_mid2, x_mid1, delta_mid2)
    w_out_new = update_w(w_out, x_mid2, delta_out)
    return w_mid1_new, w_mid2_new, w_out_new

def test(x_test, w_mid1, w_mid2, w_out, func):
    x_in, u_mid1, x_mid1 = part_calc(x_test, w_mid1, func)
    x_mid1, u_mid2, x_mid2 = part_calc(x_mid1, w_mid2, func)
    x_mid2, u_out, x_out = part_calc(x_mid2, w_out, func)
    return x_out

def judge(i):
    if label[i] == labels_test[i]:
        return 1
    else:
        return 0

images_train, labels_train = load_mnist(path='mnist', kind='train')
images_test, labels_test = load_mnist(path='mnist', kind='t10k')

eta = 0.01
alpha = 4.0
num_mid1 = 100
num_mid2 = 50
batch = 100
count = 0
finish = 50
Threshold = 500.0 / alpha
w_mid1 = np.empty((785, num_mid1))
w_mid2 = np.empty((num_mid1+1, num_mid2))
w_out = np.empty((num_mid2+1, 10))
rate = []

y_train = np.zeros((len(labels_train), 10))
for i in range(len(labels_train)):
    label = labels_train[i]
    y_train[i][label] = 1

w_mid1 = init_w(w_mid1)
w_mid2 = init_w(w_mid2)
w_out = init_w(w_out)

while True:
    for i in range(len(labels_train) / batch):
        w_mid1, w_mid2, w_out = train(i, images_train, y_train, w_mid1, w_mid2, w_out, sigmoid, dsigmoid)

    y_hat = test(images_test, w_mid1, w_mid2, w_out, sigmoid)
    label = np.argmax(y_hat, axis=1)
    p = Pool(mp.cpu_count())
    match = p.map(judge, np.arange(len(label)))
    p.close()
    r = 100.0 * sum(match) / len(match)
    rate = np.append(rate, r)

    count += 1
    print("{0} / {1} epochs".format(count, finish))

    if count == finish:
        break

epoch = np.array(range(1, finish+1))
plt.figure()
plt.plot(epoch, rate)
plt.xlabel("epochs [times]")
plt.ylabel("accuracy [%]")
plt.xlim(1, finish)
plt.ylim(0, 100)
plt.grid(True)
plt.show()
