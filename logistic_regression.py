#coding: utf-8
import os
import sys
import time
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt


def plot_data(X, y):
    """ グラフ表示関数 """
    # positiveクラスのデータのインデックス
    positive = [i for i in range(len(y)) if y[i] == 1]
    # negativeクラスのデータのインデックス
    negative = [i for i in range(len(y)) if y[i] == 0]

    plt.scatter(
        X[positive, 0], X[positive, 1],
        c='red', marker='o', label='positive'
    )
    plt.scatter(
        X[negative, 0], X[negative, 1],
        c='blue', marker='o', label='negative'
    )


# データを読み込む
# numpyの配列に格納
data = np.genfromtxt("../logistic_regression/ex2data1.txt", delimiter=",")
data_x = data[:, (0, 1)]
data_y = data[:, 2]

# 訓練データをプロット
plt.figure(1)
plot_data(data_x, data_y)


# コスト関数を定義する(sigmoid)
# コスト関数の微分を定義
# 勾配降下法
# パラメータを更新
# 訓練用の関数を定義
# 繰り返す
# 決定境界を表示

# plotを表示
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim((30, 100))
plt.ylim((30, 100))
plt.legend()
plt.show()
