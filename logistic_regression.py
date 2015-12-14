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


def main():
    # データを読み込む
    # numpyの配列に格納
    data = np.genfromtxt("../logistic_regression/ex2data1.txt", delimiter=",")
    data_x = data[:, (0, 1)]
    data_y = data[:, 2]

    # 訓練データをプロット
    # plt.figure(1)
    # plot_data(data_x, data_y)

    # データの個数を取得
    m = len(data_x)

    # 特徴量data_xに定数項を追加 .. x_0
    data_x = np.hstack((np.ones((m, 1)), data_x))

    # 訓練データを共有変数にする
    X = theano.shared(
        np.asarray(data_x, dtype=theano.config.floatX),
        name='X',
        borrow=True)
    y = theano.shared(
        np.asarray(data_y, dtype=theano.config.floatX),
        name='y',
        borrow=True)

    # パラメータを共有変数とする
    theta = theano.shared(
        np.zeros(3, dtype=theano.config.floatX),
        name='theta',
        borrow=True)

    # 仮説関数
    h = T.nnet.sigmoid(T.dot(X, theta))

    # コスト関数を定義する(sigmoid)
    cost = (1.0 / m) * T.sum(
        -y * T.log(h) - (1 - y) * T.log(1 - h)
    )

    # コスト関数の微分を定義
    g_cost = theano.grad(cost=cost, wrt=theta)

    # パラメータを更新
    learning_rate = 0.001
    updates = {theta: theta - learning_rate * g_cost}

    # 訓練用の関数を定義
    train_model = theano.function(
        inputs=[],
        outputs=cost,
        updates=updates
    )

    # 繰り返す
    cost_graph = []
    interations = 300000
    for iter in range(interations):
        current_cost = train_model()
        if iter % 100 == 0:
            cost_graph.append(current_cost)
            print iter, current_cost

    # thetaの値を更新
    t = theta.get_value()

    # 決定境界を表示
    plt.figure(1)
    xmin, xmax = min(cost_graph), max(cost_graph)
    xs = np.linspace(xmin, xmax, len(cost_graph))
    ys = [y * 100 for y in cost_graph]
    plt.plot(xs, ys, 'b-', label="cost graph")

    # plotを表示
    plt.xlabel("train times")
    plt.ylabel("cost")
    # plt.xlim((30, 100))
    # plt.ylim((30, 100))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
