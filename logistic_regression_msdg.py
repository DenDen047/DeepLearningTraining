#coding: utf-8
import os
import sys
import time
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

# 正規化項を含まない場合
#     interations = 300000
#     final cost = 0.2837
# 正規化項を含む場合
#     interations = 300000
#     learning_rate = 0.001
#         = 0.1 だと振動
#     final cost = 0.29642

# 最善なthetaを記録
# theta: [-0.01317437 -0.3742542  -0.40990937  0.00077361  0.01071652  0.00170271]


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


def mapFeature(x1, x2, degree=6):
    """
    特徴x1, x2を組み合わせた degree次の項まで特徴をデータに追加
    bias項に対応するデータ1も追加
    """
    m = x1.shape[0]     # get row
    # bias項を準備
    data_x = np.ones((m, 1))
    # 高次元な項を追加
    for i in range(1, degree+1):
        for j in range(0, i+1):
            new_x = (x1 ** (i - j) * x2 ** j).reshape((m, 1))
            data_x = np.hstack((data_x, new_x))
    return data_x


def main():
    # データを読み込む
    # numpyの配列に格納
    data = np.genfromtxt("../logistic_regression/ex2data1.txt", delimiter=",")
    data_x = data[:, (0, 1)]
    data_y = data[:, 2]

    # データの個数を取得
    m = len(data_x)

    # 訓練データをプロット
    plt.figure(1)
    plot_data(data_x, data_y)

    # 特徴量をマッピング
    data_x = mapFeature(data_x[:, 0], data_x[:, 1], degree=2)

    # 訓練データをシャッフル
    p = np.random.permutation(m)
    data_x = data_x[p, :]
    data_y = data_y[p]

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
        # get data_x's column
        np.zeros(data_x.shape[1], dtype=theano.config.floatX),
        name='theta',
        borrow=True)

    index_s = T.lscalar()
    index_e = T.lscalar()

    # 仮説関数
    h = T.nnet.sigmoid(T.dot(X[index_s:index_e, :], theta))

    # コスト関数を定義する(sigmoid)

    # 交差エントロピー誤差関数
    # 正規化項を導入
    lam = 1.0
    cost = (1.0 / m) * T.sum(-y[index_s:index_e] * T.log(h) - (1 - y[index_s:index_e]) * T.log(1 - h)) + (lam / (2 * m)) * T.sum(theta ** 2)
    # 誤差関数
    # cost = T.mean((y[index_s:index_e] - h) ** 2)

    # コスト関数の微分を定義
    g_cost = theano.grad(cost=cost, wrt=theta)

    # パラメータを更新
    learning_rate = 0.00001
    updates = [(theta, theta - learning_rate * g_cost)]

    # 訓練用の関数を定義
    train_model = theano.function(
        inputs=[index_s, index_e],
        outputs=cost,
        updates=updates
    )

    # 繰り返す
    cost_graph = []
    batch_size = 10
    for cnt in range(50000):
        for i in range(0, m, batch_size):
            current_cost = train_model(i, i + batch_size)
        if cnt % 100 == 0:
            cost_graph.append(current_cost)
            print cnt, current_cost

    # thetaの値を更新
    t = theta.get_value()
    print 'theta:', t

    # 実際の分布を表示する
    # plt.figure(1)
    # xmin, xmax = min(data_x[:, 1]), max(data_x[:, 1])
    # xs = np.linspace(xmin, xmax, 100)
    # ys = [- (t[0] / t[2]) - (t[1] / t[2]) * x for x in xs]
    # plt.plot(xs, ys, 'b-', label="decision boundary")

    # 決定境界を描画
    plt.figure(1)
    gridsize = 100
    xmin, xmax = min(data_x[:, 1]), max(data_x[:, 1])
    x1_vals = np.linspace(xmin, xmax, gridsize)
    x2_vals = np.linspace(xmin, xmax, gridsize)
    x1_vals, x2_vals = np.meshgrid(x1_vals, x2_vals)
    z = np.zeros((gridsize, gridsize))
    for i in range(gridsize):
        for j in range(gridsize):
            x1 = np.array([x1_vals[i, j]])
            x2 = np.array([x2_vals[i, j]])
            z[i, j] = np.dot(mapFeature(x1, x2, degree=2), theta.get_value())

    # 決定境界はsigmoid(z)=0.5、すなわちz=0の場所
    plt.contour(x1_vals, x2_vals, z, levels=[0])

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim((30, 100))
    plt.ylim((30, 100))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
