#coding: utf-8
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

    # データの個数を取得
    m = len(data_x)

    # 訓練データをプロット
    plt.figure(1)
    plot_data(data_x, data_y)

    # 特徴量data_xに定数項を追加 .. x_0
    data_x = np.hstack((np.ones((m, 1)), data_x))

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
        np.zeros(3, dtype=theano.config.floatX),
        name='theta',
        borrow=True)

    index = T.lscalar()

    # 仮説関数
    h = T.nnet.sigmoid(T.dot(theta, X[index, :]))

    # コスト関数を定義する(sigmoid)

    # 交差エントロピー誤差関数
    # 正規化項を導入
    # lam = 1.0
    # cost = (1.0 / m) * T.sum(-y[index] * T.log(h) - (1 - y[index]) * T.log(1 - h)) \
    #     + lam / (2 * m) * T.sum(theta ** 2)
    # cost = -y[index] * T.log(h) - (1 - y[index]) * T.log(1 - h)

    # 誤差関数
    cost = T.mean((y[index] - h) ** 2)

    # コスト関数の微分を定義
    g_cost = theano.grad(cost=cost, wrt=theta)

    # パラメータを更新
    learning_rate = 0.0005
    updates = [(theta, theta - learning_rate * g_cost)]

    # 訓練用の関数を定義
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates
    )

    # 繰り返す
    cost_graph = []
    cnt = 0
    for cnt in range(10000):
        for i in range(m):
            current_cost = train_model(i)
        if cnt % 100 == 0:
            cost_graph.append(current_cost)
            print cnt, current_cost

    # thetaの値を更新
    t = theta.get_value()

    # 実際の分布を表示する
    plt.figure(1)
    xmin, xmax = min(data_x[:, 1]), max(data_x[:, 1])
    xs = np.linspace(xmin, xmax, 100)
    ys = [- (t[0] / t[2]) - (t[1] / t[2]) * x for x in xs]
    plt.plot(xs, ys, 'b-', label="decision boundary")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim((30, 100))
    plt.ylim((30, 100))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
