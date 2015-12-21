#coding: utf-8
import argparse
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import time

from Dataset import DataSet


parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument(
    '--gpu', '-g',
    default=-1,
    type=int,
    help='GPU ID (negative value indicates CPU')    # 'gpu' option

# GPUが使えるかを確認
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
# GPUが使えるなら
xp = cuda.cupy if args.gpu >= 0 else np

# 学習のパラメータ
batchsize = 100
n_epoch = 20
i_units = 28 * 28
n_units = 1024
o_units = 10

# テスト精度を出力するファイル名
accfile = './data/accuracy.txt'

# データセットをロード
x_train, x_test, y_train, y_test = DataSet().getDataset(N_rate=0.87)
N_test = y_test.size

# 多層パーセプトロンのモデル（パラメータ集合）
model = chainer.FunctionSet(
    l1=F.Linear(i_units, n_units),  # input - hidden1
    l2=F.Linear(n_units, n_units),  # hidden1 - hidden2
    l3=F.Linear(n_units, o_units)   # hidden2 - output
)

# GPU使用のときはGPUにモデルを転送
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()


def forward(x_data, y_data, train=True):
    """ 順伝搬の処理を定義 """
    # train ... 学習フラグ（Falseにすると学習しない）
    # 訓練時は，dropoutを実行
    # テスト時は，dropoutを無効

    # 入力と教師データ
    # データ配列は，Chainer.Variable型にしないといけない
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    # 隠れ層1の出力
    h1 = F.dropout(F.relu(model.l1(x)), train=train)
    # 隠れ層2の出力
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    # 出力層の出力
    y = model.l3(h2)

    # 訓練時とテスト時で返す値を変える
    # y ... ネットワークの出力（仮説）
    # t ... 教師データ
    if train:   # 訓練
        # 誤差を返す
        # 多クラス分類なので，誤差関数としてソフトマックス関数の
        # クロスエントロピー関数を使う
        loss = F.softmax_cross_entropy(y, t)
        return loss
    else:   # テスト
        # 精度を返す
        acc = F.accuracy(y, t)
        return acc


# Optimizerをセット
# 最適化対象であるパラメータ集合のmodelを渡しておく
optimizer = optimizers.Adam()
optimizer.setup(model)

# テスト精度を出力するファイル
fp = open(accfile, 'w')
fp.write('epoch\ttest_accuracy\n')

start_time = time.clock()

# 訓練ループ
# 各epochでテスト精度を求める
# epoch ... 訓練回数
N = x_train.shape[0]
for epoch in range(1, n_epoch + 1):
    print 'epoch: %d' % epoch

    # 訓練データを用いてパラメータを更新する
    perm = np.random.permutation(N)     # 1~Nまでの数字をランダムに並び替える
    sum_loss = 0
    for i in range(0, N, batchsize):
        # asanyarray ... pythonの配列を，xpの配列に変換
        x_batch = xp.asanyarray(x_train[perm[i:i + batchsize]])
        y_batch = xp.asanyarray(y_train[perm[i:i + batchsize]])

        # 勾配を初期化
        optimizer.zero_grads()
        # 順伝播させて誤差と精度を算出
        loss = forward(x_batch, y_batch)
        # 誤差逆伝播で勾配を計算
        loss.backward()
        # 勾配が求まったので，それをもとに最適化を実行する
        optimizer.update()

        # len(y_batch)は，実質batchsize
        sum_loss += float(loss.data) * len(y_batch)

    print 'train mean loss: %f' % (sum_loss / N)

    # テストデータを用いて精度を評価する
    # accuracy ... 精度
    sum_accuracy = 0
    for i in range(0, N_test, batchsize):
        x_batch = xp.asanyarray(x_train[perm[i:i + batchsize]])
        y_batch = xp.asanyarray(y_train[perm[i:i + batchsize]])

        # 順伝播させて精度を算出
        # train ... 学習フラグ（Falseにすると学習しない = 精度だけreturn）
        acc = forward(x_batch, y_batch, train=False)
        # 精度の合計を計算
        sum_accuracy += float(acc.data) * len(y_batch)

    print 'test accuracy: %f' % (sum_accuracy / N_test)

    fp.write('%d\t%f\n' % (epoch, sum_accuracy / N_test))
    fp.flush()

end_time = time.clock()
print end_time - start_time

fp.close()
