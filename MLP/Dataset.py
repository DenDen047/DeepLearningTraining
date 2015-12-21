#coding: utf-8

import numpy as np
import data


class DataSet(object):
    """docstring for DataSet"""
    def __init__(self):
        super(DataSet, self).__init__()

    def getDataset(self, N_rate):
        dataset = self.loadDataset()
        return self.split2TrainTest(dataset, N_rate=N_rate)

    # データセットをロード
    def loadDataset(self):
        print 'load MNIST dataset'
        dataset = data.load_mnist_data()
        dataset['data'] = dataset['data'].astype(np.float32)
        dataset['data'] /= 255    # 最終的にnpの行列データを取得
        dataset['target'] = dataset['target'].astype(np.int32)
        return dataset

    # 訓練データとテストデータに分割
    # N ... 訓練データの個数
    def split2TrainTest(self, dataset, N_rate=0.8):
        N = int(dataset['data'].shape[0] * N_rate)
        x_train, x_test = np.split(dataset['data'], [N])  # [0:N], [N:]
        N = int(dataset['target'].shape[0] * N_rate)
        y_train, y_test = np.split(dataset['target'], [N])
        # N_test = y_test.size
        return [x_train, x_test, y_train, y_test]
