#coding: utf-8
import os
import time
import numpy as np
import theano
import theano.tensor as T


# データを読み込む
# 訓練データをプロット
# コスト関数を定義する(sigmoid)
# コスト関数の微分を定義
# 勾配降下法
# パラメータを更新
# 訓練用の関数を定義
# 繰り返す
# 決定境界を表示
