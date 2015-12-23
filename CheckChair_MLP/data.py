import os
import csv
import numpy as np

parent_dir = \
    '~/Documents/programming/python/DeepLearning/DeepLearningTraining/CheckChair_MLP/data/'
train_csv = parent_dir + 'train.csv'
train_csv = os.path.expanduser(train_csv)


def load_train_data():
    csvf = open(train_csv)
    reader = csv.reader(csvf)
    next(reader, None)  # skip header
    # clean up data
    dataset = {}
    dataset['data'] = []
    dataset['target'] = []
    for line in reader:
        dataset['data'].append(line[0:50])
        dataset['target'].append(line[50])
    dataset['data'] = np.array(dataset['data'], dtype=np.float32)
    dataset['target'] = np.array(dataset['target'], dtype=np.int32)
    return dataset

# dataset = load_train_data()
# print len(dataset['data'][0])
# print dataset
