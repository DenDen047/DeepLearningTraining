# config:utf-8
from pylearn2.datasets.csv_dataset import CSVDataset
import pickle


print 'convert: train.csv -> train.pkl'

pyln_data = CSVDataset(
    "./train.csv",
    expect_headers=True,
    delimiter=',')
pickle.dump(pyln_data, open("./train.pkl", 'w'))
