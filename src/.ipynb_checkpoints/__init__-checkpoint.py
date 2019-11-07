import time
import pandas as pd
import os
import pydotplus as pdp
import pickle
from contextlib import contextmanager
from sklearn import tree

path_to_data = '../data/row/train.csv'
path_to_output_data = '../data/output/'
data_encoding = "SHIFT-JIS"

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    
def load_datasets(feats):
    train = pd.read_csv(path_to_data, encoding=data_encoding, index_col=0)
    x_train = train[feats].values
    return x_train

def load_target(target_name):
    train = pd.read_csv(path_to_data, encoding=data_encoding, index_col=0)
    x_train = train[target_name].values
    return x_train

def df_load_datasets(feats):
    train = pd.read_csv(path_to_data, encoding=data_encoding, index_col=0)
    x_train = train[feats]
    return x_train

def load_target(target_name):
    train = pd.read_csv(path_to_data, encoding=data_encoding, index_col=0)
    x_train = train[target_name]
    return x_train

def load_model(name):
    with open('../data/models/' + str(name) + '.pickle', 'rb') as model:
        model = pickle.load(model)
    return model

def graph_insight(data):
    df_num = data.select_dtypes(include = ['float64', 'int64'])
    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);
    
def make_decision_tree(model, feats):
    #生成された木を可視化
    for i, one_estimator in enumerator(model.estimators_):
        estimator = one_estimator
        filename = path_to_output_data + one_estimator.__class__.__name__ + str(i) + '.png'
        dot_data = tree.export_graphviz(
            estimator,
            out_file=None,
            filled=True,
            rounded=True,
            feature_names=feats,
            class_names=['false', 'true'],
            special_characters=True
        )
        graph = pdp.graph_from_dot_data(dot_data)
        graph.write_png(filename)

