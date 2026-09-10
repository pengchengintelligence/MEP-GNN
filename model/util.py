import numpy as np
import math
import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Dataset
import os
from scipy.stats import ttest_ind
import logging
import time
import csv
import codecs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import pickle
import copy
import sklearn.metrics
import torch_geometric
from scipy.sparse import coo_matrix

from sklearn.metrics import auc, f1_score, roc_curve, precision_score, recall_score, cohen_kappa_score
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomDatasetWithAdj(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        features = torch.FloatTensor(sample[1:-1].astype(float))
        label = sample[-1]
        return features, label
    
class CustomGeneDataset(Dataset):
    def __init__(self, feature_file, label_file):
        self.features = pd.read_csv(feature_file, index_col=0)

        self.labels = pd.read_csv(label_file, header=None).squeeze()  
        
        assert len(self.features) == len(self.labels), "特征数量和标签数量不一致"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features.iloc[idx].values.astype(float))
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return feature, label
    
class MultiModalWrapperDataset(Dataset):
    def __init__(self, dataset_list):
        self.datasets = dataset_list
        assert all(len(ds) == len(self.datasets[0]) for ds in self.datasets), "模态样本数量必须一致"

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        features = [ds[idx][0] for ds in self.datasets]
        label = self.datasets[0][idx][1]
        return features, label


def select_topk_features_by_ttest(csv_path, mut=10, label_col=-1, id_col=0):

    df = pd.read_csv(csv_path)

    feature_cols = [i for i in range(df.shape[1]) if i != label_col and i != id_col]
    features = df.iloc[:, feature_cols].values
    labels = df.iloc[:, label_col].values

    group1 = features[labels == 0]
    group2 = features[labels == 1]

    t_vals, p_vals = ttest_ind(group1, group2, equal_var=False)

    topk_indices = np.argsort(p_vals)[:mut]

    topk_indices = topk_indices - 1

    return topk_indices



def load_adj_data(folder_path,x):
    file_names = os.listdir(folder_path)

    adj_dict = {}

    for file_name in file_names:

        file_path = os.path.join(folder_path, file_name)

        adj_data = pd.read_csv(file_path, header=None).values.astype(float)
        adj_data = torch.LongTensor(np.where(adj_data > x, 1, 0))

        file_name_without_extension = os.path.splitext(file_name)[0]

        adj_dict[file_name_without_extension] = adj_data
    return adj_dict



def specificity_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return spe
def sensitivity_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = sum((y_true == 1) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return sen

def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer

def adj_to_PyG_edge_index(adj):
    coo_A = coo_matrix(adj)
    edge_index, edge_weight = torch_geometric.utils.convert.from_scipy_sparse_matrix(coo_A)
    return edge_index

def data_to_PyG_data(x, edge_index, y):
    out_data = x
    out_edge_index = edge_index
    out_label = y
    PyG_data = torch_geometric.data.Data(x=out_data, edge_index=out_edge_index, y=out_label)
    return PyG_data

def PyG_edge_index_to_adj(edge_index):
    adj = torch_geometric.utils.to_dense_adj(edge_index=edge_index)
    return adj

def data_write_csv(file_name, datas):
  file_csv = codecs.open(file_name,'w+','utf-8')
  writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
  for data in datas:
    writer.writerow(data)
  print("doc saved")


def save_tcp_values_csv(epoch, view, p_target, daoshu_p_target, tcp_conf, daoshu_tcp_conf, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame({
        "epoch": [epoch] * len(p_target),
        "p_target": p_target.detach().cpu().numpy(),
        "Daoshu_p_target": daoshu_p_target.detach().cpu().numpy(),
        "TCPConfidence": tcp_conf.detach().cpu().numpy(),
        "Daoshu_TCPConfidence": daoshu_tcp_conf.detach().cpu().numpy()
    })

    filename = f"view{view}_tcp_values.csv"
    file_path = os.path.join(save_dir, filename)
    df.to_csv(file_path, mode='w', header=True, index=False)