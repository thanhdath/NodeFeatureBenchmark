"""
PyTorch compatible dataloader

"""

import time
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import dgl
from dgl import DGLGraph
import os 
from main import get_feature_initialization

# default collate function
def collate(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, labels = map(list, zip(*samples))
    for g in graphs:
        # deal with node feats
        for feat in g.node_attr_schemes().keys():
            # TODO torch.Tensor is not recommended
            # torch.DoubleTensor and torch.tensor
            # will meet error in executor.py@runtime line 472, tensor.py@backend line 147
            # RuntimeError: expected type torch.cuda.DoubleTensor but got torch.cuda.FloatTensor
            g.ndata[feat] = torch.Tensor(g.ndata[feat])
        # no edge feats
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels


class GraphDataLoader():
    def __init__(self,
                 dataset,
                 batch_size,
                 device,
                 collate_fn=collate,
                 seed=40,
                 shuffle=True,
                 split_ratio=[.7, .1, .2]):

        self.kwargs = {'pin_memory': True} if 'cuda' in device.type else {}
        labels = [l for _, l in dataset]

        train_idx, valid_idx, test_idx = self._split_rand(labels, split_ratio, seed)
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        self.train_loader = DataLoader(
            dataset, sampler=train_sampler,
            batch_size=batch_size, collate_fn=collate, **self.kwargs)
        self.valid_loader = DataLoader(
            dataset, sampler=valid_sampler,
            batch_size=batch_size, collate_fn=collate, **self.kwargs)
        self.test_loader = DataLoader(
            dataset, sampler=test_sampler,
            batch_size=batch_size, collate_fn=collate, **self.kwargs)

    def train_valid_loader(self):
        return self.train_loader, self.valid_loader, self.test_loader

    def _split_rand(self, labels, split_ratio=[.7, .1, .2], seed=0):
        state = np.random.get_state()
        np.random.seed(seed)
        num_entries = len(labels)
        indices = list(range(num_entries))
        np.random.shuffle(indices)
        n_train = int(num_entries*split_ratio[0])
        n_val = int(num_entries*split_ratio[1])
        train_idx = indices[:n_train]
        valid_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]

        print(
            "train-val-test= %d : %d : %d",
            len(train_idx), len(valid_idx), len(test_idx))
        
        np.random.set_state(state)
        return train_idx, valid_idx, test_idx

class TUDataset(object):
    """
    TUDataset contains lots of graph kernel datasets for graph classification.
    Use provided node feature by default. If no feature provided, use one-hot node label instead.
    If neither labels provided, use constant for node feature.

    :param name: Dataset Name, such as `ENZYMES`, `DD`, `COLLAB`
    :param use_pandas: Default: False.
        Numpy's file read function has performance issue when file is large,
        using pandas can be faster.
    :param hidden_size: Default 10. Some dataset doesn't contain features.
        Use constant node features initialization instead, with hidden size as `hidden_size`.

    """
    def __init__(self, name, args, use_pandas=False, hidden_size=10):

        self.name = name
        self.hidden_size = hidden_size
        # self.extract_dir = self._download()
        self.extract_dir = "data"
        if use_pandas:
            import pandas as pd
            DS_edge_list = self._idx_from_zero(
                pd.read_csv(self._file_path("A"), delimiter=",", dtype=int, header=None).values)
        else:
            DS_edge_list = self._idx_from_zero(
                np.genfromtxt(self._file_path("A"), delimiter=",", dtype=int))

        DS_indicator = self._idx_from_zero(
            np.genfromtxt(self._file_path("graph_indicator"), dtype=int))
        DS_graph_labels = self._idx_from_zero(
            np.genfromtxt(self._file_path("graph_labels"), dtype=int))

        g = dgl.DGLGraph()
        g.add_nodes(int(DS_edge_list.max()) + 1)
        g.add_edges(DS_edge_list[:, 0], DS_edge_list[:, 1])

        node_idx_list = []
        for idx in range(np.max(DS_indicator) + 1):
            node_idx = np.where(DS_indicator == idx)
            node_idx_list.append(node_idx[0])
        self.graphs = g.subgraphs(node_idx_list)
        self.labels = DS_graph_labels

        stime = time.time()
        if args.init == "ori": # use node attributes
            print("Init features: Original , node attributes")
            DS_node_attr = np.loadtxt(self._file_path("node_attributes"), delimiter=",")
            for idxs, g in zip(node_idx_list, self.graphs):
                g.ndata['attr'] = DS_node_attr[idxs, :]
            if args.norm_features:
                g.ndata['attr'] = (g.ndata['attr'] - g.ndata['attr'].mean())/g.ndata['attr'].std()
        elif args.init == "label": # use node label as node features
            print("Init features: node labels")
            DS_node_labels = self._idx_from_zero(np.loadtxt(self._file_path("node_labels"), dtype=int))
            g.ndata['node_label'] = DS_node_labels
            one_hot_node_labels = self._to_onehot(DS_node_labels)
            for idxs, g in zip(node_idx_list, self.graphs):
                g.ndata['attr'] = one_hot_node_labels[idxs, :]
            if args.norm_features:
                g.ndata['attr'] = (g.ndata['attr'] - g.ndata['attr'].mean())/g.ndata['attr'].std()
        else:
            print("Init features:", args.init)
            for graph in self.graphs:
                features = get_feature_initialization(args, graph.to_networkx(), inplace=False)
                graph.ndata['attr'] = np.array([features[int(x)] for x in graph.nodes()])
        print("Time init features: {:.3f}s".format(time.time()-stime))

    def __getitem__(self, idx):
        """Get the i^th sample.
        Paramters
        ---------
        idx : int
            The sample index.
        Returns
        -------
        (dgl.DGLGraph, int)
            DGLGraph with node feature stored in `feat` field and node label in `node_label` if available.
            And its label.
        """
        g = self.graphs[idx]
        return g, self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def _file_path(self, category):
        return os.path.join(self.extract_dir, self.name, "{}_{}.txt".format(self.name, category))

    @staticmethod
    def _idx_from_zero(idx_tensor):
        return idx_tensor - np.min(idx_tensor)

    @staticmethod
    def _to_onehot(label_tensor):
        label_num = label_tensor.shape[0]
        assert np.min(label_tensor) == 0
        one_hot_tensor = np.zeros((label_num, np.max(label_tensor) + 1))
        one_hot_tensor[np.arange(label_num), label_tensor] = 1
        return one_hot_tensor

    def statistics(self):
        input_dim = self.graphs[0].ndata['attr'].shape[1]
        label_dim = self.labels.max() + 1
        max_num_nodes = max([len(x.nodes()) for x in self.graphs])
        return input_dim, label_dim, max_num_nodes