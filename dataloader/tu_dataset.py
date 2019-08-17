import time
from dgl.data.utils import download, get_download_dir
import dgl 
import numpy as np
import os
import torch
import tarfile 
import zipfile

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
    _url = r"https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/{}.zip"

    def __init__(self, datadir, use_pandas=False, ratio=[.7, .1, .2]):
        elms = datadir.split('/')
        if len(elms[-1]) == 0: elms = elms[:-1]
        self.download_dir = '/'.join(elms[:-1])
        self.name = elms[-1]
        self.extract_dir = self._download() 
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
        self.graph_lists = g.subgraphs(node_idx_list)
        self.graph_labels = DS_graph_labels

        # load node attributes
        try:
            self.node_attr = np.loadtxt(self._file_path("node_attributes"), delimiter=",")
            if len(self.node_attr.shape) == 1:
                self.node_attr = np.expand_dims(self.node_attr, 1)
        except:
            print("Graph does not has node attributes.")

        # load node labels
        try:
            DS_node_labels = self._idx_from_zero(np.loadtxt(self._file_path("node_labels"), dtype=int))
            g.ndata['node_label'] = DS_node_labels
            self.one_hot_node_labels = self._to_onehot(DS_node_labels)
        except:
            print("Graph does not has node attributes.")

        # split train val test
        train_size = int(ratio[0] * len(self.graph_lists))
        test_size = int(ratio[2] * len(self.graph_lists))
        val_size = int(len(self.graph_lists) - train_size - test_size)
        self.dataset_train, self.dataset_val, self.dataset_test = torch.utils.data.random_split(self, (train_size, val_size, test_size))

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
        g = self.graph_lists[idx]
        return g, self.graph_labels[idx]

    def __len__(self):
        return len(self.graph_lists)

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
        input_dim = self.graph_lists[0].ndata['feat'].shape[1]
        label_dim = self.graph_labels.max() + 1
        max_num_nodes = max([len(x.nodes()) for x in self.graph_lists])
        return input_dim, label_dim, max_num_nodes
    
    def _download(self):
        if os.path.isdir(self.download_dir + '/' + self.name):
            return self.download_dir
        zip_file_path = os.path.join(self.download_dir, "tu_{}.zip".format(self.name))
        download(self._url.format(self.name), path=zip_file_path)
        self.extract_archive(zip_file_path, self.download_dir)
        return self.download_dir

    def extract_archive(self, file, target_dir):
        if file.endswith('.gz') or file.endswith('.tar') or file.endswith('.tgz'):
            archive = tarfile.open(file, 'r')
        elif file.endswith('.zip'):
            archive = zipfile.ZipFile(file, 'r')
        else:
            raise Exception('Unrecognized file type: ' + file)
        print('Extracting file to {}'.format(target_dir))
        archive.extractall(path=target_dir)
        archive.close()
