import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os, sys

import dgl
from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url
from .default_dataloader import _sample_mask
from dgl.data.citation_graph import _parse_index_file, _preprocess_features
import torch

_urls = {
    'cora' : 'dataset/cora_raw.zip',
    'citeseer' : 'dataset/citeseer.zip',
    'pubmed' : 'dataset/pubmed.zip',
    'cora_binary' : 'dataset/cora_binary.zip',
}

def _pickle_load(pkl_file):
    if sys.version_info > (3, 0):
        return pkl.load(pkl_file, encoding='latin1')
    else:
        return pkl.load(pkl_file)

class NELLDataloader(object):
    def __init__(self, datadir):
        elms = datadir.split('/')
        self.name = elms[-1]
        self.dir = '/'.join(elms[:-1])
        self._load()

    def _load(self):
        """Loads input data from gcn/data directory

        ind.name.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.name.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.name.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.name.x) as scipy.sparse.csr.csr_matrix object;
        ind.name.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.name.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.name.ally => the labels for instances in ind.name.allx as numpy.ndarray object;
        ind.name.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.name.test.index => the indices of test instances in graph, for the inductive setting as list object.

        All objects above must be saved using python pickle module.

        :param name: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
        dataset = 'nell.0.1'
        root = '{}/{}'.format(self.dir, self.name)
        objnames = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(objnames)):
            with open("{}/ind.{}.{}".format(root, dataset, objnames[i]), 'rb') as f:
                objects.append(_pickle_load(f))
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = _parse_index_file("{}/ind.{}.test.index".format(root, dataset))
        test_idx_range = np.sort(test_idx_reorder)
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        graph = nx.from_scipy_sparse_matrix(adj)

        onehot_labels = np.vstack((ally, ty))
        onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
        labels = np.argmax(onehot_labels, 1)

        idx_train = range(len(y))
        idx_test = test_idx_range.tolist()
        idx_val = range(len(y), len(y)+500)
        
        train_mask = _sample_mask(idx_train, labels.shape[0])
        val_mask = _sample_mask(idx_val, labels.shape[0])
        test_mask = _sample_mask(idx_test, labels.shape[0])

        self.graph = graph
        self.features = features
        self.labels = torch.LongTensor(labels)
        self.onehot_labels = onehot_labels
        self.num_labels = onehot_labels.shape[1]
        self.train_mask = torch.ByteTensor(train_mask)
        self.val_mask = torch.ByteTensor(val_mask)
        self.test_mask = torch.ByteTensor(test_mask)
        self.multiclass = False
        self.n_classes = int(self.labels.max() + 1)

        features_file = self.dir + '/' + self.name + '/features.npz'
        if not os.path.isfile(features_file):
            features = np.asarray(features.todense())
            features_dict = {node: features[i] for i, node in enumerate(graph.nodes())}
            np.savez_compressed(features_file, features=features_dict)

        label_file = self.dir + '/' + self.name + '/labels.npz'
        if not os.path.isfile(label_file):
            labels_dict = {node: labels[i] for i, node in enumerate(graph.nodes())}
            np.savez_compressed(label_file, labels=labels_dict)

        print('Finished data loading and preprocessing.')
        print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
        print('  NumEdges: {}'.format(self.graph.number_of_edges()))
        print('  NumFeats: {}'.format(self.features.shape[1]))
        print('  NumClasses: {}'.format(self.num_labels))
        print('  NumTrainingSamples: {}'.format(self.train_mask.sum()))
        print('  NumValidationSamples: {}'.format(self.val_mask.sum()))
        print('  NumTestSamples: {}'.format(self.test_mask.sum()))

    def __getitem__(self, idx):
        return self

    def __len__(self):
        return 1
