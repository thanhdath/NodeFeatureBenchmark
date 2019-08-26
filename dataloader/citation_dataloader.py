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
import json
from networkx.readwrite import json_graph

_urls = {
    # 'cora' : 'dataset/cora_raw.zip',
    'citeseer' : 'dataset/citeseer.zip',
    'pubmed' : 'dataset/pubmed.zip',
    # 'cora_binary' : 'dataset/cora_binary.zip',
}

def _pickle_load(pkl_file):
    if sys.version_info > (3, 0):
        return pkl.load(pkl_file, encoding='latin1')
    else:
        return pkl.load(pkl_file)

class CitationDataloader(object):
    def __init__(self, datadir):
        elms = datadir.split('/')
        self.name = elms[-1]
        self.dir = '/'.join(elms[:-1])
        self.zip_file_path='{}/{}.zip'.format(self.dir, self.name)
        if not os.path.isfile(self.zip_file_path):
            download(_get_dgl_url(_urls[self.name]), path=self.zip_file_path)
            extract_archive(self.zip_file_path, '{}/{}'.format(self.dir, self.name))
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
        root = '{}/{}'.format(self.dir, self.name)
        objnames = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(objnames)):
            with open("{}/ind.{}.{}".format(root, self.name, objnames[i]), 'rb') as f:
                objects.append(_pickle_load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = _parse_index_file("{}/ind.{}.test.index".format(root, self.name))
        test_idx_range = np.sort(test_idx_reorder)

        if self.name == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        graph = nx.DiGraph(nx.from_dict_of_lists(graph))

        onehot_labels = np.vstack((ally, ty))
        onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
        labels = np.argmax(onehot_labels, 1)

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
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

        features_file = self.dir + '/' + self.name + '/features.npy'
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

class CitationInductiveDataloader(object):
    def __init__(self, datadir, mode):
        """
        mode: choose from "train", "valid", "test"
        """
        elms = datadir.split('/')
        self.name = elms[-1]
        self.dir = '/'.join(elms[:-1])
        self.zip_file_path='{}/{}.zip'.format(self.dir, self.name)
        if not os.path.isfile(self.zip_file_path):
            download(_get_dgl_url(_urls[self.name]), path=self.zip_file_path)
            extract_archive(self.zip_file_path, '{}/{}'.format(self.dir, self.name))
        self.mode = mode
        self._load()

    def _load(self):
        graph_file = '{}/{}/{}_graph.npz'.format(self.dir, self.name, self.mode)
        if not os.path.isfile(graph_file):  
            self._store_train_val_test()
        npz = np.load(graph_file, allow_pickle=True)
        adj = npz['graph'][()]
        self.graph = nx.from_scipy_sparse_matrix(adj)
        self.labels = npz['labels'][()]
        self.features = np.load('{}/{}/{}_feats.npy'.format(self.dir, self.name, self.mode), allow_pickle=True)
        # if self.mode == "train":
        #     self.train_nodes = npz["train_nodes"][()]
        # elif self.mode == "valid":
        #     self.valid_nodes = npz["valid_nodes"][()]
        # elif self.mode == "test":
        #     self.test_nodes = npz["test_nodes"][()]
        self.mask = npz["{}_nodes".format(self.mode)][()]
        self.n_classes = int(npz["n_classes"][()])
        self.labels = torch.LongTensor(self.labels)
        self.multiclass = False

    def _store_train_val_test(self):
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
        root = '{}/{}'.format(self.dir, self.name)
        objnames = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(objnames)):
            with open("{}/ind.{}.{}".format(root, self.name, objnames[i]), 'rb') as f:
                objects.append(_pickle_load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = _parse_index_file("{}/ind.{}.test.index".format(root, self.name))
        test_idx_range = np.sort(test_idx_reorder)

        if self.name == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        graph = nx.DiGraph(nx.from_dict_of_lists(graph))

        onehot_labels = np.vstack((ally, ty))
        onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
        labels = np.argmax(onehot_labels, 1)

        idx_test = test_idx_range.tolist()
        idx_train = list(range(len(y)))
        idx_val = list(range(len(y), len(y)+500))

        train_mask = _sample_mask(idx_train, labels.shape[0])
        val_mask = _sample_mask(idx_val, labels.shape[0])
        test_mask = _sample_mask(idx_test, labels.shape[0])

        adj = nx.to_scipy_sparse_matrix(graph)
        train_adj = adj[idx_train, :][:, idx_train]
        val_adj = adj[idx_train+idx_val, :][:, idx_train+idx_val]
        test_adj = adj

        # save train_graph
        extract_dir = self.dir + '/' + self.name
        np.savez_compressed(extract_dir+'/train_graph.npz', graph=train_adj, 
            labels=labels[idx_train], train_nodes=idx_train, 
            n_classes=labels.max()+1)
        np.save(extract_dir+'/train_feats.npy', features[idx_train].todense())
        # save valid graph
        np.savez_compressed(extract_dir+'/valid_graph.npz', graph=val_adj, 
            labels=labels[idx_val], valid_nodes=list(range(len(idx_train), len(idx_train)+len(idx_val))), 
            n_classes=labels.max()+1)
        np.save(extract_dir+'/valid_feats.npy', features[idx_train+idx_val].todense())
        # save valid graph
        test_labels = labels[idx_test]
        np.savez_compressed(extract_dir+'/test_graph.npz', graph=test_adj, 
            labels=labels[idx_test], test_nodes=idx_test,
            n_classes=labels.max()+1)
        np.save(extract_dir+'/test_feats.npy', features.todense())

    def __getitem__(self, idx):
        return self

    def __len__(self):
        return 1
