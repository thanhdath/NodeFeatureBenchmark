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

class CitationDataloader(object):
    def __init__(self, datadir):
        elms = datadir.split('/')
        self.name = elms[-1]
        self.dir = '/'.join(elms[:-1])
        self.zip_file_path='{}/{}.zip'.format(self.dir, self.name)
        if not os.path.isfile(self.zip_file_path):
            download(_get_dgl_url(_urls[self.name]), path=self.zip_file_path)
            extract_archive(self.zip_file_path, '{}/{}'.format(self.dir, self.name))
            self._load(True)
        else:
            self._load(False)

    def _load(self, first_time=True):
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

        if first_time:
            features = np.asarray(features.todense())
            with open(self.dir + '/' + self.name + '/features.txt', 'w+') as fp:
                for i, node in enumerate(graph.nodes()):
                    fp.write("{} {}\n".format(node, ' '.join(map(str, features[i]))))

        print('Finished data loading and preprocessing.')
        print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
        print('  NumEdges: {}'.format(self.graph.number_of_edges()))
        print('  NumFeats: {}'.format(self.features.shape[1]))
        print('  NumClasses: {}'.format(self.num_labels))
        print('  NumTrainingSamples: {}'.format(len(np.nonzero(self.train_mask)[0])))
        print('  NumValidationSamples: {}'.format(len(np.nonzero(self.val_mask)[0])))
        print('  NumTestSamples: {}'.format(len(np.nonzero(self.test_mask)[0])))

    def __getitem__(self, idx):
        return self

    def __len__(self):
        return 1
