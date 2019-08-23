import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch
import scipy.sparse as sp

def read_node_label(filename):
    fin = open(filename, 'r')
    labels = {}
    multiclass = False
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        if len(vec[1:]) > 1: multiclass = True
        labels[vec[0]] = vec[1:]
    fin.close()
    return labels, multiclass

def scipy_to_sparse_tensor(matrix):
    coo = matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

def convert_labels_to_binary(labels, graph):
    mlb = MultiLabelBinarizer()
    labels_arr = np.array([labels[str(x)] for x in graph.nodes()])
    mlb.fit_transform(labels_arr)
    return mlb.transform(labels_arr)


def _sample_mask(idx, length):
    mask = np.zeros((length), dtype=np.int32)
    mask[idx] = True
    return mask


class DefaultDataloader():
    """
    load data from data/ directory which contains edgelist.txt, labels.txt
    """

    def __init__(self, datadir):
        self.datadir = datadir
        labels, self.multiclass = read_node_label(datadir+'/labels.txt')
        # build graph
        edges = []
        with open(datadir+'/edgelist.txt') as fp:
            for line in fp:
                src, trg = line.strip().split()[:2]
                edges.append([int(src), int(trg)])
        edges = np.array(edges)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])),
            shape=(len(labels), len(labels)), dtype=np.int32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        self.graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
        
        labels = convert_labels_to_binary(labels, self.graph)
        if self.multiclass:
            self.labels = torch.FloatTensor(labels)
            self.n_classes = self.labels.shape[1]
        else:
            self.labels = torch.LongTensor(labels.argmax(axis=1))
            self.n_classes = int(self.labels.max() + 1)

        if "cora" in datadir:
            print("Split train-val-test by default for cora dataset.")
            self.train_mask = torch.ByteTensor(_sample_mask(range(140), self.labels.shape[0]))
            self.val_mask = torch.ByteTensor(_sample_mask(range(200, 500), self.labels.shape[0]))
            self.test_mask = torch.ByteTensor(_sample_mask(range(500, 1500), self.labels.shape[0]))
        else:
            print("Split train-val-test 0.7-0.1-0.2.")
            indices = np.random.permutation(np.arange(self.labels.shape[0]))
            n_train = int(len(indices) * 0.7)
            n_val = int(len(indices) * 0.1)
            self.train_mask = torch.ByteTensor(_sample_mask(indices[:n_train], self.labels.shape[0]))
            self.val_mask = torch.ByteTensor(_sample_mask(indices[n_train:n_train+n_val], self.labels.shape[0]))
            self.test_mask = torch.ByteTensor(_sample_mask(indices[n_train+n_val:], self.labels.shape[0]))
        print("""Graph info: 
            - Number of nodes: {}
            - Number of edges: {}
            - Number of classes: {} - multiclass: {}
            - Train samples: {}
            - Val   samples: {}
            - Test  samples: {}""".format(self.graph.number_of_nodes(),
                                          len(self.graph.edges()), 
                                          self.n_classes, self.multiclass,
                                          self.train_mask.sum(), 
                                          self.val_mask.sum(), 
                                          self.test_mask.sum()))
