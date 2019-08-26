import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch
import scipy.sparse as sp
# from networkit import *
import os

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

        # edgelist_file = self.datadir + "/edgelist_networkit.txt"
        # adj = self.graph.adj
        # if not os.path.isfile(edgelist_file):
        #     with open(edgelist_file, "w+") as fp:
        #         fp.write("{} {}".format(adj.shape[0], int(adj.sum())))
        #         for i in range(adj.shape[0]):
        #             fp.write("\n" + " ".join(map(lambda x : str(x+1), adj[i].nonzero()[1])) )
        # self.graph = readGraph(edgelist_file, Format.METIS)
        # import pdb; pdb.set_trace()

        features = {}
        with open(datadir+'/labels.txt') as fp:
            for line in fp:
                elms = line.strip().split()
                features[int(elms[0])] = np.array([float(x) for x in elms[1:]])
        self.features = np.array([features[x] for x in self.graph.nodes()])

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

class DefaultInductiveDataloader():
    """
    load data from data/ directory which contains edgelist.txt, labels.txt
    """

    def __init__(self, datadir, mode):
        self.datadir = datadir
        self.mode = mode
        self._load()

    def _load(self):
        graph_file = '{}/{}_graph.npz'.format(self.datadir, self.mode)
        if not os.path.isfile(graph_file):  
            self._store_train_val_test()
        npz = np.load(graph_file)
        adj = npz['graph'][()]
        self.graph = nx.from_scipy_sparse_matrix(adj)
        self.labels = npz['labels'][()]
        self.features = np.load('{}/{}_feats.npy'.format(self.datadir, self.mode))
        # if self.mode == "train":
        #     self.train_nodes = npz["train_nodes"][()]
        # elif self.mode == "valid":
        #     self.valid_nodes = npz["valid_nodes"][()]
        # elif self.mode == "test":
        #     self.test_nodes = npz["test_nodes"][()]
        self.mask = npz["{}_nodes".format(self.mode)][()]
        self.n_classes = int(npz["n_classes"][()])
        self.labels = torch.LongTensor(self.labels)
        self.multiclass = npz["multiclass"][()]

    def _store_train_val_test(self):
        datadir = self.datadir
        labels, multiclass = read_node_label(datadir+'/labels.txt')
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

        features = {}
        with open(datadir+'/features.txt') as fp:
            for line in fp:
                elms = line.strip().split()
                features[int(elms[0])] = np.array([float(x) for x in elms[1:]])
        features = np.array([features[x] for x in self.graph.nodes()])

        labels = convert_labels_to_binary(labels, self.graph)
        if multiclass:
            labels = torch.FloatTensor(labels)
            n_classes = labels.shape[1]
        else:
            labels = torch.LongTensor(labels.argmax(axis=1))
            n_classes = int(labels.max() + 1)

        if "cora" in datadir:
            print("Split train-val-test by default for cora dataset.")
            idx_train = list(range(140))
            idx_val = list(range(200, 500))
            idx_test = list(range(500, 1500))
        else:
            print("Split train-val-test 0.7-0.1-0.2.")
            indices = np.random.permutation(np.arange(self.labels.shape[0]))
            n_train = int(len(indices) * 0.7)
            n_val = int(len(indices) * 0.1)
            idx_train = indices[:n_train]
            idx_val = indices[n_train:n_train+n_val]
            idx_test = indices[n_train+n_val:]

        train_adj = adj[idx_train, :][:, idx_train]
        val_adj = adj[idx_train+idx_val, :][:, idx_train+idx_val]
        test_adj = adj

        # save train_graph
        extract_dir = self.datadir
        np.savez_compressed(extract_dir+'/train_graph.npz', graph=train_adj, 
            labels=labels[idx_train], train_nodes=idx_train, 
            n_classes=labels.max()+1, multiclass=multiclass)
        np.save(extract_dir+'/train_feats.npy', features[idx_train])
        # save valid graph
        np.savez_compressed(extract_dir+'/valid_graph.npz', graph=val_adj, 
            labels=labels[idx_val], valid_nodes=list(range(len(idx_train), len(idx_train)+len(idx_val))), 
            n_classes=labels.max()+1, multiclass=multiclass)
        np.save(extract_dir+'/valid_feats.npy', features[idx_train+idx_val])
        # save valid graph
        test_labels = labels[idx_test]
        np.savez_compressed(extract_dir+'/test_graph.npz', graph=test_adj, 
            labels=labels[idx_test], test_nodes=idx_test,
            n_classes=labels.max()+1, multiclass=multiclass)
        np.save(extract_dir+'/test_feats.npy', features)