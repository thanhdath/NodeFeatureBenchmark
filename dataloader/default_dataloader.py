import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import scipy.sparse as sp


def read_node_label(filename):
    fin = open(filename, 'r')
    labels = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        labels[vec[0]] = vec[1:]
    fin.close()
    return labels


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

    def __init__(self, datadir, add_self_loop=False):
        self.datadir = datadir
        labels = read_node_label(datadir+'/labels.txt')
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
        
        # if add_self_loop:
        #     print("Add self loop to graph")
        #     self.graph.add_edges_from(list(zip(self.graph.nodes(), self.graph.nodes())))

        labels = convert_labels_to_binary(labels, self.graph)
        self.labels = labels.argmax(axis=1)

        self.train_mask = _sample_mask(range(140), self.labels.shape[0])
        self.val_mask = _sample_mask(range(200, 500), self.labels.shape[0])
        self.test_mask = _sample_mask(range(500, 1500), self.labels.shape[0])
        print("""Graph info: 
            - Number of nodes: {}
            - Number of edges: {}
            - Number of classes: {}
            - Add self loops: {}
            - Train samples: {}
            - Val   samples: {}
            - Test  samples: {}""".format(self.graph.number_of_nodes(),
                                          len(self.graph.edges()), 
                                          self.labels.max() + 1,
                                          add_self_loop,
                                          self.train_mask.sum(), 
                                          self.val_mask.sum(), 
                                          self.test_mask.sum()))
