import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
import torch

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

class Dataset(object):
    def __init__(self, datadir):
        self.datadir = datadir
        self._load()
    def _load(self):
        # edgelist
        edgelist_file = self.datadir+'/edgelist.txt'
        self.graph = nx.read_edgelist(edgelist_file, nodetype=int)
        self.id2idx = {node: i for i, node in enumerate(self.graph.nodes())}
        self.idx2id = {v: k for k,v in self.id2idx.items()}
        self.graph = nx.relabel_nodes(self.graph, self.id2idx)
        # adj = nx.to_scipy_sparse_matrix(self.graph)
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # self.graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
        # labels
        labels_path = self.datadir + '/labels.txt'
        self.labels = read_node_label(labels_path)
        self.labels = self._convert_labels_to_binary(self.labels)
        self.n_classes = int(self.labels.max()) + 1
        
    def _convert_labels_to_binary(self, labels):
        labels_arr = []
        for node in sorted(self.graph.nodes()):
            labels_arr.append(labels[str(self.idx2id[node])])
        binarizer = MultiLabelBinarizer(sparse_output=False)
        binarizer.fit(labels_arr)
        labels = binarizer.transform(labels_arr)
        labels = torch.LongTensor(labels).argmax(dim=1)
        return labels
        