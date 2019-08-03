"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import argparse
import time
import networkx as nx
import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args
import dgl.function as fn
from features_init import lookup as lookup_feature_init
from SGC.normalization import row_normalize
from sklearn.preprocessing import MultiLabelBinarizer
import random
from SGC.metrics import f1

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
        # adj = nx.to_scipy_sparse_matrix(graph)
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
            labels_arr.append(labels[str(node)])
        binarizer = MultiLabelBinarizer(sparse_output=False)
        binarizer.fit(labels_arr)
        labels = binarizer.transform(labels_arr)
        labels = torch.LongTensor(labels).argmax(dim=1)
        return labels
    

class Aggregator(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation=None, bias=True):
        super(Aggregator, self).__init__()
        self.g = g
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)  # (F, EF) or (2F, EF)
        self.activation = activation
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, node):
        nei = node.mailbox['m']  # (B, N, F)
        h = node.data['h']  # (B, F)
        h = self.concat(h, nei, node)  # (B, F) or (B, 2F)
        h = self.linear(h)   # (B, EF)
        if self.activation:
            h = self.activation(h)
        norm = torch.pow(h, 2)
        norm = torch.sum(norm, 1, keepdim=True)
        norm = torch.pow(norm, -0.5)
        norm[torch.isinf(norm)] = 0
        # h = h * norm
        return {'h': h}

    @abc.abstractmethod
    def concat(self, h, nei, nodes):
        raise NotImplementedError


class MeanAggregator(Aggregator):
    def __init__(self, g, in_feats, out_feats, activation, bias):
        super(MeanAggregator, self).__init__(g, in_feats, out_feats, activation, bias)

    def concat(self, h, nei, nodes):
        degs = self.g.in_degrees(nodes.nodes()).float()
        if h.is_cuda:
            degs = degs.cuda(h.device)
        concatenate = torch.cat((nei, h.unsqueeze(1)), 1)
        concatenate = torch.sum(concatenate, 1) / degs.unsqueeze(1)
        return concatenate  # (B, F)


class PoolingAggregator(Aggregator):
    def __init__(self, g, in_feats, out_feats, activation, bias):  # (2F, F)
        super(PoolingAggregator, self).__init__(g, in_feats*2, out_feats, activation, bias)
        self.mlp = PoolingAggregator.MLP(in_feats, in_feats, F.relu, False, True)

    def concat(self, h, nei, nodes):
        nei = self.mlp(nei)  # (B, F)
        concatenate = torch.cat((nei, h), 1)  # (B, 2F)
        return concatenate

    class MLP(nn.Module):
        def __init__(self, in_feats, out_feats, activation, dropout, bias):  # (F, F)
            super(PoolingAggregator.MLP, self).__init__()
            self.linear = nn.Linear(in_feats, out_feats, bias=bias)  # (F, F)
            self.dropout = nn.Dropout(p=dropout)
            self.activation = activation
            nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

        def forward(self, nei):
            nei = self.dropout(nei)  # (B, N, F)
            nei = self.linear(nei)
            if self.activation:
                nei = self.activation(nei)
            max_value = torch.max(nei, dim=1)[0]  # (B, F)
            return max_value


class GraphSAGELayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 aggregator_type,
                 bias=True,
                 ):
        super(GraphSAGELayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(p=dropout)
        if aggregator_type == "pooling":
            self.aggregator = PoolingAggregator(g, in_feats, out_feats, activation, bias)
        else:
            self.aggregator = MeanAggregator(g, in_feats, out_feats, activation, bias)

    def forward(self, h):
        h = self.dropout(h)
        self.g.ndata['h'] = h
        self.g.update_all(fn.copy_src(src='h', out='m'), self.aggregator)
        h = self.g.ndata.pop('h')
        return h


class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GraphSAGELayer(g, in_feats, n_hidden, activation, dropout, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphSAGELayer(g, n_hidden, n_hidden, activation, dropout, aggregator_type))
        # output layer
        self.layers.append(GraphSAGELayer(g, n_hidden, n_classes, None, dropout, aggregator_type))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def sample_mask(idx, length):
    mask = np.zeros((length))
    mask[idx] = 1
    return torch.ByteTensor(mask)

def split_train_test(length, ratio=[.7, .1, .2]):
    n_train = int(length*ratio[0])
    n_val = int(length*ratio[1])
    indices = np.random.permutation(np.arange(length))
    train_mask = sample_mask(indices[:n_train], length)
    val_mask = sample_mask(indices[n_train:n_train+n_val], length)
    test_mask = sample_mask(indices[n_train+n_val:], length)
    return train_mask, val_mask, test_mask


def get_feature_initialization(args, graph, inplace = True):
    if args.init not in lookup_feature_init:
        raise NotImplementedError
    kwargs = {}
    if args.init == "ori":
        kwargs = {"feature_path": args.dataset+"/features.txt"}
    elif args.init == "ssvd0.5":
        args.init = "ssvd"
        kwargs = {"alpha": 0.5}
    elif args.init == "ssvd1":
        args.init = "ssvd"
        kwargs = {"alpha": 1}
    init_feature = lookup_feature_init[args.init](**kwargs)
    return init_feature.generate(graph, args.feature_size, inplace=inplace, 
        normalize=args.norm_features)

def main(args):
    data = Dataset(args.dataset)
    labels = data.labels
    # features
    features = get_feature_initialization(args, data.graph, inplace=False)
    features = np.array([features[node] for node in sorted(data.graph.nodes())])
    features = row_normalize(features)
    features = torch.FloatTensor(features)
    # print(labels)
    # print(features.sum())
    # print(list(sorted(data.graph.nodes()))[:10])
    # import sys; sys.exit()
    # 
    train_mask, val_mask, test_mask = split_train_test(len(labels))

    in_feats = features.shape[1]
    n_classes = data.n_classes
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.sum().item(),
           val_mask.sum().item(),
           test_mask.sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        print("use cuda:", args.gpu)

    # graph preprocess and calculate normalization factor
    g = DGLGraph(data.graph)
    n_edges = g.number_of_edges()

    # create GraphSAGE model
    model = GraphSAGE(g,
                      in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.aggregator_type
                      )
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_acc = 0
    stime = time.time()
    for epoch in range(args.n_epochs):
        model.train()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        etime = time.time() - stime
        if epoch % 20 == 0:
            print('Epoch {} - loss {} - time: {}'.format(epoch, loss.item(), etime))
        stime = etime
        acc = evaluate(model, features, labels, val_mask)
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), 'graphsage-best-model.pkl')
            print('== Epoch {} - Best val acc: {:.3f}'.format(epoch, acc))
    model.load_state_dict(torch.load('graphsage-best-model.pkl'))
    with torch.no_grad():
        model.eval()
        output = model(features)
        output = output[test_mask]
        labels = labels[test_mask]
        micro, macro = f1(output, labels)
        print('Test micro-macro: {:.3f}\t{:.3f}'.format(micro, macro))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=64,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="mean",
                        help="mean or pool")
    parser.add_argument('--init', type=str, default="ori", help="Features initialization method")
    parser.add_argument('--feature_size', type=int, default=5, help="Features dimension")
    parser.add_argument('--norm_features', action='store_true', help="norm features by standard scaler.")
    parser.add_argument('--seed', type=int, default=40)
    args = parser.parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
