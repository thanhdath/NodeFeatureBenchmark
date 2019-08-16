"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import time
import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from utils import f1, accuracy


class Aggregator(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, bias=True):
        super(Aggregator, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats,
                                bias=bias)  # (F, EF) or (2F, EF)
        self.activation = activation
        nn.init.xavier_uniform_(
            self.linear.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, node):
        nei = node.mailbox['m']  # (B, N, F)
        h = node.data['h']  # (B, F)
        h = self.concat(h, nei, node, node.data['degs'])  # (B, F) or (B, 2F)
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
    def __init__(self, in_feats, out_feats, activation, bias):
        super(MeanAggregator, self).__init__(
            in_feats, out_feats, activation, bias)

    def concat(self, h, nei, nodes, degs):
        # degs = g.in_degrees(nodes.nodes()).float()
        if h.is_cuda:
            degs = degs.cuda(h.device)
        concatenate = torch.cat((nei, h.unsqueeze(1)), 1)
        concatenate = torch.sum(concatenate, 1) / degs.unsqueeze(1)
        return concatenate  # (B, F)


class PoolingAggregator(Aggregator):
    def __init__(self, in_feats, out_feats, activation, bias):  # (2F, F)
        super(PoolingAggregator, self).__init__(
            in_feats*2, out_feats, activation, bias)
        self.mlp = PoolingAggregator.MLP(
            in_feats, in_feats, F.relu, False, True)

    def concat(self, h, nei, nodes, degs):
        nei = self.mlp(nei)  # (B, F)
        concatenate = torch.cat((nei, h), 1)  # (B, 2F)
        return concatenate

    class MLP(nn.Module):
        def __init__(self, in_feats, out_feats, activation, dropout, bias):  # (F, F)
            super(PoolingAggregator.MLP, self).__init__()
            self.linear = nn.Linear(in_feats, out_feats, bias=bias)  # (F, F)
            self.dropout = nn.Dropout(p=dropout)
            self.activation = activation
            nn.init.xavier_uniform_(
                self.linear.weight, gain=nn.init.calculate_gain('relu'))

        def forward(self, nei):
            nei = self.dropout(nei)  # (B, N, F)
            nei = self.linear(nei)
            if self.activation:
                nei = self.activation(nei)
            max_value = torch.max(nei, dim=1)[0]  # (B, F)
            return max_value


class GraphSAGELayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 aggregator_type,
                 bias=True,
                 ):
        super(GraphSAGELayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        if aggregator_type == "pooling":
            self.aggregator = PoolingAggregator(
                in_feats, out_feats, activation, bias)
        elif aggregator_type == "mean":
            self.aggregator = MeanAggregator(
                in_feats, out_feats, activation, bias)
        else:
            raise NotImplementedError

    def forward(self, h, g):
        h = self.dropout(h)
        g.ndata['h'] = h
        g.ndata['degs'] = g.in_degrees(g.nodes()).float()
        g.update_all(fn.copy_src(src='h', out='m'), self.aggregator)
        h = g.ndata.pop('h')
        return h


class GraphSAGE(nn.Module):
    def __init__(self,
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
        self.layers.append(GraphSAGELayer(
            in_feats, n_hidden, activation, dropout, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphSAGELayer(
                n_hidden, n_hidden, activation, dropout, aggregator_type))
        # output layer
        self.layers.append(GraphSAGELayer(
            n_hidden, n_classes, None, dropout, aggregator_type))

    def forward(self, features, g):
        h = features
        for layer in self.layers:
            h = layer(h, g)
        return h


class GraphsageInductive():
    def __init__(self, train_data, val_data, test_data,
                 train_features, val_features, test_features,
                 dropout=0.5, cuda=True, lr=1e-2,
                 epochs=200, hidden=16, layers=2, weight_decay=5e-4, aggregator="mean"):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_features = train_features
        self.val_features = val_features
        self.test_features = test_features

        self.train_graph = self.preprocess_graph(train_data.graph)
        self.val_graph = self.preprocess_graph(val_data.graph)
        self.test_graph = self.preprocess_graph(test_data.graph)

        self.dropout = dropout
        self.cuda = cuda
        self.lr = lr
        self.epochs = epochs
        self.hidden = hidden
        self.layers = layers
        self.weight_decay = weight_decay
        self.aggregator = aggregator
        self.multiclass = train_data.multiclass
        if self.multiclass:
            print("Train graphsage with multiclass")

        self.model = GraphSAGE(self.train_features.shape[1],
                               self.hidden,
                               self.train_data.n_classes,
                               self.layers,
                               F.relu,
                               self.dropout,
                               self.aggregator
                               )

    def preprocess_graph(self, graph):
        if graph.__class__.__name__ != "DGLGraph":
            g = DGLGraph(graph)
            return g
        return graph

    def train(self):
        train_features = self.train_features
        val_features = self.val_features
        test_features = self.test_features
        train_labels = self.train_data.labels
        val_labels = self.val_data.labels
        test_labels = self.test_data.labels
        model = self.model

        if self.cuda:
            model.cuda()
            train_features = train_features.cuda()
            train_labels = train_labels.cuda()
            val_features = val_features.cuda()
            val_labels = val_labels.cuda()

        if self.multiclass:
            loss_fcn = torch.nn.BCEWithLogitsLoss()
        else:
            loss_fcn = torch.nn.CrossEntropyLoss()

        # use optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_val_acc = 0
        npt = 0
        max_patience = 3
        for epoch in range(self.epochs):
            stime = time.time()
            model.train()
            # forward
            logits = model(train_features, self.train_graph)
            loss = loss_fcn(logits, train_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            etime = time.time() - stime
            if epoch % 20 == 0:
                print('Epoch {} - loss {} - time: {}'.format(epoch, loss.item(), etime))
                # evaluate too slow
                model.eval()
                with torch.no_grad():
                    logits = model(val_features, self.val_graph)
                    acc = accuracy(logits, val_labels,
                                   multiclass=self.multiclass)
                    if acc > best_val_acc:
                        best_val_acc = acc
                        torch.save(model.state_dict(),
                                   'graphsage-best-model.pkl')
                        print('== Epoch {} - Best val acc: {:.3f}'.format(epoch, acc))
                        npt = 0
                    else:
                        npt += 1
                    if npt > max_patience:
                        print("Early stopping")
                        break
        model.load_state_dict(torch.load('graphsage-best-model.pkl'))
        with torch.no_grad():
            model.eval()
            output = model(test_features, self.test_graph)
            micro, macro = f1(output, test_labels, multiclass=self.multiclass)
            print('Test micro-macro: {:.3f}\t{:.3f}'.format(micro, macro))
