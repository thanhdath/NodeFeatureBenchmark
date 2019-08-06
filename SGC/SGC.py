import os
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .models import get_model
from torch.autograd import Variable
from .utils import sgc_precompute, preprocess_citation, sparse_mx_to_torch_sparse_tensor
from .metrics import f1, accuracy
from time import perf_counter
from sklearn.preprocessing import MultiLabelBinarizer
import pdb
from .normalization import fetch_normalization
from utils import split_train_test


class SGC(nn.Module):
    def __init__(self, G, labels, hidden=0, dropout=0,
                 degree=2, epochs=100, weight_decay=5e-6, lr=0.2, cuda=True,
                 ratio=[.7, .1, .2], trainable_features=False):
        super(SGC, self).__init__()

        self.G = G
        self._convert_labels_to_binary(labels)
        self.id2idx = {}
        for i, node in enumerate(self.G.nodes()):
            self.id2idx[node] = i
        self._process_adj()
        self.features = []
        for node in G.nodes():
            self.features.append(G.node[node]['feature'])
        self.features = np.array(self.features)
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.lr = lr
        self.dropout = dropout
        self.cuda = cuda
        self.ratio = ratio

        # precompute features and adj
        adj_normalizer = fetch_normalization("AugNormAdj")
        self.adj = adj_normalizer(adj)

        self.adj = sparse_mx_to_torch_sparse_tensor(self.adj).float()
        self.features = torch.FloatTensor(self.features).float()
        self.features, _ = sgc_precompute(self.features, self.adj, degree)
        self._split_train_test()
        self.W = nn.Linear(self.features.shape[1], self.n_classes)
        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_regression()

    def _convert_labels_to_binary(self, labels):
        labels_arr = []
        for node in self.G.nodes():
            labels_arr.append(labels[str(node)])
        self.binarizer = MultiLabelBinarizer(sparse_output=False)
        self.binarizer.fit(labels_arr)
        self.labels = self.binarizer.transform(labels_arr)
        self.labels = torch.LongTensor(self.labels).argmax(dim=1)
        self.n_classes = int(self.labels.max()) + 1

    def _process_adj(self):
        self.adj = nx.adjacency_matrix(self.G)
        self.degrees = np.asarray(self.adj.sum(axis=0))[0]
        self.adj = self.adj + \
            self.adj.T.multiply(self.adj.T > self.adj) - \
            self.adj.multiply(self.adj.T > self.adj)

    def _split_train_test(self):
        n_train = int(len(self.features)*self.ratio[0])
        n_val = int(len(self.features)*self.ratio[1])
        indices = np.random.permutation(np.arange(len(self.features)))
        self.train_indices = indices[:n_train]
        self.val_indices = indices[n_train:n_val+n_train]
        self.test_indices = indices[n_train+n_val:]

    def train_regression(self):
        train_features = self.features[self.train_indices]
        train_labels = self.labels[self.train_indices]
        val_features = self.features[self.val_indices]
        val_labels = self.labels[self.val_indices]
        test_features = self.features[self.test_indices]
        test_labels = self.labels[self.test_indices]

        optimizer = optim.Adam(self.W.parameters(),
                               lr=self.lr, weight_decay=self.weight_decay)
        t = perf_counter()
        if self.cuda:
            train_labels = train_labels.cuda()
            train_features = train_features.cuda()
            val_labels = val_labels.cuda()
            val_features = val_features.cuda()
            self.W.cuda()
        best_val_acc = 0
        for epoch in range(self.epochs): 
            self.W.train()
            optimizer.zero_grad()
            output = self.W(train_features)
            loss_train = F.cross_entropy(output, train_labels)
            # loss_train = self.loss_fn(output, train_labels) # multiple classes
            loss_train.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print('Epoch {} - loss {}'.format(epoch, loss_train.item()))
            with torch.no_grad():
                self.W.eval()
                output = self.W(val_features)
                acc = accuracy(output, val_labels)
                if acc > best_val_acc:
                    best_val_acc = acc
                    torch.save(self.W.state_dict(), 'sgc-best-model.pkl')
                    print('== Epoch {} - Best val acc: {:.3f}'.format(epoch, acc.item()))
        train_time = perf_counter()-t
        print('Train time: {:.3f}'.format(train_time))
        self.W.load_state_dict(torch.load('sgc-best-model.pkl'))
        if self.cuda:
            train_labels = train_labels.cpu()
            train_features = train_features.cpu()
            val_labels = val_labels.cpu()
            val_features = val_features.cpu()
            test_labels = test_labels.cuda()
            test_features = test_features.cuda()

        with torch.no_grad():
            self.W.eval()
            output = self.W(test_features)
            micro, macro = f1(output, test_labels)
            print('Test micro-macro: {:.3f}\t{:.3f}'.format(micro, macro))
