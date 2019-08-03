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
from sklearn.metrics import accuracy_score
import pdb

class SGC(nn.Module):
    def __init__(self, train_G, val_G, labels, hidden=0, dropout=0,
                 degree=2, epochs=100, weight_decay=5e-6, lr=0.2, cuda=True, trainable_features=False):
        """
        labels: dict, key = nodeid, value = binary encoder. multiple labels, binary encoder
        """
        super(SGC, self).__init__()

        self.train_G = train_G 
        self.val_G = val_G
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.lr = lr
        self.dropout = dropout
        self.cuda = cuda
        self.degree = degree

        self.n_classes = len(list(labels.values())[0])

        self.train_adj, self.train_features = self._process_adj_and_features(self.train_G)
        self.train_labels = self._process_labels(self.train_G, labels)
        self.val_adj, self.val_features = self._process_adj_and_features(self.val_G)
        self.val_labels = self._process_labels(self.val_G, labels)
        # 
        self.W = nn.Linear(self.train_features.shape[1], self.n_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train()

    def _process_labels(self, G, labels):
        labels_arr = np.array([labels[str(x)] for x in G.nodes()])
        labels_arr = torch.FloatTensor(labels_arr)
        return labels_arr

    def _process_adj_and_features(self, G):
        adj = nx.to_scipy_sparse_matrix(G)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        features = np.array([G.node[node]['feature'] for node in G.nodes()])
        adj, features = preprocess_citation(adj, features, "AugNormAdj")
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        features = torch.FloatTensor(features).float()
        return adj, features

    def train(self):
        train_features, _ = sgc_precompute(self.train_features, self.train_adj, self.degree)
        val_features, _ = sgc_precompute(self.val_features, self.val_adj, self.degree)

        optimizer = optim.Adam(self.W.parameters(),
                               lr=self.lr, weight_decay=self.weight_decay)
        t = perf_counter()
        if self.cuda:
            train_labels = self.train_labels.cuda()
            train_features = train_features.cuda()
            val_labels = self.val_labels.cuda()
            val_features = val_features.cuda()
            self.W.cuda()
        best_val_acc = 0
        for epoch in range(self.epochs):
            self.W.train()
            optimizer.zero_grad()
            output = self.W(train_features)
            # loss_train = F.cross_entropy(output, train_labels)
            loss_train = self.loss_fn(output, train_labels) # multiple classes
            loss_train.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print('Epoch {} - loss {}'.format(epoch, loss_train.item()))
            with torch.no_grad():
                self.W.eval()
                output = self.W(val_features)
                output = F.sigmoid(output)
                output = torch.round(output)
                acc = accuracy_score(output.cpu().numpy(), val_labels.cpu().numpy())
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

    def test(self, test_G, labels):
        test_adj, test_features = self._process_adj_and_features(test_G)
        test_labels = self._process_labels(test_G, labels)
        test_features, _ = sgc_precompute(test_features, test_adj, self.degree)
        if self.cuda:
            test_labels = test_labels.cuda()
            test_features = test_features.cuda()
        with torch.no_grad():
            self.W.eval()
            output = self.W(test_features)
            output = F.sigmoid(output)
            output = torch.round(output)
            acc = accuracy_score(output.cpu().numpy(), val_labels.cpu().numpy())
            print('Test acc: {:.3f}'.format(acc))
            # micro, macro = f1(output, test_labels)
            # print('Test micro-macro: {:.3f}\t{:.3f}'.format(micro, macro))
        