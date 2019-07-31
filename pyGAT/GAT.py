from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import MultiLabelBinarizer
from .utils import load_data, accuracy
from .models import GAT, SpGAT
import scipy.sparse as sp
import networkx as nx

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class GATAPI():
    def __init__(self, G, labels, cuda=False, fastmode=False,
        sparse=True, epochs=10000, lr=0.005, weight_decay=5e-4,
        hidden=8, nb_heads=8, dropout=.6, alpha=.2, patience=100,
        train_val_ratio=[.7, .1, .2]):
        self.G = G
        self._convert_labels_to_binary(labels)
        self._process_adj()
        self.n_classes = self.labels.shape[1]

        self.cuda=  cuda
        self.fastmode=fastmode
        self.sparse=sparse
        self.epochs=epochs
        self.lr=lr
        self.weight_decay=weight_decay
        self.hidden=hidden
        self.nb_heads=nb_heads
        self.dropout=dropout
        self.alpha=alpha
        self.patience = patience
        self.ratio = train_val_ratio

        # 
        self._process_features()
        # 
        self.split_train_val()
        


        self.model = SpGAT(self.features.shape[1],
            nhid=self.hidden,
            nclass=self.n_classes,
            dropout=self.dropout,
            nheads=self.nb_heads,
            alpha=self.alpha)
        self.optimizer = optim.Adam(self.model.parameters(), 
                       lr=self.lr, 
                       weight_decay=self.weight_decay)
        self.train()

    def split_train_val(self):
        indices = np.random.permutation(np.arange(len(self.features)))
        n_train = int(len(self.features)*self.ratio[0])
        n_val = int(len(self.features)*self.ratio[1])
        self.idx_train = indices[:n_train]
        self.idx_val = indices[n_train:n_train+n_val]
        self.idx_test = indices[n_train+n_val:]

    def _convert_labels_to_binary(self, labels):
        labels_arr = []
        for node in self.G.nodes():
            labels_arr.append(labels[str(node)])
        self.binarizer = MultiLabelBinarizer(sparse_output=False)
        self.binarizer.fit(labels_arr)
        self.labels = self.binarizer.transform(labels_arr)
        self.labels = torch.FloatTensor(self.labels)
    
    def _process_adj(self):
        self.adj = nx.adjacency_matrix(self.G)
        self.adj = self.adj + self.adj.T.multiply(self.adj.T > self.adj) - self.adj.multiply(self.adj.T > self.adj)
        self.adj = normalize_adj(self.adj + sp.eye(self.adj.shape[0]))
        self.adj = torch.FloatTensor(np.array(self.adj.todense()))

    def _process_features(self):
        features = []
        for node in self.G.nodes():
            features.append(self.G.node[node]['feature'])
        features = np.array(features)
        features = normalize_features(features)
        self.features = torch.FloatTensor(features)

    def train_one_epoch(self, epoch):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.features, self.adj)
        loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
        acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])
        loss_train.backward()
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        output = self.model(self.features, self.adj)

        loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
        acc_val = accuracy(output[self.idx_val], self.labels[self.idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.data.item()),
            'acc_train: {:.4f}'.format(acc_train.data.item()),
            'loss_val: {:.4f}'.format(loss_val.data.item()),
            'acc_val: {:.4f}'.format(acc_val.data.item()),
            'time: {:.4f}s'.format(time.time() - t))

        return loss_val.data.item()

    def train(self):
        if self.cuda:
            self.model = self.model.cuda()
            self.features = self.features.cuda()
            self.adj = self.adj.cuda()
            self.labels = self.labels.cuda()
            
        # Train model
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = self.epochs + 1
        best_epoch = 0
        for epoch in range(self.epochs):
            loss_values.append(self.train_one_epoch(epoch))

            torch.save(self.model.state_dict(), '{}.pkl'.format(epoch))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == self.patience:
                break

            files = glob.glob('*.pkl')
            for file in files:
                epoch_nb = int(file.split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(file)
        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(file)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        # Restore best model
        print('Loading {}th epoch'.format(best_epoch))
        self.model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

        # Testing
        self.compute_test()

    def compute_test(self):
        self.model.eval()
        output = self.model(self.features, self.adj)
        loss_test = F.nll_loss(output[self.idx_test], self.labels[self.idx_test])
        acc_test = accuracy(output[self.idx_test], self.labels[self.idx_test])
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.data[0]),
            "accuracy= {:.4f}".format(acc_test.data[0]))

