import os
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .models import get_model
from .utils import sgc_precompute, preprocess_citation, sparse_mx_to_torch_sparse_tensor
from .metrics import f1, accuracy
from time import perf_counter
from sklearn.preprocessing import MultiLabelBinarizer
import pdb

class SGC(nn.Module):
    def __init__(self, G, labels, hidden=0, dropout=0, 
        degree=2, epochs=1, weight_decay=5e-6, lr=0.2, cuda=True,
        train_ratio=0.8, trainable_features=False):
        super(SGC, self).__init__()

        self.G = G
        self._convert_labels_to_binary(labels)
        self.n_classes = self.labels.shape[1]
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
        self.train_ratio = train_ratio

        # precompute features
        self.adj, self.features = preprocess_citation(self.adj, self.features, "AugNormAdj")
        self.adj = sparse_mx_to_torch_sparse_tensor(self.adj).float()
        self.features = torch.FloatTensor(self.features).float()
        if trainable_features:
            self.features = nn.Parameter(self.features)
        print("Trainable features: ", self.features.requires_grad)
        self.features, precompute_time = sgc_precompute(self.features, self.adj, degree)
        self.W = nn.Linear(self.features.shape[1], self.n_classes)
        if cuda:
            self.features = self.features.cuda()
            self.W = self.W.cuda()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_regression()
        

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
        self.degrees = np.asarray(self.adj.sum(axis=0))[0]
        self.adj = self.adj + self.adj.T.multiply(self.adj.T > self.adj) - self.adj.multiply(self.adj.T > self.adj)

    def _to_multilabel(self, labels):
        mlb = MultiLabelBinarizer()
        return mlb.fit_transform(labels)

    def train_regression(self):
        n_train = int(len(self.features)*self.train_ratio)
        indices = np.random.permutation(np.arange(len(self.features)))
        train_features = self.features[indices][:n_train]
        train_labels = self.labels[indices][:n_train]
        test_features = self.features[indices][n_train:]
        test_labels = self.labels[indices][n_train:]
        #  train_features, train_labels, test_features, test_labels

        optimizer = optim.Adam(self.W.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        t = perf_counter()
        if self.cuda:
            train_labels = train_labels.cuda()
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
        train_time = perf_counter()-t
        print('Train time: {:.3f}'.format(train_time))

        if self.cuda:
            train_labels = train_labels.cpu() 
            test_labels = test_labels.argmax(dim=1)
        
        with torch.no_grad():
            self.W.eval()
            output = self.W(test_features)
            # acc = accuracy(output, test_labels.argmax(dim=1))
            # print('Test acc', acc)
            micro, macro = f1(output, test_labels)
            print('Test micro-macro: {:.3f}\t{:.3f}'.format(micro, macro))

    # def get_vectors(self):
    #     with torch.no_grad():
    #         self.eval()
    #         output = self.W(self.features)
    #         vectors = {}
    #         for i, node in enumerate(self.G.nodes()):
    #             vectors[str(node)] = output[i].cpu().numpy()
    #         return vectors
