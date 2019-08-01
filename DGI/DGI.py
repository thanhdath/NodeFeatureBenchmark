import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
import os 
import glob
from .models import DGI, LogReg
from .utils import process
from SGC.metrics import f1

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class DGIAPI():
    def __init__(self, G, labels, batch_size=1,epochs = 10000,
        patience = 20,lr = 0.001,l2_coef = 0.0,drop_prob = 0.0,
        hid_units = 512, sparse = True, nonlinearity = 'prelu', ratio=[.7, .1, .2],
        cuda=True):
        print("Warning: GAT currently not support for multiple labels.")
        self.G = G 
        self.labels = labels
        self.batch_size=batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.l2_coef=l2_coef
        self.drop_prob=drop_prob
        self.hid_units=hid_units
        self.sparse=sparse
        self.nonlinearity=nonlinearity
        self.ratio = ratio
        self.cuda = cuda

        self._process_adj()
        self._process_features()
        self._convert_labels_to_binary()
        self._split_train_val()
        self.model = DGI(self.features.shape[1], 
            self.hid_units, self.nonlinearity)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, 
            weight_decay=self.l2_coef)
        self.train()

    def _process_adj(self):
        adj = nx.to_scipy_sparse_matrix(self.G)
        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        if self.sparse:
            self.sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        else:
            adj = (adj + sp.eye(adj.shape[0])).todense()
            self.adj = torch.FloatTensor(adj)

    def _process_features(self):
        features = []
        for node in self.G.nodes():
            features.append(self.G.node[node]['feature'])
        features = np.array(features)
        features = normalize_features(features)
        self.features = torch.FloatTensor(features)

    def _convert_labels_to_binary(self):
        labels = self.labels
        labels_arr = []
        for node in self.G.nodes():
            labels_arr.append(labels[str(node)])
        self.binarizer = MultiLabelBinarizer(sparse_output=False)
        self.binarizer.fit(labels_arr)
        self.labels = self.binarizer.transform(labels_arr)
        self.labels = torch.LongTensor(self.labels).argmax(dim=1)
        self.n_classes = int(self.labels.max() + 1)

    def _split_train_val(self):
        indices = np.random.permutation(np.arange(len(self.features)))
        n_train = int(len(self.features)*self.ratio[0])
        n_val = int(len(self.features)*self.ratio[1])
        self.idx_train = torch.LongTensor(indices[:n_train])
        self.idx_val = torch.LongTensor(indices[n_train:n_train+n_val])
        self.idx_test = torch.LongTensor(indices[n_train+n_val:])

    def train(self):
        if self.cuda:
            self.model.cuda()
            self.features = self.features.cuda()
            if self.sparse:
                sp_adj = self.sp_adj.cuda()
            else:
                adj = self.adj.cuda()
            self.labels = self.labels.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0
        nb_nodes = self.features.shape[0]
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()

            idx = np.random.permutation(nb_nodes)
            shuf_fts = self.features[idx, :]

            lbl_1 = torch.ones(self.batch_size, nb_nodes)
            lbl_2 = torch.zeros(self.batch_size, nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)

            if self.cuda:
                shuf_fts = shuf_fts.cuda()
                lbl = lbl.cuda()
            
            logits = self.model(self.features, shuf_fts, sp_adj if self.sparse else adj, self.sparse, None, None, None) 

            loss = b_xent(logits, lbl)

            if epoch % 20 == 0:
                print('Epoch {} - Loss: {}'.format(epoch, loss.item()))

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(self.model.state_dict(), 'best_dgi.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == self.patience:
                print('Early stopping!')
                break

            loss.backward()
            self.optimizer.step()

        print('Loading {}th epoch'.format(best_t))
        self.model.load_state_dict(torch.load('best_dgi.pkl'))

        embeds, _ = self.model.embed(self.features, sp_adj if self.sparse else adj, self.sparse, None)
        train_embs = embeds[0, self.idx_train]
        # val_embs = embeds[0, self.idx_val]
        test_embs = embeds[0, self.idx_test]

        train_lbls = self.labels[self.idx_train]
        # val_lbls = self.labels[self.idx_val]
        test_lbls = self.labels[self.idx_test]

        tot = torch.zeros(1)
        tot = tot.cuda()

        f1s = []
        for _ in range(10):
            log = LogReg(self.hid_units, self.n_classes)
            opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
            log.cuda()
            for _ in range(100):
                log.train()
                opt.zero_grad()
                logits = log(train_embs)
                loss = xent(logits, train_lbls)
                loss.backward()
                opt.step()
            logits = log(test_embs)
            micro, macro = f1(logits, test_lbls)
            f1s.append([micro, macro])
        f1s = np.array(f1s)
        micro = f1s[:,0].mean()
        macro = f1s[:,1].mean()
        print('Test micro-macro: {:.3f}\t{:.3f}'.format(micro, macro))

