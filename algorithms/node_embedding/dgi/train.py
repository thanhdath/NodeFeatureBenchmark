import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from .dgi import DGI


class DGIAPI():
    def __init__(self, data, features, dropout=0, lr=1e-3, epochs=300,
                 hidden=512, layers=1, weight_decay=0., patience=20, self_loop=False, cuda=True):
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.hidden = hidden
        self.layers = layers
        self.weight_decay = weight_decay
        self.patience = patience
        self.self_loop = self_loop
        self.cuda = cuda
        self.data = data
        self.graph = self.preprocess_graph(data.graph)
        self.features = torch.FloatTensor(features)

        self.dgi = DGI(self.features.shape[1], self.hidden, self.layers, nn.PReLU(
            self.hidden), self.dropout)

    def preprocess_graph(self, graph):
        # graph preprocess and calculate normalization factor
        if graph.__class__.__name__ != "DGLGraph":
            if self.self_loop:
                graph.remove_edges_from(graph.selfloop_edges())
                graph.add_edges_from(
                    zip(graph.nodes(), graph.nodes()))
            g = DGLGraph(graph)
            return g
        return graph

    def train(self):
        features = self.features
        if self.cuda:
            features = self.features.cuda()
        # create DGI model
        dgi = self.dgi
        if self.cuda:
            dgi.cuda()
        dgi_optimizer = torch.optim.Adam(dgi.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        # train deep graph infomax
        cnt_wait = 0
        best = 1e9
        best_t = 0
        dur = []
        for epoch in range(self.epochs):
            dgi.train()
            if epoch >= 3:
                t0 = time.time()

            dgi_optimizer.zero_grad()
            loss = dgi(features, self.graph)
            loss.backward()
            dgi_optimizer.step()

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(dgi.state_dict(), 'dgi-best-model.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == self.patience:
                print('Early stopping!')
                break

            if epoch >= 3:
                dur.append(time.time() - t0)

            if epoch % 20 == 0:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f}".format(
                    epoch, np.mean(dur), loss.item()))
        print('Loading {}th epoch'.format(best_t))
        dgi.load_state_dict(torch.load('dgi-best-model.pkl'))
        embeds = dgi.encoder(features, self.graph, corrupt=False)
        embeds = embeds.detach()
        return embeds

    def get_embeds(self, features, g):
        g = self.preprocess_graph(g)
        return self.dgi.encoder(features, g, corrupt=False)
