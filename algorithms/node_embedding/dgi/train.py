import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from .dgi import DGI
import itertools
import os
import tensorboardX

class DGIAPI():
    def __init__(self, data, features, dropout=0, lr=1e-3, epochs=300,
                 hidden=512, layers=1, weight_decay=0., patience=60, self_loop=False, cuda=True,
                 learnable_features=False, suffix="", load_model=None):
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.hidden = hidden
        self.layers = layers
        self.weight_decay = weight_decay
        self.patience = patience
        self.self_loop = self_loop
        self.cuda = cuda
        self.suffix = suffix
        self.data = data
        self.graph = self.preprocess_graph(data.graph)
        self.load_model = load_model

        if not learnable_features:
            self.features = torch.FloatTensor(features)
        else:
            print("Learnable features")
            self.features_embedding = nn.Embedding(features.shape[0], features.shape[1])
            self.features_embedding.weight = nn.Parameter(features)
            self.features = self.features_embedding.weight

        self.dgi = DGI(self.features.shape[1], self.hidden, self.layers, nn.PReLU(
            self.hidden), self.dropout)
        if not learnable_features:
            self.dgi_optimizer = torch.optim.Adam(self.dgi.parameters(),
                                                  lr=self.lr,
                                                  weight_decay=self.weight_decay)
        else:
            self.dgi_optimizer = torch.optim.Adam(
                itertools.chain(self.dgi.parameters(),
                                self.features_embedding.parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay)

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

        # train deep graph infomax
        cnt_wait = 0
        best = 1e9
        best_t = 0
        dur = []
        if self.load_model is not None:
            self.dgi.load_state_dict(torch.load(self.load_model))
            from_data = self.load_model.replace(".pkl", "").replace("dgi-best-model-", "")
            best_model_name = 'dgi-best-model-{}-from-{}.pkl'.format(
                self.suffix, from_data)
            print("Load pretrained model ", self.load_model)
            writer = tensorboardX.SummaryWriter("summary/"+best_model_name.replace("dgi-best-model-", "").replace(".pkl", ""))
        else:
            best_model_name = 'dgi-best-model-{}.pkl'.format(self.suffix)
        print("Save best model to ", best_model_name)

        for epoch in range(self.epochs):
            dgi.train()
            if epoch >= 3:
                t0 = time.time()

            self.dgi_optimizer.zero_grad()
            loss = dgi(features, self.graph)
            loss.backward()
            self.dgi_optimizer.step()

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(dgi.state_dict(), best_model_name)
            else:
                cnt_wait += 1

            if cnt_wait == self.patience:
                print('Early stopping!')
                break

            if epoch >= 3:
                dur.append(time.time() - t0)

            if epoch % 1 == 0:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss {}".format(epoch, np.mean(dur), loss.item()))
            
            if self.load_model is not None:
                writer.add_scalar("dgi loss", loss, epoch)
        print('Loading {}th epoch'.format(best_t))
        if os.path.isfile(best_model_name):
            dgi.load_state_dict(torch.load(best_model_name))
        # os.remove(best_model_name)
        embeds = dgi.encoder(features, self.graph, corrupt=False)
        embeds = embeds.detach()
        return embeds

    def get_embeds(self, features, g):
        if self.cuda:
            self.dgi.cpu()
        g = self.preprocess_graph(g)
        embeds = self.dgi.encoder(features, g, corrupt=False)
        if self.cuda:
            self.dgi.cuda()
        return embeds
