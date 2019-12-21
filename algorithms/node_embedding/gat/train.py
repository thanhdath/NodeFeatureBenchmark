"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
Compared with the original paper, this code does not implement
early stopping.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from .gat import GAT
from utils import f1, accuracy
import scipy.sparse as sp
import numpy as np
import dgl
import os, sys
import networkx

def evaluate(model, features, labels, mask, multiclass=False):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels, multiclass=multiclass)

class GATAPI():
    def __init__(self, data, features, 
        num_heads=8, num_layers=1, num_out_heads=1, num_hidden=8,
        in_drop=.6, attn_drop=.6, alpha=.2, residual=True, epochs=200,
        self_loop=False, cuda=True,
        learnable_features=False, suffix="", load_model=None):
        self.self_loop = self_loop
        self.data = data
        self.graph = self.preprocess_graph(data.graph)
        self.features = torch.FloatTensor(features) 
        self.cuda = cuda
        self.epochs = epochs
        self.suffix = suffix
        self.validation_steps = 1
        self.multiclass = data.multiclass

        heads = ([num_heads] * num_layers) + [num_out_heads]
        self.model = GAT(self.graph,
                    num_layers,
                    self.features.shape[1],
                    num_hidden,
                    self.data.n_classes,
                    heads,
                    F.elu,
                    in_drop,
                    attn_drop,
                    alpha,
                    residual)

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
        model = self.model
        features = self.features
        labels = self.data.labels
        train_mask = self.data.train_mask 
        val_mask = self.data.val_mask 
        test_mask = self.data.test_mask
        if self.cuda:
            model = model.cuda()
            features = features.cuda()
            labels = data.labels.cuda()
            train_mask = train_mask.cuda()
            val_mask = val_mask.cuda()
            test_mask = test_mask.cuda()
        
        if self.multiclass:
            loss_fcn = torch.nn.BCEWithLogitsLoss()
        else:
            loss_fcn = torch.nn.CrossEntropyLoss()
        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        best_model_name = 'gat-best-model-{}.pkl'.format(self.suffix)
        best_val_acc = 0
        for epoch in range(self.epochs):
            stime = time.time()
            model.train()
            # forward
            logits = model(features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            etime = time.time() - stime

            if epoch % self.validation_steps == 0:
                # train_acc = accuracy(logits[train_mask], labels[train_mask])
                val_acc = evaluate(model, features, labels, val_mask, multiclass=self.multiclass)
                print('Epoch {} - loss {} - time: {}'.format(epoch, loss.item(), etime))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc 
                    torch.save(model.state_dict(), best_model_name)
                    print('== Epoch {} - Best val acc: {:.3f}'.format(epoch, val_acc))
        if os.path.isfile(best_model_name):
            model.load_state_dict(torch.load(best_model_name))

        with torch.no_grad():
            model.eval()
            output = model(features)
            output = output[test_mask]
            labels = labels[test_mask]
            micro, macro = f1(output, labels, multiclass=self.multiclass)
            print('Test micro-macro: {:.3f}\t{:.3f}'.format(micro, macro))
