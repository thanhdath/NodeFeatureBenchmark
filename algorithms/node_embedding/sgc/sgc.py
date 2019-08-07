"""
This code was modified from the GCN implementation in DGL examples.
Simplifying Graph Convolutional Networks
Paper: https://arxiv.org/abs/1902.07153
Code: https://github.com/Tiiiger/SGC
SGC implementation in DGL.
"""
import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

class SGC():
    def __init__(self, data, features, degree=2, cuda=True):
        """
        DGL default load data
        """
        self.cuda = cuda
        self.degree = degree
        self.data = data
        self.graph = self.preprocess_graph(data)
        self.features = torch.FloatTensor(features)
    
    def preprocess_graph(self, data):
        # graph preprocess and calculate normalization factor
        g = DGLGraph(data.graph)
        return g
 
    def sgc_compute(self):
        h = self.features
        if self.cuda:
            h = h.cuda()
        # precomputing message passing
        for _ in range(self.degree):
            # normalization by square root of src degree
            h = h * self.graph.ndata['norm']
            self.graph.ndata['h'] = h
            self.graph.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
            h = self.graph.ndata.pop('h')
            # normalization by square root of dst degree
            h = h * self.graph.ndata['norm']
        return h 
    
    def train(self):
        degs = self.graph.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        if self.cuda:
            norm = norm.cuda()
        self.graph.ndata['norm'] = norm.unsqueeze(1)

        # create SGC model
        embs = self.sgc_compute()
        return embs
