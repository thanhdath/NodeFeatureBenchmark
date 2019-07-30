import torch
import torch.nn as nn
from torch.autograd import Variable
from graphsage.inits import Initializer

import numpy as np

#Class for preprocessing feature data for supervised train

class IdentityPrep(nn.Module):
    def __init__(self, input_dim, n_nodes=None):
        """ Example of preprocessor -- doesn't do anything """
        super(IdentityPrep, self).__init__()
        self.input_dim = input_dim
    
    @property
    def output_dim(self):
        return self.input_dim
    
    def forward(self, ids, feats, hop_idx=0):
        return feats 

class NodeEmbeddingPrep(nn.Module):
    def __init__(self, input_dim, n_nodes, identity_dim=64, weight=None):
        """ adds node embedding """
        super(NodeEmbeddingPrep, self).__init__()

        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.identity_dim = identity_dim
        self.embedding = nn.Embedding(num_embeddings=n_nodes + 1, embedding_dim=identity_dim)
        if weight is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(weight))
            self.embedding.weight.requires_grad = True

    @property
    def output_dim(self):
        if self.input_dim:
            return self.input_dim + self.identity_dim
        else:
            return self.identity_dim
    
    def forward(self, ids, feats, hop_idx):
        if hop_idx > 0:
            embs = self.embedding(ids)
        else:
            # Don't look at node's own embedding for prediction, or you'll probably overfit a lot
            embs = self.embedding(Variable(ids.clone().data.zero_() + self.n_nodes))
        
        # embs = self.fc(embs)
        if self.input_dim:
            return torch.cat([feats, embs], dim=1)
        else:
            return embs

prep_lookup = {
    "use_original_features" : IdentityPrep,
    "use_identity_features" : NodeEmbeddingPrep,    
}

