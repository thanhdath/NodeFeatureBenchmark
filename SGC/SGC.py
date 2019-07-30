import os
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .models import get_model
from .utils import sgc_precompute, preprocess_citation, sparse_mx_to_torch_sparse_tensor
# from .metrics import accuracy
from time import perf_counter
from sklearn.preprocessing import MultiLabelBinarizer
from graphsage.prediction import BipartiteEdgePredLayer

def fixed_unigram_candidate_sampler(num_sampled, unique, range_max, distortion, unigrams):
    weights = unigrams**distortion
    prob = weights/weights.sum()
    sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
    return sampled

class SGC(nn.Module):
    def __init__(self, dim_size, G, hidden=0, dropout=0, 
        num_walks=20, walk_length=10, degree=2, neg_sample_size=20,
        epochs=1, batch_size=32, max_iter=1000, weight_decay=5e-6, lr=0.2, cuda=True):
        super(SGC, self).__init__()

        self.dim_size = dim_size
        self.G = G
        self.id2idx = {}
        for i, node in enumerate(self.G.nodes()):
            self.id2idx[node] = i
        self._process_adj()
        self.features = []
        for node in G.nodes():
            self.features.append(G.node[node]['feature'])
        self.features = np.array(self.features)

        self.num_walks = num_walks
        self.walk_length = walk_length
        self.batch_size = batch_size
        self.max_iter = max_iter
        if self.max_iter is None:
            self.max_iter = 1e9
        self.epochs = epochs 
        self.weight_decay = weight_decay
        self.lr = lr
        self.dropout = dropout
        self.cuda = cuda
        self.neg_sample_size = neg_sample_size
        self.link_pred_layer = BipartiteEdgePredLayer(is_normalized_input=True)

        # precompute features
        self.adj, self.features = preprocess_citation(self.adj, self.features, "AugNormAdj")
        self.adj = sparse_mx_to_torch_sparse_tensor(self.adj).float()
        self.features = torch.FloatTensor(self.features).float()
        self.features, precompute_time = sgc_precompute(self.features, self.adj, degree)
        self.W = nn.Linear(self.features.shape[1], dim_size)
        if cuda:
            self.features = self.features.cuda()
            self.W = self.W.cuda()
        train_time = self.train_unsupervised()

    def _process_adj(self):
        self.adj = nx.adjacency_matrix(self.G)
        self.degrees = np.asarray(self.adj.sum(axis=0))[0]
        self.adj = self.adj + self.adj.T.multiply(self.adj.T > self.adj) - self.adj.multiply(self.adj.T > self.adj)
        # self.adj = self.adj.tocoo()
        # self.adj = torch.sparse.FloatTensor(
        #     torch.LongTensor(np.vstack((self.adj.row, self.adj.col))),
        #     torch.FloatTensor(self.adj.data),
        #     torch.Size(self.adj.shape)
        # )

    def _run_random_walks(self):
        nodes = self.G.nodes()
        pairs = []
        print_every = len(nodes) // 2
        for count, node in enumerate(nodes):
            if self.G.degree(node) == 0:
                continue
            for i in range(self.num_walks):
                curr_node = node
                for j in range(self.walk_length):
                    next_node = np.random.choice([x for x in self.G.neighbors(curr_node)])
                    # self co-occurrences are useless
                    if curr_node != node:
                        pairs.append((node,curr_node))
                    curr_node = next_node
            if count % print_every == 0:
                print("Done walks for", count, "nodes")
        return pairs

    def _to_multilabel(self, labels):
        mlb = MultiLabelBinarizer()
        return mlb.fit_transform(labels)

    def loss(self, inputs1, inputs2):
        batch_size = inputs1.shape[0]
        outputs1, outputs2, neg_outputs  = self.forward(inputs1, inputs2)        
        loss = self.link_pred_layer.loss(outputs1, outputs2, neg_outputs) / batch_size
        return loss, outputs1, outputs2, neg_outputs

    def forward(self, inputs1, inputs2):
        neg = fixed_unigram_candidate_sampler(
            num_sampled=self.neg_sample_size,
            unique=False,
            range_max=self.adj.shape[0],
            distortion=0.75,
            unigrams=self.degrees
        )
        # neg = torch.LongTensor(neg)
        neg_features = self.features[neg]
        inputs1_features = self.features[[self.id2idx[x] for x in inputs1]]
        inputs2_features = self.features[[self.id2idx[x] for x in inputs2]]
        outputs1 = self.W(inputs1_features)
        outputs2 = self.W(inputs2_features)
        neg_outputs = self.W(neg_features)

        outputs1 = F.normalize(outputs1, dim=1)
        outputs2 = F.normalize(outputs2, dim=1)
        neg_outputs = F.normalize(neg_outputs, dim=1)
        return outputs1, outputs2, neg_outputs


    def train_unsupervised(self):
        batch_size = self.batch_size

        # edges = np.array(self._run_random_walks())
        edges = np.array(self.G.edges())
        # edges = torch.LongTensor(edges)
        optimizer = optim.Adam(self.W.parameters(), lr=self.lr,
                            weight_decay=self.weight_decay)
        t = perf_counter()
        n_iters = len(edges) // batch_size
        indices = np.arange(len(edges))
        print_every = n_iters//5
        total_iter = 0
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            edges = edges[indices]
            for iter in range(n_iters):
                self.train()
                optimizer.zero_grad()

                batch_edges = edges[batch_size*iter:batch_size*(iter+1)]
                loss_train, outputs1, outputs2, neg_outputs = self.loss(batch_edges[:,0], batch_edges[:,1])
                loss_train.backward()
                optimizer.step()
                if iter % print_every == 0:
                    print('Epoch {} - Iter {} - loss {}'.format(epoch, iter, loss_train.item()))
                total_iter += 1
                if total_iter > self.max_iter:
                    break
            if total_iter > self.max_iter:
                    break
        train_time = perf_counter()-t
        return train_time

    def get_vectors(self):
        with torch.no_grad():
            self.eval()
            output = self.W(self.features)
            vectors = {}
            for i, node in enumerate(self.G.nodes()):
                vectors[str(node)] = output[i].cpu().numpy()
            return vectors

def init_features_SGC(G, dim=32):
    prep_dict = {}
    for idx, node in enumerate(G.nodes()):
        prep_dict[node] = np.array([G.degree(node)]+[1.]*(dim-1))
    return prep_dict
