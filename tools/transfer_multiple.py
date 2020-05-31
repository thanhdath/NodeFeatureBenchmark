import dgl
from dgl.data.ppi import LegacyPPIDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def collate(sample):
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels

train_dataset = LegacyPPIDataset(mode="train")
ids = np.random.permutation(len(train_dataset))
ids = [id for id in ids if train_dataset.train_graphs[id].number_of_nodes() < 1500]
Gs = [train_dataset.train_graphs[i] for i in ids[:10]]
As = [torch.FloatTensor(np.asarray(G.adjacency_matrix_scipy().todense())).to(device) for G in Gs]
Ls = [train_dataset.train_labels[i] for i in ids[:10]]
for A in As:
    A[A>0] = 1
print("Number of graphs", len(As))

from sklearn.metrics import f1_score
def compute_f1(pA, A):
    pA = pA.detach().cpu().numpy()
    pA[pA >= 0.5] = 1 
    pA[pA < 0.5] = 0
    A = A.cpu().numpy()
    f1 = f1_score(A, pA, average="micro")
    return f1

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        ori_dim = 128
        self.X1 = nn.Embedding(As[0].shape[0], ori_dim)
        self.linears = []
        self.Ms = []
        for i, A in enumerate(As): 
            l = A.shape[0]
            self.linears.append(
                nn.Sequential(
                    nn.Linear(l*(l-1)//2, ori_dim, bias=False),
                    nn.Linear(ori_dim, l*(l-1)//2, bias=False)
                )
            )
            if i>0:
                self.Ms.append(nn.Linear(As[0].shape[0], l, bias=False))
        self.linears = nn.Sequential(*self.linears)
        self.Ms = nn.Sequential(*self.Ms)
        
    def forward(self):
        X1 = self.X1.weight
        x1 = torch.pdist(X1, p=2)
        x1 = self.linears[0](x1)
        xs = [x1]
        for M, linear in zip(self.Ms, self.linears[1:]):
            x2 = M(X1.t()).t()
            x2 = torch.pdist(x2, p=2)
            x2 = linear(x2)
            xs.append(x2)
        return xs

model = Model().to(device)
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
for iter in range(1000):
    model.train()
    optim.zero_grad()
    pred_As = model()
    upper_halfs = [A[torch.triu(torch.ones_like(A), diagonal=1) == 1] for A in As]
    loss = 0
    for pred_A, upper_half in zip(pred_As, upper_halfs):
        loss += loss_fn(pred_A, upper_half)
    loss.backward()
    optim.step()
    if iter%50 == 0:
        microf11 = compute_f1(pred_As[0], upper_halfs[0])
        microf12 = compute_f1(pred_As[1], upper_halfs[1])
        print(f"Iter {iter} - loss {loss:.4f} - f1 {microf11:.3f}  {microf12:.3f}")


# gen edgelist, labels, features
all_edges = []
idx = 0
for A in As:
    src, trg = A.cpu().numpy().nonzero()
    edges = np.zeros((len(src), 2), dtype=np.int32)
    edges[:, 0] = src 
    edges[:, 1] = trg
    edges += idx
    all_edges.append(edges)
    idx += len(A)

all_features = [model.X1.weight.detach().cpu().numpy()]
for M in model.Ms:
    X = M(model.X1.weight.t()).t()
    all_features.append(X.detach().cpu().numpy())

train_edges = np.concatenate(all_edges[:-1], axis=0)
train_features = np.concatenate(all_features[:-1], axis=0)
from scipy.sparse import csr_matrix
train_adj = np.zeros((len(train_features), len(train_features)))
train_adj[train_edges[:,0], train_edges[:,1]] = 1
labels = np.concatenate(Ls[:-1], axis=0)[:,52]

import os
outdir = "data-autoencoder/ppi-mul/"
if not os.path.isdir(outdir):
    os.makedirs(outdir)
np.savez_compressed(outdir + "/train_graph.npz", graph=csr_matrix(train_adj),
    train_nodes=list(range(len(train_adj))),
    labels=labels[list(range(len(train_adj)))], 
    n_classes=2, multiclass=False)
np.save(outdir+"/train_feats.npy", train_features)

valid_edges = np.concatenate(all_edges, axis=0)
valid_features = np.concatenate(all_features, axis=0)
labels = np.concatenate(Ls, axis=0)[:,52]
valid_adj = np.zeros((len(valid_features), len(valid_features)))
valid_adj[valid_edges[:,0], valid_edges[:,1]] = 1
np.savez_compressed(outdir + "/valid_graph.npz", graph=csr_matrix(valid_adj),
    valid_nodes=list(range(len(valid_features))),
    labels=labels, 
    n_classes=2, multiclass=False)
np.save(outdir+"/valid_feats.npy", valid_features)

from shutil import copyfile
copyfile(outdir+"/valid_feats.npy", outdir+"/test_feats.npy")
np.savez_compressed(outdir + "/test_graph.npz", graph=csr_matrix(valid_adj),
    test_nodes=list(range(len(valid_features))),
    labels=labels, n_classes=2, multiclass=False)