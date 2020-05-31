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

def generate_graph(features, kind="sigmoid", threshold=None, k=5, noise_knn=0.0):
    features_norm = F.normalize(features, dim=1)
    scores = features_norm.mm(features_norm.t())
    # print(f"Generate graph using {kind}")
    if kind == "sigmoid":
        scores = torch.sigmoid(scores)
        if threshold is None:
            threshold = scores.mean()
        # print(f"Scores range: {scores.min()}-{scores.max()}")
        # print("Threshold: ", threshold)
        adj = scores > threshold
        adj = adj.int()
        edge_index = adj.nonzero().cpu().numpy()
    elif kind == "knn":
        # print(f"Knn k = {k}")
        sorted_scores = torch.argsort(-scores, dim=1)[:, :k]
        edge_index = np.zeros((len(scores)*k, 2), dtype=np.int32)
        N = len(scores)
        for i in range(k):
            edge_index[i*N:(i+1)*N, 0] = np.arange(N)
            edge_index[i*N:(i+1)*N, 1] = sorted_scores[:, i]

        if noise_knn > 0:
            adj = np.zeros((N, N), dtype=np.int32)
            adj[edge_index[:,0], edge_index[:,1]] = 1
            n_added_edges = int(len(adj)**2 * noise_knn)
            no_edge_index = np.argwhere(adj == 0)
            add_edge_index = np.random.permutation(no_edge_index)[:n_added_edges]
            adj[add_edge_index[:,0], add_edge_index[:,1]] = 1
            src, trg = adj.nonzero()
            edge_index = np.concatenate([src.reshape(-1,1), trg.reshape(-1,1)], axis=1)
    else:
        raise NotImplementedError
    
    # print("Number of edges: ", edge_index.shape[0])
    return edge_index

def gen_graph(n=200):
    v = [1]*(n//2) + [0]*(n//2)
    random.shuffle(v)
    p = 64
    d = 5
    lam = 1
    mu = 0
    """# Generate B (i.e. X)"""
    m_u = np.zeros(p)
    cov_u = np.eye(p)/p
    u = np.random.multivariate_normal(m_u, cov_u, n)
    Z = np.random.randn(n, p)
    B = np.zeros((n, p))

    for i in range(n):
        a = np.sqrt(mu/ n)*v[i]*u[i]
        b = Z[i]/np.sqrt(p)
        B[i,:] =  a + b
    
    """# Generate A"""
    c_in = d + lam*np.sqrt(d)
    c_out = d - lam*np.sqrt(d)

    p_A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if v[i] == v[j]:
                p_A[i,j] = c_in / n
            else:
                p_A[i,j] = c_out / n

    p_samples = np.random.sample((n,n))
    A = np.zeros((n,n))
    edge_index = generate_graph(torch.FloatTensor(B), kind="knn", k=5)
    A[edge_index[:,0], edge_index[:,1]] = 1
    return A, B, np.array(v)

A1, F1, L1 = gen_graph(n=200)
A2, F2, L2 = gen_graph(n=200)


# gen edgelist, labels, featuresh
features = F1
edgelist = np.argwhere(A1 > 0)
labels = L1

import os
outdir = "data-autoencoder/syn/0"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

with open(outdir + "/edgelist.txt", "w+") as fp:
    for src, trg in edgelist:
        fp.write(f"{src} {trg}\n")
with open(outdir + "/labels.txt", "w+") as fp:
    for i, label in enumerate(labels):
        fp.write(f"{i} {label}\n")
np.savez_compressed(outdir + "/features.npz", features=features)

features = F2
edgelist = np.argwhere(A2 > 0)
labels = L2

outdir = "data-autoencoder/syn/1"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

with open(outdir + "/edgelist.txt", "w+") as fp:
    for src, trg in edgelist:
        fp.write(f"{src} {trg}\n")
with open(outdir + "/labels.txt", "w+") as fp:
    for i, label in enumerate(labels):
        fp.write(f"{i} {label}\n")
np.savez_compressed(outdir + "/features.npz", features=features)

"""
for i in 0 1
do
echo $i
    python -u -W ignore main.py --dataset temp/data-autoencoder/syn/$i --init ori --cuda graphsage --aggregator mean > logs/syn$i.log
done
python -u -W ignore main.py --dataset temp/data-autoencoder/syn/1 --init ori --cuda graphsage --aggregator mean --load-model graphsage-best-model-0-ori-40.pkl > logs/syn1-tf-0.log
python -u -W ignore main.py --dataset temp/data-autoencoder/syn/0 --init ori --cuda graphsage --aggregator mean --load-model graphsage-best-model-1-ori-40.pkl > logs/syn0-tf-1.log


for i in 0 1
do
echo $i
    python -u -W ignore main.py --dataset temp/data-autoencoder/syn/$i --init ori --cuda gat > logs/syn$i.log
done
python -u -W ignore main.py --dataset temp/data-autoencoder/syn/1 --init ori --cuda gat --load-model gat-best-model-0-ori-40.pkl > logs/syn1-tf-0.log
python -u -W ignore main.py --dataset temp/data-autoencoder/syn/0 --init ori --cuda gat --load-model gat-best-model-1-ori-40.pkl > logs/syn0-tf-1.log
"""
