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

# def collate(sample):
#     graphs, feats, labels = map(list, zip(*sample))
#     graph = dgl.batch(graphs)
#     feats = torch.from_numpy(np.concatenate(feats))
#     labels = torch.from_numpy(np.concatenate(labels))
#     return graph, feats, labels

# train_dataset = LegacyPPIDataset(mode="train")
# # train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collate)

# # graphs = list(train_dataloader)
# ids = np.random.permutation(len(train_dataset))
# ids = [id for id in ids if train_dataset.train_graphs[id].number_of_nodes() < 2000]


# G1 = train_dataset.train_graphs[ids[0]]
# G2 = train_dataset.train_graphs[ids[1]]

# A1 = np.asarray(G1.adjacency_matrix_scipy().todense())
# A2 = np.asarray(G2.adjacency_matrix_scipy().todense())

# if len(A1) < len(A2):
#     A1, A2 = A2, A1
# print(A1.shape, A2.shape)

# L1 = train_dataset.train_labels[ids[0]]
# L2 = train_dataset.train_labels[ids[1]]

# A1[A1 > 0] = 1
# A2[A2 > 0] = 1


def gen_graph(n=200):
    v = [1]*(n//2) + [0]*(n//2)
    random.shuffle(v)
    p = 64
    d = 5
    lam = 0.5
    mu = 5
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
    A[p_A > p_samples] = 1
    return A, B, np.array(v)

A1, F1, L1 = gen_graph(n=200)
A2, F2, L2 = gen_graph(n=200)

A1 = torch.FloatTensor(A1).cuda()
A2 = torch.FloatTensor(A2).cuda()

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
        l = A1.shape[0]
        l2 = A2.shape[0]
        ori_dim = 128
        self.X1 = nn.Embedding(l, ori_dim)
        self.M = nn.Linear(l, l2, bias=False)
        self.linear1 = nn.Sequential(
            nn.Linear(l*(l-1)//2, ori_dim, bias=False),
            nn.Linear(ori_dim, l*(l-1)//2, bias=False)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(l2*(l2-1)//2, ori_dim, bias=False),
            nn.Linear(ori_dim, l2*(l2-1)//2, bias=False)
        )
        
    def forward(self):
        X1 = self.X1.weight
        x1 = torch.pdist(X1, p=2)
        x1 = self.linear1(x1)
        # x1 = torch.sigmoid(x1)

        x2 = self.M(X1.t()).t()
        x2 = torch.pdist(x2, p=2)
        x2 = self.linear2(x2)
        # x2 = torch.sigmoid(x2)
        return x1, x2

model = Model().cuda()
# loss_fn = nn.BCELoss()
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
for iter in range(400):
    model.train()
    optim.zero_grad()
    pred_A1, pred_A2 = model()

    # f2 = X2.weight
    # scores = f2.mm(f2.t())
    # pred_A2 = scores
    upper_half1 = A1[torch.triu(torch.ones_like(A1), diagonal=1) == 1]
    upper_half2 = A2[torch.triu(torch.ones_like(A2), diagonal=1) == 1]
    loss = loss_fn(pred_A1, upper_half1) + loss_fn(pred_A2, upper_half2)
    loss.backward()
    optim.step()
    if iter%50 == 0:
        microf11 = compute_f1(pred_A1, upper_half1)
        microf12 = compute_f1(pred_A2, upper_half2)
        print(f"Iter {iter} - loss {loss:.4f} - f1 {microf11:.3f}  {microf12:.3f}")
    

# gen edgelist, labels, featuresh
X1 = model.X1.weight
X2 = model.M(X1.t()).t()
features = X1.detach().cpu().numpy()
edgelist = np.argwhere(A1.detach().cpu().numpy() > 0)
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

features = X2.detach().cpu().numpy()
edgelist = np.argwhere(A2.detach().cpu().numpy() > 0)
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
"""

# python -u main.py --dataset temp/data-autoencoder/ppi/1/ --init ori --cuda graphsage --aggregator mean --load-model graphsage-best-model-0-ori-40.pkl > logs/ppi1-transfer-from-0.log
# python -u main.py --dataset temp/data-autoencoder/ppi/0/ --init ori --cuda graphsage --aggregator mean --load-model graphsage-best-model-1-ori-40.pkl > logs/ppi0-transfer-from-1.log