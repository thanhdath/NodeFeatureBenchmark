"""
for s in 100 101 102 103 104
do
    echo $s 
    python torus_sphere.py --seed $s --graph-method knn
    python torus_sphere.py --seed $s --graph-method sigmoid
done
"""


import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import math
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import torch.nn.functional as F
import os 
import dgl

def sample_sphere(num_nodes):
    N = num_nodes

    phi = np.random.uniform(low=0,high=2*np.pi, size=N)
    costheta = np.random.uniform(low=-1,high=1,size=N)
    u = np.random.uniform(low=0,high=1,size=N)

    theta = np.arccos( costheta )
    r = 1.0

    x = r * np.sin( theta) * np.cos( phi )
    y = r * np.sin( theta) * np.sin( phi )
    z = r * np.cos( theta )

    return torch.tensor(list(zip(x, y, z)))

def sample_torus(R, r, n_nodes):
    angle = np.linspace(0, 2*np.pi, 32)
    theta, phi = np.meshgrid(angle, angle)
    X = (R + r * np.cos(phi)) * np.cos(theta)
    Y = (R + r * np.cos(phi)) * np.sin(theta)  
    Z = r * np.sin(phi)

    ps_x = []
    ps_y = []
    ps_z = []
    for _ in range(n_nodes):
        u = random.random()
        v = random.random()
        w = random.random()
        omega = 2*np.pi*u
        theta = 2*np.pi*v 
        threshold = (R + r*math.cos(omega))/(R+r)
        if w <= threshold:
            x = (R+r*math.cos(omega))*math.cos(theta)
            y = (R+r*math.cos(omega))*math.sin(theta)
            z = r*math.sin(omega)
            ps_x.append(x)
            ps_y.append(y)
            ps_z.append(z)
    return torch.tensor(list(zip(ps_x, ps_y, ps_z)))

def generate_graph(features, kind="sigmoid", k=5, log=True):
    features_norm = F.normalize(features, dim=1)
    # scores = features_norm.mm(features_norm.t())
    N = len(features_norm)
    scores = torch.pdist(features_norm)
    # print("Scores before sigmoid")
    # print(scores)
    if log:
        print(f"Generate graph using {kind}")
    if kind == "sigmoid":
        # print("Scores after sigmoid")
        scores = 1 - torch.sigmoid(scores)
        # find index to cut 
        n_edges = int((k*N - N)/2)
        threshold = scores[torch.argsort(-scores)[n_edges]]
        if log:
            print(f"Scores range: {scores.min():.3f}-{scores.max():.3f}")
            print(f"Expected average degree: {k} => Threshold: {threshold:.3f}")
        edges = scores >= threshold
        adj = np.zeros((len(features), len(features)), dtype=np.int)
        inds = torch.triu(torch.ones(len(adj),len(adj))) 
        inds[np.arange(len(adj)), np.arange(len(adj))] = 0
        adj[inds == 1] = edges.cpu().numpy().astype(np.int)
        adj = adj + adj.T
        adj[adj > 0] = 1
        src, trg = adj.nonzero()
        edge_index = np.concatenate([src.reshape(1, -1), trg.reshape(1,-1)], axis=0).T
    elif kind == "knn":
        k = int(k)
        if log:
            print(f"Knn k = {k}")
        scores_matrix = np.zeros((len(features), len(features)))
        inds = torch.triu(torch.ones(len(features),len(features))) 
        inds[np.arange(len(features)), np.arange(len(features))] = 0
        scores_matrix[inds == 1] = scores
        scores_matrix = scores_matrix + scores_matrix.T
        if len(scores_matrix) > 60000: # avoid memory error
            edge_index = []
            for i, node_scores in enumerate(scores_matrix):
                candidate_nodes = np.argsort(node_scores)[:k]
                edge_index += [[i, node] for node in candidate_nodes]
            edge_index = np.array(edge_index, dtype=np.int32)
        else:
            sorted_scores = np.argsort(scores_matrix, axis=1)[:, :k]
            edge_index = np.zeros((len(scores_matrix)*k, 2), dtype=np.int32)
            N = len(scores_matrix)
            for i in range(k):
                edge_index[i*N:(i+1)*N, 0] = np.arange(N)
                edge_index[i*N:(i+1)*N, 1] = sorted_scores[:, i]
    else:
        raise NotImplementedError
    if log:
        print("Number of edges: ", edge_index.shape[0])
    return edge_index

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num-graphs', type=int, default=1000)
parser.add_argument('--graph-method', default='knn')
parser.add_argument('--seed', type=int, default=100)
args = parser.parse_args()


num_graphs = args.num_graphs
# num_nodes_per_graph = 500
graph_method = args.graph_method
if graph_method == "knn":
    k = 5
else:
    k = 20
# noises = [0.0, 0.0001, 0.001, 0.01, 0.1]
noises = [0.0]
seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

for noise in noises:
    print("Noise: ", noise)
    features_list = []
    graph_list = []
    labels = []

    # Generate sphere graphs
    for _ in range(num_graphs//2):
        n_nodes = np.random.randint(100, 200)
        features = sample_sphere(n_nodes)
        features_list.append(features)
        edge_index = generate_graph(features, kind=graph_method, k=k)
        assert edge_index.max() <= len(features), f"Wrong edge index {edge_index.max()} - {len(features)}"
        graph_list.append(edge_index)
        labels.append(0)

    # Generate torus graphs
    for _ in range(num_graphs//2):
        n_nodes = np.random.randint(100, 200)
        features = sample_torus(80, 40, n_nodes) / 120
        features_list.append(features)
        edge_index = generate_graph(features, kind=graph_method, k=k)
        assert edge_index.max() <= len(features), "Wrong edge index"
        graph_list.append(edge_index)
        labels.append(1)

    all_edges = []
    edges_attr = []
    all_nodes = []
    graph_labels = []
    graph_indicators = []
    node_attributes = []
    node_labels = []
    inc = 0
    for idx in range(len(features_list)):
        features = features_list[idx]
        edgelist = graph_list[idx] + inc
        graph_label = labels[idx]
        inc += len(features)
        
        all_edges.append(edgelist)
        graph_labels.append(graph_label)
        graph_indicators += [idx for _ in range(len(features))]
        node_attributes.append(features)
    all_edges = np.concatenate(all_edges, axis=0).astype(np.int32) + 1
    # all_nodes = np.concatn(all_nodes)
    graph_indicators = np.array(graph_indicators) + 1
    node_attributes = np.concatenate(node_attributes, axis=0)

    print("edges: ", len(all_edges))
    print("nodes: ", len(node_attributes))

    datasetname = f"torus_vs_sphere-{graph_method}-n{noise}-seed{seed}"
    outdir = f"data/{datasetname}"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    with open(outdir + f"/{datasetname}_A.txt", "w+") as fp:
        for src, trg in all_edges:
            fp.write("{},{}\n".format(src, trg))
    with open(outdir + f"/{datasetname}_graph_indicator.txt", "w+") as fp:
        for i in graph_indicators:
            fp.write(f"{i}\n")
    with open(outdir + f"/{datasetname}_graph_labels.txt", "w+") as fp:
        for i in graph_labels:
            fp.write(f"{i}\n")
    with open(outdir + f"/{datasetname}_node_attributes.txt", "w+") as fp:
        for i in node_attributes:
            fp.write("{}\n".format(",".join([f"{x:.4f}" for x in i])))
