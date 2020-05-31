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
import networkx as nx
from networkx.readwrite import json_graph
import json
from tqdm import tqdm

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
num_graphs = 1000
# num_nodes_per_graph = 500
graph_method = 'knn'
k = 5
noises = [0.0, 0.0001, 0.001, 0.01, 0.1]

for noise in noises:
    print("Noise: ", noise)
    features_list = []
    graph_list = []
    labels = []

    # Generate sphere graphs
    for _ in range(num_graphs//2):
        # n_nodes = np.random.randint(50, 150)
        n_nodes = 100
        features = sample_sphere(n_nodes)
        features_list.append(features)
        edge_index = generate_graph(features, kind=graph_method, k=k, threshold=.72, noise_knn=noise)
        assert edge_index.max() <= len(features), f"Wrong edge index {edge_index.max()} - {len(features)}"
        graph_list.append(edge_index)
        labels.append(0)

    # Generate torus graphs
    for _ in range(num_graphs//2):
        # n_nodes = np.random.randint(50, 150)
        n_nodes = 100
        features = sample_torus(80, 40, n_nodes) / 120
        features_list.append(features)
        edge_index = generate_graph(features, kind=graph_method, k=k, threshold=.72, noise_knn=noise)
        assert edge_index.max() <= len(features), "Wrong edge index"
        graph_list.append(edge_index)
        labels.append(1)
    
    datasetname = f"torus_vs_sphere-{graph_method}-n{noise}"
    outdir = f"data-multigraph/{datasetname}"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    for i, (edge_index, features, label) in tqdm(enumerate(zip(graph_list, features_list, labels))):
        g = nx.DiGraph()
        g.add_edges_from([[int(src), int(trg)] for src, trg in edge_index])
        for node in g.nodes():
            g.node[node]["features"] = features[node].tolist()
            outfile = outdir + f"/{i}.json"
            with open(outfile, "w+") as fp:
                fp.write(json.dumps(json_graph.node_link_data(g)))
    

    # all_edges = []
    # edges_attr = []
    # all_nodes = []
    # graph_labels = []
    # graph_indicators = []
    # node_attributes = []
    # node_labels = []
    # inc = 0
    # for idx in range(len(features_list)):
    #     features = features_list[idx]
    #     edgelist = graph_list[idx] + inc
    #     graph_label = labels[idx]
    #     inc += len(features)
        
    #     all_edges.append(edgelist)
    #     graph_labels.append(graph_label)
    #     graph_indicators += [idx for _ in range(len(features))]
    #     node_attributes.append(features)
    # all_edges = np.concatenate(all_edges, axis=0).astype(np.int32) + 1
    # # all_nodes = np.concatn(all_nodes)
    # graph_indicators = np.array(graph_indicators) + 1
    # node_attributes = np.concatenate(node_attributes, axis=0)

    # print("edges: ", len(all_edges))
    # print("nodes: ", len(node_attributes))

    # datasetname = f"torus_vs_sphere-{graph_method}-n{noise}"
    # outdir = f"data/{datasetname}"
    # if not os.path.isdir(outdir):
    #     os.makedirs(outdir)

    # with open(outdir + f"/{datasetname}_A.txt", "w+") as fp:
    #     for src, trg in all_edges:
    #         fp.write("{},{}\n".format(src, trg))
    # with open(outdir + f"/{datasetname}_graph_indicator.txt", "w+") as fp:
    #     for i in graph_indicators:
    #         fp.write(f"{i}\n")
    # with open(outdir + f"/{datasetname}_graph_labels.txt", "w+") as fp:
    #     for i in graph_labels:
    #         fp.write(f"{i}\n")
    # with open(outdir + f"/{datasetname}_node_attributes.txt", "w+") as fp:
    #     for i in node_attributes:
    #         fp.write("{}\n".format(",".join([f"{x:.4f}" for x in i])))