
import os
from shutil import copyfile
import numpy as np
import networkx as nx
graph = nx.read_edgelist('data/bc/edgelist.txt', nodetype=int)

nodes = list(graph.nodes())
n_train = int(len(nodes)*0.8)
train_nodes = np.random.permutation(nodes)[:n_train].tolist()
test_nodes = np.random.permutation(nodes)[n_train:].tolist()

train_graph = graph.subgraph(train_nodes)
test_graph = graph.subgraph(test_nodes)
train_graph = nx.Graph(train_graph)
test_graph = nx.Graph(test_graph)
train_graph.remove_nodes_from(list(nx.isolates(train_graph)))
test_graph.remove_nodes_from(list(nx.isolates(test_graph)))

if not os.path.isdir("data/bc1-train"):
    os.makedirs("data/bc1-train")
if not os.path.isdir("data/bc1-test"):
    os.makedirs("data/bc1-test")

nx.write_edgelist(train_graph, "data/bc1-train/edgelist.txt", delimiter=" ", data=False)
copyfile("data/bc/labels.txt", "data/bc1-train/labels.txt")

nx.write_edgelist(train_graph, "data/bc1-test/edgelist.txt", delimiter=" ", data=False)
copyfile("data/bc/labels.txt", "data/bc1-test/labels.txt")
