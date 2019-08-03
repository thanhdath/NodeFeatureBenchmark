import networkx as nx 
from networkx.readwrite import json_graph
import json 
import pdb
import random
import numpy as np

class_map = json.load(open('../data/ppi/ppi-class_map.json'))
G = json.load(open('../data/ppi/ppi-G.json'))
G = json_graph.node_link_graph(G)
idmap = json.load(open('../data/ppi/ppi-id_map.json'))
classmap = json.load(open('../data/ppi/ppi-class_map.json'))
feats = np.load('../data/ppi/ppi-feats.npy')

with open('../data/ppi/features.txt', 'w+') as fp:
    for node in G.nodes():
        fp.write("{} {}\n".format(idmap[str(node)], ' '.join(map(str, feats[idmap[str(node)]]))))
with open('../data/ppi/labels.txt', 'w+') as fp:
    for node in G.nodes():
        fp.write("{} {}\n".format(node, ' '.join(map(str,class_map[str(node)]))))

subgraphs = list(nx.connected_component_subgraphs(G))

selected_graphs = [x for x in subgraphs if len(x.edges()) >= 15000]
print(len(selected_graphs))

candidate_tests = [x for x in selected_graphs if len(x.edges()) >= 35000]
np.random.shuffle(candidate_tests)
test_graphs = candidate_tests[:2]
candidate_tests = candidate_tests[2:]

val_graphs = candidate_tests[:2]
candidate_tests = candidate_tests[2:]

for graph in val_graphs+test_graphs:
    selected_graphs.remove(graph)
train_graphs = selected_graphs
print('test', len(test_graphs))
print('val', len(val_graphs))
print('train', len(train_graphs))

print("test graph info")
for graph in test_graphs:
    print(len(graph.edges()))
print("val graph info")
for graph in val_graphs:
    print(len(graph.edges()))
print("train graph info")
for graph in train_graphs:
    print(len(graph.edges()))

import os 
path = '../data/ppi/train'
if not os.path.isdir(path):
    os.makedirs(path)
path = '../data/ppi/val'
if not os.path.isdir(path):
    os.makedirs(path)
path = '../data/ppi/test'
if not os.path.isdir(path):
    os.makedirs(path)
for i, graph in enumerate(train_graphs):
    nx.write_edgelist(graph, "../data/ppi/train/edgelist{}.txt".format(i), delimiter=' ', data=False)
for i, graph in enumerate(val_graphs):
    nx.write_edgelist(graph, "../data/ppi/val/edgelist{}.txt".format(i), delimiter=' ', data=False)
for i, graph in enumerate(test_graphs):
    nx.write_edgelist(graph, "../data/ppi/test/edgelist{}.txt".format(i), delimiter=' ', data=False)
