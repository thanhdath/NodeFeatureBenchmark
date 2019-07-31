import networkx as nx 
from networkx.readwrite import json_graph
import json 


data = json.load(open('reddit/reddit-G_full.json'))
G = json_graph.node_link_graph(data)
directed = nx.is_directed(G)
print('Directed: ', directed)

id2idx = json.load(open('reddit/reddit-id_map.json'))

with open('reddit/edgelist.txt', 'w+') as fp:
    for src, trg in G.edges():
        fp.write("{} {}\n".format(id2idx[src], id2idx[trg]))
        if not directed:
            fp.write("{} {}\n".format(id2idx[trg], id2idx[src]))

import numpy as np 
features = np.load('reddit/reddit-feats.npy')
with open('reddit/features.txt', 'w+') as fp:
    for node in G.nodes():
        fp.write("{} {}\n".format(id2idx[node], 
            ' '.join(map(str, features[id2idx[node]]))))

class_map = json.load(open('reddit/reddit-class_map.json'))

with open('reddit/labels.txt', 'w+') as fp:
    for node in G.nodes():
        fp.write("{} {}\n".format(id2idx[node], class_map[node]))