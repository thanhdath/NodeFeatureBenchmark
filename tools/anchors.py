from dataloader import DefaultDataloader, CitationDataloader
import sys
import random

data1name = sys.argv[1]
data2name = sys.argv[2]

def load_data(dataname):
    if dataname in "cora".split():
        return DefaultDataloader("data/{}".format(dataname))
    elif dataname in "citeseer pubmed".split():
        return CitationDataloader("data/{}".format(dataname))
data1 = load_data(data1name)
data2 = load_data(data2name)

import networkx as nx
degrees1 = {x: data1.graph.degree(x) for x in data1.graph.nodes()}
degrees2 = {x: data2.graph.degree(x) for x in data2.graph.nodes()}

degree2node1 = {}
for node, d in degrees1.items():
    degree2node1[d] = degree2node1.get(d, []) + [node]
degree2node2 = {}
for node, d in degrees2.items():
    degree2node2[d] = degree2node2.get(d, []) + [node]
degree2node1 = {k: v for k, v in degree2node1.items() if k in degree2node2}
degree2node2 = {k: v for k, v in degree2node2.items() if k in degree2node1}

degrees = list(degree2node1.keys())
degrees = sorted(degrees, reverse=True)

anchors = []
for d in degrees:
    node1s = degree2node1[d]
    node2s = degree2node2[d]
    if len(node1s) == 0 or len(node2s) == 0:
        continue 
    node1 = random.choice(node1s)
    node2 = random.choice(node2s)
    anchors.append((node1, node2))
    node1s.remove(node1)
    node2s.remove(node2)
print("Number of anchors: ", len(anchors))
with open("anchors-{}-{}.dict".format(data1name, data2name), 'w+') as fp:
    for n1, n2 in anchors:
        fp.write("{} {}\n".format(n1, n2))
