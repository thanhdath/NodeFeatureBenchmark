import networkx as nx
import glob
from main import get_feature_initialization
import argparse
from SGC.SGC_inductive import SGC
import random
import numpy as np 
import time
import torch

def parse_args():
    args = argparse.ArgumentParser(description="Node feature initialization benchmark.")
    args.add_argument('--data', default="data/cora")
    args.add_argument('--alg', default="sgc")
    args.add_argument('--init', default="random")
    # args.add_argument('--epochs', default=100, type=int)
    args.add_argument('--feature_size', default=5, type=int)
    args.add_argument('--norm_features', action='store_true')
    args.add_argument('--train_features', action='store_true')
    args.add_argument('--seed', type=int, default=40)
    args.add_argument('--verbose', type=int, default=1)
    return args.parse_args()

def add_weight(subgraph):
    for n1, n2 in subgraph.edges():
        subgraph[n1][n2]['weight'] = 1
    return subgraph

def load_multiple_graphs(args, data_dir):
    big_graph = nx.Graph()
    for file in glob.glob(data_dir+'/edgelist*.txt'):
        graph = nx.read_edgelist(file, nodetype=int)
        add_weight(graph)
        get_feature_initialization(args, graph)
        big_graph = nx.compose(big_graph, graph)
    return big_graph

def read_node_label(filename):
    fin = open(filename, 'r')
    labels = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        labels[vec[0]] = list(map(float, vec[1:]))
    fin.close()
    return labels

def get_algorithm(args):
    if args.alg == "sgc":
        return SGC
    # elif args.alg == "gat":
    #     return GATAPI
    # elif args.alg == "dgi":
    #     return DGIAPI
    # elif args.alg == "logistic":
    #     return LogisticRegressionPytorch
    else:
        raise NotImplementedError

def main(args):
    # load graph
    train_graphs = load_multiple_graphs(args, args.data+'/train')
    val_graphs = load_multiple_graphs(args, args.data+'/val')
    test_graphs = load_multiple_graphs(args, args.data+'/test')
    # load label 
    labels = read_node_label(args.data+'/labels.txt')
    alg = get_algorithm(args)
    # train
    if args.alg == "sgc":
        alg = alg(train_graphs, val_graphs, labels, trainable_features=args.train_features)
    else:
        alg = alg(train_graphs, labels)
    # test
    alg.test(test_graphs, labels)

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)
    print(args)
    main(args)
