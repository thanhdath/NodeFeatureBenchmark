from SGC.SGC import SGC
from pyGAT.GAT import GATAPI
import argparse 
import numpy as np
import networkx as nx
from features_init import lookup as lookup_feature_init
from sklearn.linear_model import LogisticRegression
import torch 
import random

def parse_args():
    args = argparse.ArgumentParser(description="Node feature initialization benchmark.")
    args.add_argument('--data', default="data/cora")
    args.add_argument('--alg', default="sgc")
    args.add_argument('--init', default="random")
    args.add_argument('--epochs', default=100, type=int)
    args.add_argument('--feature_size', default=5, type=int)
    args.add_argument('--seed', type=int, default=40)
    return args.parse_args()

def add_weight(subgraph):
    for n1, n2 in subgraph.edges():
        subgraph[n1][n2]['weight'] = 1
    return subgraph

def read_node_label(filename):
    fin = open(filename, 'r')
    labels = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        labels[vec[0]] = vec[1:]
    fin.close()
    return labels

def load_graph(data_dir):
    print("Loading graph ...")
    graph = nx.read_edgelist(data_dir+'/edgelist.txt', nodetype=int)
    add_weight(graph)
    labels = read_node_label(data_dir+'/labels.txt')
    print("== Done loading graph ")
    return graph, labels

def get_algorithm(args):
    if args.alg == "sgc":
        return SGC
    elif args.alg == "gat":
        return GATAPI
    else:
        raise NotImplementedError

def get_feature_initialization(args, graph, inplace = True):
    if args.init not in lookup_feature_init:
        raise NotImplementedError
    kwargs = {}
    if args.init == "ori":
        kwargs = {"label_path": args.data+"/labels.txt"}
    init_feature = lookup_feature_init[args.init](**kwargs)
    return init_feature.generate(graph, args.feature_size, inplace=inplace)

# def evaluate_by_classification(vectors, X, Y, seed, train_percent=0.5):
#     clf = Classifier(vectors=vectors, clf=LogisticRegression(solver="lbfgs"))
#     scores = clf.split_train_evaluate(X, Y, train_percent, seed=seed)
#     return scores

def print_classify(dictt):
    for k in sorted(dictt.keys()):
        print("{0}: {1:.3f}".format(k, dictt[k]))

def main(args):
    alg = get_algorithm(args)
    graph, labels = load_graph(args.data)
    # 
    get_feature_initialization(args, graph)

    # embed
    alg = alg(graph, labels, epochs=args.epochs)
    # vectors = alg.get_vectors()

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)
    print(args)
    main(args)
