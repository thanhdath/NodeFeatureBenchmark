from SGC.SGC import SGC
import argparse 
import numpy as np
import networkx as nx
from utils import lookup as lookup_feature_init
from openne.classify import Classifier, read_node_label
from sklearn.linear_model import LogisticRegression

def parse_args():
    args = argparse.ArgumentParser(description="Node feature initialization benchmark.")
    args.add_argument('--data', default="data/cora")
    args.add_argument('--alg', default="sgc")
    args.add_argument('--init', default="random")
    args.add_argument('--feature_size', default=5, type=int)
    args.add_argument('--dim_size', default=128, type=int)
    args.add_argument('--seed', type=int, default=40)
    return args.parse_args()

def load_graph(data_dir):
    print("Loading graph ...")
    graph = nx.read_edgelist(data_dir+'/edgelist.txt', nodetype=int)
    print("== Done loading graph ")
    return graph

def get_algorithm(args):
    if args.alg == "sgc":
        return SGC
    else:
        raise NotImplementedError

def get_feature_initialization(args):
    if args.init not in lookup_feature_init:
        raise NotImplementedError
    return lookup_feature_init[args.init]()

def evaluate_by_classification(vectors, X, Y, seed, train_percent=0.5):
    clf = Classifier(vectors=vectors, clf=LogisticRegression(solver="lbfgs"))
    scores = clf.split_train_evaluate(X, Y, train_percent, seed=seed)
    return scores

def print_classify(dictt):
    for k in sorted(dictt.keys()):
        print("{0}: {1:.3f}".format(k, dictt[k]))

def main(args):
    alg = get_algorithm(args)
    graph = load_graph(args.data)
    # 
    init_feature = get_feature_initialization(args)
    init_feature.generate(graph, args.feature_size, inplace=True)

    # embed
    alg = alg(args.dim_size, graph, num_walks=5, walk_length=10,
        neg_sample_size=5)
    vectors = alg.get_vectors()

    # evaluate
    X, Y = read_node_label(args.data + '/labels.txt')
    scores = evaluate_by_classification(vectors, X, Y, args.seed)
    print("Micro: {:.3f}\t Macro: {:.3f}".format(scores['micro'], scores['macro']))

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    print(args)
    main(args)
