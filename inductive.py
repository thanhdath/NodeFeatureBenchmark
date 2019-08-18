import argparse
import numpy as np
import networkx as nx
from dataloader import PPIDataset, RedditInductiveDataset
from features_init import lookup as lookup_feature_init
import torch
import random
from dgl.data import citation_graph as citegrh
from parser import *
from algorithms.node_embedding import SGC, DGIAPI, GraphsageInductive
from algorithms.logreg_inductive import LogisticRegressionInductive
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Node feature initialization benchmark.")
    parser.add_argument('--dataset', default="data/ppi")
    parser.add_argument('--init', default="ori")
    parser.add_argument('--feature_size', default=128, type=int)
    # args.add_argument('--train_features', action='store_true')
    parser.add_argument('--shuffle', action='store_true',
                        help="Whether shuffle features or not.")
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--cuda', action='store_true')

    # for logistic regression
    parser.add_argument('--logreg-bias', action='store_true',
                        dest='logreg_bias', help="Whether use bias in logistic regression or not.")
    parser.add_argument('--logreg-wc', dest='logreg_weight_decay', type=float,
                        default=5e-6, help="Weight decay for logistic regression.")
    parser.add_argument('--logreg-epochs',
                        dest='logreg_epochs', default=200, type=int)

    subparsers = parser.add_subparsers(dest="alg",
                                       help='Choose 1 of the GNN algorithm from: sgc, dgi, graphsage, nope.')
    add_sgc_parser(subparsers)
    add_nope_parser(subparsers)
    add_dgi_parser(subparsers)
    add_graphsage_parser(subparsers)
    return parser.parse_args()


def get_algorithm(args, train_data, train_features, val_data=None, val_features=None, 
    test_data=None, test_features=None):
    if args.alg == "sgc":
        return SGC(train_data, train_features, degree=args.degree, cuda=args.cuda)
    elif args.alg == "dgi":
        return DGIAPI(train_data, train_features, cuda=args.cuda)
    elif args.alg == "graphsage":
        return GraphsageInductive(train_data, val_data, test_data, train_features, val_features,
            test_features, cuda=args.cuda, aggregator=args.aggregator)
    else:
        raise NotImplementedError

def add_weight(subgraph):
    for n1, n2 in subgraph.edges():
        subgraph[n1][n2]['weight'] = 1
    return subgraph

def get_feature_initialization(args, graph, mode, inplace=True):
    elms = args.init.split("-")
    if len(elms) < 2:
        init = elms[0]
        normalizer = "pass"
    else:
        init, normalizer = elms[:2]
    if init not in lookup_feature_init:
        raise NotImplementedError
    kwargs = {}
    if init == "ori":
        kwargs = {"feature_path": args.dataset+"/{}_feats.npy".format(mode)}
    elif init == "ssvd0.5":
        init = "ssvd"
        kwargs = {"alpha": 0.5}
    elif init == "ssvd1":
        init = "ssvd"
        kwargs = {"alpha": 1}
    elif init in ["gf", "node2vec"]:
        add_weight(graph)

    if "reddit" in args.dataset and init == "deepwalk":
        graph.build_neibs_dict()

    init_feature = lookup_feature_init[init](**kwargs)
    return init_feature.generate(graph, args.feature_size,
                                 inplace=inplace, normalizer=normalizer, verbose=args.verbose,
                                 shuffle=args.shuffle)


def dict2arr(dictt, graph):
    """
    Note: always sort graph nodes
    """
    dict_arr = torch.FloatTensor([dictt[int(x)] for x in graph.nodes()])
    return dict_arr


def load_data(dataset):
    data_name = dataset.split('/')[-1]
    if data_name == "ppi":
        return PPIDataset("train"), PPIDataset("valid"), PPIDataset("test")
    elif "reddit" in data_name:
        return (RedditInductiveDataset("train", self_loop=("self_loop" in data_name)), 
            RedditInductiveDataset("valid", self_loop=("self_loop" in data_name)), 
            RedditInductiveDataset("test", self_loop=("self_loop" in data_name)))

def load_features(mode, graph, args):
    inits_one = "degree-standard triangle-standard kcore-standard egonet-standard clique-standard coloring-standard".split()
    if args.init in inits_one:
        load_seed = 40
    else:
        load_seed=  args.seed

    feat_file = 'feats/{}-{}-{}-seed{}.npz'.format(args.dataset.split('/')[-1], 
        mode, args.init, load_seed)
    if os.path.isfile(feat_file):
        features = np.load(feat_file, allow_pickle=True)['features'][()]
    else:
        features = get_feature_initialization(args, graph, mode, inplace=False)
        if not os.path.isdir('feats'):
            os.makedirs('feats')
        np.savez_compressed(feat_file, features=features)
    features = dict2arr(features, graph)
    return features

def main(args):
    train_data, val_data, test_data = load_data(args.dataset)
    train_features = load_features('train', train_data.graph, args)
    val_features = load_features('valid', val_data.graph, args)
    test_features = load_features('test', test_data.graph, args)

    use_default_classifier = False
    if args.alg == "sgc":
        # aggregate only -> create train val test alg
        train_alg = get_algorithm(args, train_data, train_features) 
        train_embs = train_alg.train()
        val_alg = get_algorithm(args, val_data, val_features)
        val_embs = val_alg.train()
        test_alg = get_algorithm(args, test_data, test_features)
        test_embs = test_alg.train()
        use_default_classifier = True
    elif args.alg == "dgi":
        alg = get_algorithm(args, train_data, train_features)
        train_embs = alg.train()
        val_embs = alg.get_embeds(val_features, val_data.graph)
        test_embs = alg.get_embeds(test_features, test_data.graph)
        use_default_classifier = True
    elif args.alg == "graphsage":
        alg = get_algorithm(args, train_data, train_features, val_data, val_features,
            test_data, test_features)
        alg.train()

    if use_default_classifier:
        print("Using default logistic regression")
        classifier = LogisticRegressionInductive(train_embs, val_embs, test_embs, 
            train_data.labels, val_data.labels, test_data.labels,
            epochs=args.logreg_epochs, weight_decay=args.logreg_weight_decay,
            bias=args.logreg_bias, cuda=args.cuda, 
            multiclass=train_data.multiclass)

def init_environment(args):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)


if __name__ == '__main__':
    args = parse_args()
    init_environment(args)
    print(args)
    main(args)
