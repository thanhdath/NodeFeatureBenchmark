import argparse
import numpy as np
import networkx as nx
from dataloader import DefaultDataloader, CitationDataloader, RedditDataset
from features_init import lookup as lookup_feature_init
import torch
import random
from dgl.data import citation_graph as citegrh
from parser import *
from algorithms.node_embedding import SGC, Nope, DGIAPI, GraphsageAPI
from algorithms.node_embedding.graphsagetf.api import Graphsage
from algorithms.logistic_regression import LogisticRegressionPytorch
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Node feature initialization benchmark.")
    parser.add_argument('--dataset', default="data/cora")
    parser.add_argument('--init', default="ori")
    parser.add_argument('--feature_size', default=128, type=int)
    parser.add_argument('--learn-features', dest='learnable_features', action='store_true')
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
                        dest='logreg_epochs', default=300, type=int)

    subparsers = parser.add_subparsers(dest="alg",
                                       help='Choose 1 of the GNN algorithm from: sgc, dgi, graphsage, nope.')
    add_sgc_parser(subparsers)
    add_nope_parser(subparsers)
    add_dgi_parser(subparsers)
    add_graphsage_parser(subparsers)
    return parser.parse_args()


def get_algorithm(args, data, features):
    if args.alg == "sgc":
        return SGC(data, features, degree=args.degree, cuda=args.cuda)
    elif args.alg == "nope":
        return Nope(features)
    elif args.alg == "dgi":
        return DGIAPI(data, features, self_loop=args.self_loop, cuda=args.cuda,
            learnable_features=args.learnable_features, suffix=args.dataset.split('/')[-1])
    elif args.alg == "graphsage":
        if features.shape[0] > 10000:
            return Graphsage(data, features)
        else:
            return GraphsageAPI(data, features, cuda=args.cuda, aggregator=args.aggregator,
                learnable_features=args.learnable_features, suffix=args.dataset.split('/')[-1])
    else:
        raise NotImplementedError

def add_weight(subgraph):
    for n1, n2 in subgraph.edges():
        subgraph[n1][n2]['weight'] = 1
    return subgraph

def get_feature_initialization(args, data, inplace=True):
    graph = data.graph
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
        if "reddit" in args.dataset:
            kwargs = {"feature_path": args.dataset+"/features.npy"}
        else:
            kwargs = {"feature_path": args.dataset+"/features.txt"}
    elif init == "label":
        kwargs = {"label_path": args.dataset+"/labels.txt"}
    elif init == "ssvd0.5":
        init = "ssvd"
        kwargs = {"alpha": 0.5}
    elif init == "ssvd1":
        init = "ssvd"
        kwargs = {"alpha": 1}
    elif init in ["node2vec"]:
        add_weight(graph)

    if "reddit" in args.dataset:
        if init == "deepwalk":
            graph.build_neibs_dict()
        elif init in "pagerank".split():
            kwargs = {"use_networkit": True}
            graph = data.graph_networkit()
            inplace = False
            print("Warning: Init using {} will be inplace = False".format(init))

    # super slow feature initialization method
    # walk_file = "{}-seed{}.npy".format(args.dataset.split('/')[-1], args.seed)
    # if init == "node2vec" and os.path.isfile(walk_file):
    #     features = np.load(walk_file)
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
    if data_name in ["citeseer", "pubmed"]:
        return CitationDataloader(dataset)
    elif data_name == "reddit":
        return RedditDataset(self_loop=False)
    elif data_name == "reddit_self_loop":
        return RedditDataset(self_loop=True)
    else:
        # cora bc flickr wiki youtube homo-sapiens
        return DefaultDataloader(dataset)


def main(args):
    data = load_data(args.dataset)
    inplace = "reddit" not in args.dataset 

    inits_one = "degree-standard triangle-standard kcore-standard egonet-standard clique-standard coloring-standard".split()
    if args.init in inits_one:
        load_seed = 40
    else:
        load_seed = args.seed
    
    feat_file = 'feats/{}-{}-seed{}-dim{}.npz'.format(args.dataset.split('/')[-1], args.init, 
        load_seed, args.feature_size)

    if args.shuffle:
        features = get_feature_initialization(args, data, inplace=inplace)
    else:
        if os.path.isfile(feat_file):
            features = np.load(feat_file, allow_pickle=True)['features'][()]
        else:
            features = get_feature_initialization(args, data.graph, inplace=inplace)
            if not os.path.isdir('feats'):
                os.makedirs('feats')
            if args.init not in ["identity"]:
                np.savez_compressed(feat_file, features=features)
    features = dict2arr(features, data.graph)

    inits_fixed_dim = "ori ori-rowsum ori-standard label identity".split()
    # if args.init not in inits_fixed_dim:
    #     assert features.shape[1] == args.feature_size, "Wrong feature dimension."
    alg = get_algorithm(args, data, features)

    embeds = alg.train()

    if args.alg in ["sgc", "dgi", "nope"]:
        print("Using default logistic regression")
        classifier = LogisticRegressionPytorch(embeds,
                                               data.labels, data.train_mask, data.val_mask, data.test_mask,
                                               epochs=args.logreg_epochs, weight_decay=args.logreg_weight_decay,
                                               bias=args.logreg_bias, cuda=args.cuda, 
                                               multiclass=data.multiclass, suffix=args.dataset.split('/')[-1])


def init_environment(args):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)


if __name__ == '__main__':
    args = parse_args()
    init_environment(args)
    if args.dataset.endswith("/"):
        args.dataset = args.dataset[:-1]
    print(args)
    main(args)
