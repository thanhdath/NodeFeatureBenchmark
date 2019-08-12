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
from algorithms.logistic_regression import LogisticRegressionPytorch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Node feature initialization benchmark.")
    parser.add_argument('--dataset', default="data/cora")
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


def get_algorithm(args, data, features):
    if args.alg == "sgc":
        return SGC(data, features, degree=args.degree, cuda=args.cuda)
    elif args.alg == "nope":
        return Nope(features)
    elif args.alg == "dgi":
        return DGIAPI(data, features, self_loop=args.self_loop, cuda=args.cuda)
    elif args.alg == "graphsage":
        return GraphsageAPI(data, features, cuda=args.cuda, aggregator=args.aggregator)
    else:
        raise NotImplementedError


def get_feature_initialization(args, graph, inplace=True):
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
        kwargs = {"feature_path": args.dataset+"/features.txt"}
    elif init == "ssvd0.5":
        init = "ssvd"
        kwargs = {"alpha": 0.5}
    elif init == "ssvd1":
        init = "ssvd"
        kwargs = {"alpha": 1}
    # elif init == "node2vec":
    #     add_weight(graph)
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
    if dataset == "data/cora":
        return DefaultDataloader(dataset)
    elif dataset in ["data/citeseer", "data/pubmed"]:
        return CitationDataloader(dataset)
    elif dataset == "data/reddit":
        return RedditDataset(self_loop=False)
    elif dataset == "data/reddit_self_loop":
        return RedditDataset(self_loop=True)


def main(args):
    data = load_data(args.dataset)
    inplace = "reddit" not in args.dataset 
    features = get_feature_initialization(args, data.graph, inplace=inplace)
    features = dict2arr(features, data.graph)
    alg = get_algorithm(args, data, features)

    embeds = alg.train()

    if args.alg in ["sgc", "dgi", "nope"]:
        print("Using default logistic regression")
        classifier = LogisticRegressionPytorch(embeds,
                                               data.labels, data.train_mask, data.val_mask, data.test_mask,
                                               epochs=args.logreg_epochs, weight_decay=args.logreg_weight_decay,
                                               bias=args.logreg_bias, cuda=args.cuda)


def init_environment(args):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)


if __name__ == '__main__':
    args = parse_args()
    init_environment(args)
    print(args)
    main(args)
