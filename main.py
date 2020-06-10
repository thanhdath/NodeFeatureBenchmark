import argparse
import numpy as np
import networkx as nx
from dataloader import DefaultDataloader, CitationDataloader, RedditDataset, NELLDataloader
from features_init import lookup as lookup_feature_init
import torch
import random
from dgl.data import citation_graph as citegrh
from algorithms.node_embedding import *
import os
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser(
        description="Node feature initialization benchmark.")
    parser.add_argument('--dataset', default="data/cora")
    parser.add_argument('--init', default="ori")
    parser.add_argument('--feature_size', default=128, type=int)
    parser.add_argument('--learn-features', dest='learnable_features',
                        action='store_true')
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

    parser.add_argument('--aggregator', default="mean",
                        help="Aggregator type (mean or pooling)")
    parser.add_argument('--load-model', dest='load_model',
                        help="Path to pretrain embeds model.")
    parser.add_argument('--max_degree', default=25, type=int,
                        help="Max degree for neighbors sampling.")
    parser.add_argument('--samples_1', default=25, type=int, help="")
    return parser.parse_args()


def get_algorithm(args, data, features):
    return GraphsageAPI(data, features, cuda=args.cuda,
        aggregator=args.aggregator,
        learnable_features=args.learnable_features,
        suffix="{}-{}-{}".format(args.dataset.split('/')
                                    [-1], args.init, args.seed),
        load_model=args.load_model)


def add_weight(subgraph):
    for n1, n2 in subgraph.edges():
        subgraph[n1][n2]['weight'] = 1
    return subgraph


def get_feature_initialization(args, data, inplace=True, input_graph=False):
    if input_graph:
        graph = data
    else:
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
        kwargs = {"feature_path": args.dataset + "/features.npz"}
    elif init == "label":
        kwargs = {"label_path": args.dataset + "/labels.npz"}
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
        elif init in "pagerank triangle kcore".split():
            kwargs = {"use_networkit": True}
            graph = data.graph_networkit()
            inplace = False
            print("Warning: Init using {} will set inplace = False".format(init))
        elif init in "egonet coloring clique graphwave".split():
            graph = data.graph_networkx()
            inplace = False
            print("Warning: Init using {} will set inplace = False".format(init))

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
        return CitationDataloader(dataset, random_split=False)
    elif data_name == "reddit":
        return RedditDataset(self_loop=False)
    elif data_name == "reddit_self_loop":
        return RedditDataset(self_loop=True)
    elif data_name == "NELL":
        return NELLDataloader(dataset)
    else:
        # cora bc flickr wiki youtube homo-sapiens
        return DefaultDataloader(dataset, random_split=False)


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
            features = get_feature_initialization(args, data, inplace=inplace)
            if not os.path.isdir('feats'):
                os.makedirs('feats')
            if args.init not in ["identity", "ori"]:
                np.savez_compressed(feat_file, features=features)
    features = dict2arr(features, data.graph)
    alg = get_algorithm(args, data, features)
    embeds = alg.train()

def init_environment(args):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)


if __name__ == '__main__':
    args = parse_args()
    init_environment(args)
    if args.dataset.endswith("/"):
        args.dataset = args.dataset[:-1]
    if args.init == "learnable":
        args.init = "uniform"
        args.learnable_features = True
    print(args)
    main(args)
