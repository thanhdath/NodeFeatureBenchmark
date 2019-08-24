from types import SimpleNamespace
import argparse
import numpy as np
import networkx as nx
from dataloader.tu_dataset import TUDataset
import torch
import random
from parser import *
from algorithms.graph_embedding import *
from main import get_feature_initialization
from normalization import lookup as lookup_normalizer
import time
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Node feature initialization benchmark.")
    parser.add_argument('--dataset', default="data/ENZYMES")
    parser.add_argument('--init', default="ori")
    parser.add_argument('--feature_size', default=128, type=int)
    # args.add_argument('--train_features', action='store_true')
    parser.add_argument('--shuffle', action='store_true', help="Whether shuffle features or not.")
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--cuda', action='store_true')

    subparsers = parser.add_subparsers(dest="alg", 
        help='Choose from: diffpool, gin.')
    add_diffpool_parser(subparsers)
    add_gin_parser(subparsers)
    add_simple_graph_emb_parse(subparsers)
    return parser.parse_args()

def build_diffpool_params(args, data):
    params = SimpleNamespace(
        dataset=data,
        pool_ratio=args.pool_ratio,
        num_pool=args.num_pool,
        linkpred=True,
        cuda=args.cuda,
        lr=1e-3,
        clip=2.0,
        batch_size=8,
        epoch=800,
        n_worker=8,
        gc_per_block=3,
        bn=True,
        dropout=0.0,
        bias=True,
        hidden_dim=20,
        output_dim=20
    )
    return params

def build_gin_params(args, data):
    params = SimpleNamespace(
        dataset=data,
        batch_size=32,
        cuda=args.cuda,
        net='gin',
        num_layers=5,
        num_mlp_layers=2,
        hidden_dim=64,
        graph_pooling_type=args.graph_pooling_type,
        neighbor_pooling_type=args.neighbor_pooling_type,
        learn_eps=args.learn_eps,
        degree_as_tag=args.degree_as_tag,
        epochs=350,
        lr=0.01,
        final_dropout=0.5
    )
    return params

def build_simple_params(args, data):
    params = SimpleNamespace(
        dataset=data,
        cuda=args.cuda,
        operator=args.operator,
        l2_norm=args.l2_norm
    )
    return params

def get_algorithm(args, data):
    if args.alg == "diffpool":
        params = build_diffpool_params(args, data)
        return diffpool_api(params)
    elif args.alg == "gin":
        params = build_gin_params(args, data)
        return gin_api(params)
    elif args.alg == "simple":
        params = build_simple_params(args, data)
        return simple_api(params)
    else:
        raise NotImplementedError

def save_features(feat_file, features_dict):
    dirr = '/'.join(feat_file.split('/')[:-1])
    if not os.path.isdir(dirr):
        os.makedirs(dirr)
    np.savez_compressed(feat_file, features=features_dict)

def init_features(args, data: TUDataset):
    elms = args.init.split("-")
    if len(elms) < 2:
        init = elms[0]
        normalizer = "pass"
    else:
        init, normalizer = elms[:2]
    print("Normalizer: ", normalizer)
    stime = time.time()
    if init == "ori": # use node attributes
        print("Init features: Original , node attributes")
        for idx_g, g in enumerate(data.graph_lists):
            feat_file = "feats/{}/{}-{}-seed{}.npz".format(args.dataset.split("/")[-1], idx_g,  
                args.init, args.seed)
            if os.path.isfile(feat_file):
                features_dict = np.load(feat_file, allow_pickle=True)['features'][()]
            else:
                idxs = list(g.nodes())
                features = data.node_attr[idxs, :]
                nodes = [x.item() for x in g.nodes()]
                features_dict = {x: features[i] for i, x in enumerate(nodes)}
                features_dict = lookup_normalizer[normalizer].norm(features_dict, g.to_networkx(), verbose=args.verbose)
                save_features(feat_file, features_dict)
            g.ndata['feat'] = np.array([features_dict[x] for x in nodes])
    elif init == "label": # use node label as node features
        print("Init features: node labels")
        for idx_g, g in enumerate(data.graph_lists):
            feat_file = "feats/{}/{}-{}-seed{}.npz".format(args.dataset.split("/")[-1], idx_g,  
                args.init, args.seed)
            if os.path.isfile(feat_file):
                features_dict = np.load(feat_file, allow_pickle=True)['features'][()]
            else:
                idxs = list(g.nodes())
                features = data.one_hot_node_labels[idxs, :]
                nodes = [x.item() for x in g.nodes()]
                features_dict = {x: features[i] for i, x in enumerate(nodes)}
                features_dict = lookup_normalizer[normalizer].norm(features_dict, g.to_networkx(), verbose=args.verbose)
                save_features(feat_file, features_dict)
            g.ndata['feat'] = np.array([features_dict[x] for x in nodes])
    else:
        print("Init features:", init)
        for idx_g, graph in enumerate(data.graph_lists):
            feat_file = "feats/{}/{}-{}-seed{}.npz".format(args.dataset.split("/")[-1], idx_g,  
                args.init, args.seed)
            if os.path.isfile(feat_file):
                features = np.load(feat_file, allow_pickle=True)['features'][()]
            else:
                features = get_feature_initialization(args, graph.to_networkx(), inplace=False, input_graph=True)
                save_features(feat_file, features)
            graph.ndata['feat'] = np.array([features[int(x)] for x in graph.nodes()])
    print("Time init features: {:.3f}s".format(time.time()-stime))

def dict2arr(dictt, graph):
    """
    Note: always sort graph nodes
    """
    dict_arr = torch.FloatTensor([dictt[x] for x in graph.nodes()])
    return dict_arr

def load_data(dataset):
    return TUDataset(dataset)

def main(args):
    data = load_data(args.dataset)
    init_features(args, data)
    alg = get_algorithm(args, data)

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
