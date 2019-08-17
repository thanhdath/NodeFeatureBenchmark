
import argparse
import numpy as np 
import random 
import torch
import networkx as nx 
import multiprocessing
from features_init import lookup as lookup_feature_init
from main import add_weight

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre init features.")
    parser.add_argument('--dataset', default="data/cora")
    parser.add_argument('--feature_size', default=128, type=int)
    parser.add_argument('--seed', type=int, default=40)
    return parser.parse_args()

def get_feature_initialization(init_norm, feature_size, seed, data_name, graph, inplace=True, shuffle=False):
    print("init: {} - seed {}".format(init_norm, seed))
    elms = init_norm.split("-")
    if len(elms) < 2:
        init = elms[0]
        normalizer = "pass"
    else:
        init, normalizer = elms[:2]
    if init not in lookup_feature_init:
        raise NotImplementedError
    kwargs = {}
    if init == "ssvd0.5":
        init = "ssvd"
        kwargs = {"alpha": 0.5}
    elif init == "ssvd1":
        init = "ssvd"
        kwargs = {"alpha": 1}
    elif init in ["gf", "node2vec"]:
        add_weight(graph)

    try:
        init_feature = lookup_feature_init[init](**kwargs)
        features = init_feature.generate(graph, feature_size,
                                    inplace=inplace, normalizer=normalizer, 
                                    shuffle=shuffle)
        import os 
        if not os.path.isdir('feats'):
            os.makedirs('feats')
        feat_file = 'feats/{}-{}-seed{}.npz'.format(data_name, init_norm, seed)
        np.savez_compressed(feat_file, features=features)
    except Exception as err:
        print(err)

def main(args):
    graph = nx.read_edgelist(args.dataset + '/edgelist.txt', nodetype=int)
    inits = "degree-standard uniform deepwalk ssvd0.5 ssvd1 hope line gf triangle-standard kcore-standard egonet-standard pagerank-standard coloring-standard clique-standard".split()
    dataname = args.dataset.split('/')[-1]
    params = [(init, args.feature_size, seed, dataname, graph)
        for init in inits for seed in range(40, 43)]
    pool = MyPool(2)
    pool.starmap(get_feature_initialization, params)
    pool.close()
    pool.join()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
