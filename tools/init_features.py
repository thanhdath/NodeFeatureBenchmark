
import argparse
import numpy as np 
import random 
import torch
import networkx as nx 
import multiprocessing
from features_init import lookup as lookup_feature_init
from main import add_weight
import time
from shutil import copyfile 
from dataloader.reddit_dataloader import RedditDataset
# from dataloader.ppi_dataloader import PPIDataset

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
    # parser.add_argument('--seed', type=int, default=40)
    return parser.parse_args()

def get_feature_initialization(init_norm, feature_size, seed, data_name, args, inplace=True, shuffle=False):
    try:
        if "reddit" in args.dataset:
            data = RedditDataset(self_loop=("self_loop" in data_name), use_networkx=True)
        graph = data.graph
    
        print("init: {} - seed {}".format(init_norm, seed))
        np.random.seed(seed)
        elms = init_norm.split("-")
        if len(elms) < 2:
            init = elms[0]
            normalizer = "pass"
        else:
            init, normalizer = elms[:2]
        if init not in lookup_feature_init:
            raise NotImplementedError
        kwargs = {}
        if init == "ori":
            kwargs = {"feature_path": "data/"+data_name + "/features.npy"}
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
        stime = time.time()
        init_feature = lookup_feature_init[init](**kwargs)
        features = init_feature.generate(graph, feature_size,
                                    inplace=False, normalizer=normalizer,  verbose=0,
                                    shuffle=shuffle)
        import os 
        if not os.path.isdir('feats'):
            os.makedirs('feats')
        feat_file = 'feats/{}-{}-seed{}.npz'.format(data_name, init_norm, seed)
        np.savez_compressed(feat_file, features=features)
        print("Time init features {} : {:.3f} s".format(init_norm, time.time()-stime))
    except Exception as err:
        print(err)


def main(args):
    # inits = "degree-standard uniform deepwalk ssvd0.5 ssvd1 hope line gf triangle-standard kcore-standard egonet-standard pagerank-standard coloring-standard clique-standard".split()
    # inits_many = "uniform deepwalk ssvd0.5 ssvd1 hope line gf pagerank-standard".split()
    # inits_one = "ori ori-rowsum ori-standard degree-standard triangle-standard kcore-standard egonet-standard clique-standard coloring-standard".split()
    inits_many = "".split()
    inits_one = "triangle-standard kcore-standard egonet-standard pagerank-standard coloring-standard clique-standard".split()

    if args.dataset.endswith("/"):
        args.dataset = args.dataset[:-1]
    dataname = args.dataset.split('/')[-1]
    params = [(init, args.feature_size, seed, dataname, args)
        for init in inits_many for seed in range(40, 43)]
    params += [(init, args.feature_size, 40, dataname, args)
        for init in inits_one]
    # np.random.shuffle(params)
    pool = MyPool(1)
    pool.starmap(get_feature_initialization, params)
    pool.close()
    pool.join()

    for param in params:
        get_feature_initialization(*param)

if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
