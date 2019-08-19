
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
from dataloader.reddit_inductive_dataloader import RedditInductiveDataset
from dataloader.ppi_dataloader import PPIDataset

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

def get_feature_initialization(init_norm, feature_size, seed, mode, data_name, args, inplace=True, shuffle=False):
    try:
        if "reddit" in args.dataset:
            data = RedditInductiveDataset(mode, self_loop=("self_loop" in args.dataset))
        elif "ppi" in args.dataset:
            data = PPIDataset(mode)
        graph = data.graph
        stime = time.time()
        if "reddit" in data_name:
            inplace = False
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
            kwargs = {"feature_path": "data/"+data_name + "/{}_feats.npz".format(mode)}
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
        features = init_feature.generate(graph, feature_size,
                                    inplace=inplace, normalizer=normalizer,  verbose=0,
                                    shuffle=shuffle)
        import os 
        if not os.path.isdir('feats'):
            os.makedirs('feats')
        feat_file = 'feats/{}-{}-{}-seed{}.npz'.format(data_name, mode, init_norm, seed)
        np.savez_compressed(feat_file, features=features)
        print("Time init features {} : {:.3f} s".format(init_norm, time.time()-stime))
    except Exception as err:
        print(err)


def main(args):
    for mode in 'train valid test'.split():
        # inits = "degree-standard uniform deepwalk ssvd0.5 ssvd1 hope line gf triangle-standard kcore-standard egonet-standard pagerank-standard coloring-standard clique-standard".split()
        inits_many = "uniform deepwalk ssvd0.5 ssvd1 hope line gf pagerank-standard".split()
        inits_one = "ori ori-rowsum ori-standard degree-standard triangle-standard kcore-standard egonet-standard clique-standard coloring-standard".split()
        if args.dataset.endswith("/"):
            args.dataset = args.dataset[:-1]
        dataname = args.dataset.split('/')[-1]
        params = [(init, args.feature_size, seed, mode, dataname, args)
            for init in inits_many for seed in range(40, 43)]
        params += [(init, args.feature_size, 40, mode, dataname, args)
            for init in inits_one]
        np.random.shuffle(params)
        pool = MyPool(3)
        pool.starmap(get_feature_initialization, params)
        pool.close()
        pool.join()

    # for init in inits_one:
    #     try:
    #         feat_file = 'feats/{}-{}-seed{}.npz'.format(dataname, init, 40)
    #         copyfile(feat_file, feat_file.replace("seed40", "seed41"))
    #         copyfile(feat_file, feat_file.replace("seed40", "seed41"))
    #     except:
    #         pass

if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
