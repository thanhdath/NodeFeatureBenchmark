
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
from dataloader.tu_dataset import TUDataset
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

def init_features(init, feature_size, seed, dataname, args):
    try:
        data = TUDataset()


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
                features_dict = np.load(feat_file)['features'][()]
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
                features_dict = np.load(feat_file)['features'][()]
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
                features = np.load(feat_file)['features'][()]
            else:
                features = get_feature_initialization(args, graph.to_networkx(), inplace=False)
                save_features(feat_file, features)
            graph.ndata['feat'] = np.array([features[int(x)] for x in graph.nodes()])
    print("Time init features: {:.3f}s".format(time.time()-stime))


def main(args):
    # inits = "degree-standard uniform deepwalk ssvd0.5 ssvd1 hope line gf triangle-standard kcore-standard egonet-standard pagerank-standard coloring-standard clique-standard".split()
    # inits_many = "uniform deepwalk ssvd0.5 ssvd1 hope line gf pagerank-standard".split()
    # inits_one = "ori ori-rowsum ori-standard degree-standard triangle-standard kcore-standard egonet-standard clique-standard coloring-standard".split()
    inits_many = "".split()
    inits_one = "pagerank-standard triangle-standard kcore-standard coloring-standard clique-standard".split()

    if args.dataset.endswith("/"):
        args.dataset = args.dataset[:-1]
    params = [(init, args.feature_size, seed, args.dataset, args)
        for init in inits_many for seed in range(40, 43)]
    params += [(init, args.feature_size, 40, args.dataset, args)
        for init in inits_one]
    np.random.shuffle(params)
    pool = MyPool(3)
    pool.starmap(init_features, params)
    pool.close()
    pool.join()

    for param in params:
        init_features(*param)

if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
