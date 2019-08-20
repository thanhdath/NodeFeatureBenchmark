import scipy.sparse as sp
import numpy as np
import dgl
import os, sys
from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url
import networkx as nx
import torch
from .custom_dgl_graph import DGLGraph

class RedditInductiveDataset(object):
    def __init__(self, mode, self_loop=False, use_networkx=False):
        """
        mode: one of train val test
        """
        self.data_dir = "data/reddit"
        if self_loop:
            self.data_dir += "_self_loop"
        npz = np.load(os.path.join(self.data_dir, "{}_graph.npz".format(mode)), allow_pickle=True)
        coo_adj = npz['graph'][()]
        labels = npz['labels'][()]
        if mode == "train":
            self.main_nodes = np.arange(coo_adj.shape[0])
        else:
            self.main_nodes = npz["{}_nodes".format(mode)][()]
        
        if use_networkx:
            self.graph = nx.from_scipy_sparse_matrix(coo_adj)
        else:
            self.graph = DGLGraph(coo_adj, suffix="-{}".format(mode), readonly=True)
        features = np.load(self.data_dir+'/{}_feats.npz'.format(mode), allow_pickle=True)['feats'][()]
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.n_classes = int(self.labels.max() + 1)
        self.multiclass = False
    

def process(self_loop=False):
    # download_dir = get_download_dir()
    download_dir = "data/"
    self_loop_str = ""
    if self_loop:
        self_loop_str = "_self_loop"
    zip_file_path = os.path.join(download_dir, "reddit{}.zip".format(self_loop_str))
    download(_get_dgl_url("dataset/reddit{}.zip".format(self_loop_str)), path=zip_file_path)
    extract_dir = os.path.join(download_dir, "reddit{}".format(self_loop_str)) 
    extract_archive(zip_file_path, extract_dir)
    # graph
    coo_adj = sp.load_npz(os.path.join(extract_dir, "reddit{}_graph.npz".format(self_loop_str)))
    # graph = nx.from_scipy_sparse_matrix(coo_adj)
    graph = DGLGraph(coo_adj, readonly=True)
    # graph =
    import numpy as np
    reddit_data = np.load(os.path.join(extract_dir, "reddit_data.npz"))
    features = reddit_data["feature"]
    labels = reddit_data["label"]
    # tarin/val/test indices
    node_ids = reddit_data["node_ids"]
    node_types = reddit_data["node_types"]
    # subgraph
    train_indices = np.argwhere(node_types == 1).flatten()
    val_indices = np.argwhere(node_types == 2).flatten()
    test_indices = np.argwhere(node_types == 3).flatten()

    train_graph = graph.subgraph(train_indices)
    val_graph = graph.subgraph(np.hstack([train_indices, val_indices]))
    val_nodes = np.array([int(x) for x in val_graph.nodes()][len(train_indices):])
    test_graph = graph
    test_nodes = test_indices

    # test 

    print("Info")
    print("train graph - nodes {} - edges {}".format(train_graph.number_of_nodes(),
        train_graph.number_of_edges()))
    print("val graph - nodes {} - edges {}".format(val_graph.number_of_nodes(),
        val_graph.number_of_edges()))
    print("test graph - nodes {} - edges {}".format(test_graph.number_of_nodes(),
        test_graph.number_of_edges()))
    from networkx.readwrite import json_graph
    import json
    train_labels = labels[train_indices]
    val_labels = labels[np.hstack([train_indices, val_indices])]
    test_labels = labels


    train_adj = train_graph.adjacency_matrix_scipy()
    np.savez_compressed(extract_dir+'/train_graph.npz', graph=train_adj, labels=train_labels)
    val_adj = val_graph.adjacency_matrix_scipy()
    np.savez_compressed(extract_dir+'/valid_graph.npz', graph=val_adj, 
        valid_nodes=val_nodes, labels=val_labels)
    test_adj = test_graph.adjacency_matrix_scipy()
    np.savez_compressed(extract_dir+'/test_graph.npz', graph=test_adj, 
        test_nodes=test_nodes, labels=test_labels)
    

    train_features = features[train_indices]
    val_features = features[np.hstack([train_indices, val_indices])]
    test_features = features
    np.savez_compressed(extract_dir+'/train_feats.npz', feats=train_features)
    np.savez_compressed(extract_dir+'/valid_feats.npz', feats=val_features)
    np.savez_compressed(extract_dir+'/test_feats.npz', feats=test_features)

if __name__ == '__main__':
    process()
    process(self_loop=True)
