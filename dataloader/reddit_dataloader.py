import scipy.sparse as sp
import numpy as np
import dgl
import os
import sys
from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url
import networkx as nx
import torch
from .custom_dgl_graph import DGLGraph
try:
    from networkit import *
except:
    print("Warning: cannot import networkit. Install by command: pip install networkit")


class RedditDataset(object):
    def __init__(self, self_loop=False):
        # download_dir = get_download_dir()
        download_dir = "data/"
        self_loop_str = ""
        if self_loop:
            self_loop_str = "_self_loop"
        zip_file_path = os.path.join(download_dir, "reddit{}.zip".format(self_loop_str))
        download(_get_dgl_url("dataset/reddit{}.zip".format(self_loop_str)), path=zip_file_path)
        extract_dir = os.path.join(download_dir, "reddit{}".format(self_loop_str))
        first_time = not os.path.isdir(extract_dir)
        extract_archive(zip_file_path, extract_dir)
        self.datadir = extract_dir
        # graph
        coo_adj = sp.load_npz(os.path.join(
            extract_dir, "reddit{}_graph.npz".format(self_loop_str)))
        self.graph = DGLGraph(coo_adj, suffix="", readonly=True)

        # features and labels
        reddit_data = np.load(os.path.join(extract_dir, "reddit_data.npz"))
        self.features = torch.FloatTensor(reddit_data["feature"])
        self.labels = torch.LongTensor(reddit_data["label"])
        self.num_labels = 41
        # tarin/val/test indices
        self.node_ids = reddit_data["node_ids"]
        node_types = reddit_data["node_types"]
        self.train_mask = torch.ByteTensor(node_types == 1)
        self.val_mask = torch.ByteTensor(node_types == 2)
        self.test_mask = torch.ByteTensor(node_types == 3)
        self.multiclass = False
        self.n_classes = self.num_labels

        feature_file = extract_dir + '/features.npz'
        if not os.path.isfile(feature_file):
            features = self.features.numpy()
            features_dict = {int(node): features[i]
                             for i, node in enumerate(self.node_ids)}
            np.savez_compressed(feature_file, features=features_dict)

        label_file = extract_dir + '/labels.npz'
        if not os.path.isfile(label_file):
            labels = reddit_data["label"]
            labels_dict = {int(node): np.array([labels[i]])
                           for i, node in enumerate(self.node_ids)}
            np.savez_compressed(label_file, labels=labels_dict, is_multiclass=False)

        print('Finished data loading.')
        print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
        print('  NumEdges: {}'.format(self.graph.number_of_edges()))
        print('  NumFeats: {}'.format(self.features.shape[1]))
        print('  NumClasses: {}'.format(self.num_labels))
        print('  NumTrainingSamples: {}'.format(len(np.nonzero(self.train_mask)[0])))
        print('  NumValidationSamples: {}'.format(len(np.nonzero(self.val_mask)[0])))
        print('  NumTestSamples: {}'.format(len(np.nonzero(self.test_mask)[0])))

    def graph_networkit(self):
        if not hasattr(self, 'graph_nit'):
            edgelist_file = self.datadir + "/edgelist_networkit.txt"
            adj = self.graph.adj
            if not os.path.isfile(edgelist_file):
                with open(edgelist_file, "w+") as fp:
                    fp.write("{} {}".format(adj.shape[0], int(adj.sum())))
                    for i in range(adj.shape[0]):
                        fp.write(
                            "\n" + " ".join(map(lambda x: str(x+1), adj[i].nonzero()[1])))
            self.graph_nit = readGraph(edgelist_file, Format.METIS)
        return self.graph_nit

    def graph_networkx(self):
        if not hasattr(self, 'graph_nx'):
            self.graph_nx = nx.from_scipy_sparse_matrix(self.graph.adj)
        return self.graph_nx
