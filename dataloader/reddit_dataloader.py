import scipy.sparse as sp
import numpy as np
import dgl
import os, sys
from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url
import networkx as nx
from .custom_dgl_graph import DGLGraph

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
        # graph
        coo_adj = sp.load_npz(os.path.join(extract_dir, "reddit{}_graph.npz".format(self_loop_str)))
        # self.graph = nx.from_scipy_sparse_matrix(coo_adj)
        self.graph = DGLGraph(coo_adj, readonly=True)
        # features and labels
        reddit_data = np.load(os.path.join(extract_dir, "reddit_data.npz"))
        self.features = reddit_data["feature"]
        self.labels = reddit_data["label"]
        self.num_labels = 41
        # tarin/val/test indices
        self.node_ids = reddit_data["node_ids"]
        node_types = reddit_data["node_types"]
        self.train_mask = (node_types == 1)
        self.val_mask = (node_types == 2)
        self.test_mask = (node_types == 3)

        if first_time:
            features_dict = {int(node): self.features[i] for i, node in enumerate(self.node_ids)}
            np.save(extract_dir + '/features.npy', features_dict)
 
        print('Finished data loading.')
        print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
        print('  NumEdges: {}'.format(self.graph.number_of_edges()))
        print('  NumFeats: {}'.format(self.features.shape[1]))
        print('  NumClasses: {}'.format(self.num_labels))
        print('  NumTrainingSamples: {}'.format(len(np.nonzero(self.train_mask)[0])))
        print('  NumValidationSamples: {}'.format(len(np.nonzero(self.val_mask)[0])))
        print('  NumTestSamples: {}'.format(len(np.nonzero(self.test_mask)[0])))
