from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(nn.Module):
    """
    Uniformly samples neighbors.
    Samples from a dense 2D edgelist of neighbors of nodes, which looks like

        [
            [1, 2, 3, ..., 1],
            [1, 3, 3, ..., 3],
            ...
        ]

    stored as torch.LongTensor.

    This relies on a preprocessing step where we sample _exactly_ K neighbors
    for each node -- if the node has less than K neighbors, we upsample w/ replacement
    and if the node has more than K neighbors, we downsample w/o replacement.

    Padding adj lists with random re-sampling help speeding up the

    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def sample(self, ids, num_samples):
        #Select adj_lists (neighbors of batch nodes) from adj_info
        adj_lists = self.adj_info[ids,:]   # [256, 25]
        if adj_lists.shape[0] == 0:
            return adj_lists
        #Shuffle along neighbors
        shuf_ids = torch.randperm(adj_lists.shape[1])
        adj_lists = adj_lists[:,shuf_ids]
        #Sample num_samples case for each node in batch nodes
        adj_lists = adj_lists[:, :num_samples]
        return adj_lists

