from types import SimpleNamespace
import argparse
import numpy as np
import networkx as nx
from dataloader.tu_dataset import TUDataset
import torch
import random
from parser import *
from main import get_feature_initialization
from normalization import lookup as lookup_normalizer
import time
import os
import shogun as sg
from graph_classify import init_features

def parse_args():
    parser = argparse.ArgumentParser(description="Node feature initialization benchmark.")
    parser.add_argument('--dataset', default="data/ENZYMES")
    parser.add_argument('--init', default="ori")
    parser.add_argument('--feature_size', default=128, type=int)
    # args.add_argument('--train_features', action='store_true')
    parser.add_argument('--shuffle', action='store_true', help="Whether shuffle features or not.")
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--verbose', type=int, default=0)
    return parser.parse_args()

def test_mmd(emb1, emb2):
    p_val, stat, samps, bandwidth = rbf_mmd_test(np.asarray(
        emb1).astype("float64"), np.asarray(emb2).astype("float64"))
    print("p_val:", p_val)
    print("stats:", stat)
    print("bandwidth:", bandwidth)

def rbf_mmd_test(X, Y, bandwidth='median', null_samples=1000,
                 median_samples=1000, cache_size=32):
    '''
    Run an MMD test using a Gaussian kernel.
    Parameters
    ----------
    X : row-instance feature array
    Y : row-instance feature array
    bandwidth : float or 'median'
        The bandwidth of the RBF kernel (sigma).
        If 'median', estimates the median pairwise distance in the
        aggregate sample and uses that.
    null_samples : int
        How many times to sample from the null distribution.
    median_samples : int
        How many points to use for estimating the bandwidth.
    Returns
    -------
    p_val : float
        The obtained p value of the test.
    stat : float
        The test statistic.
    null_samples : array of length null_samples
        The samples from the null distribution.
    bandwidth : float
        The used kernel bandwidth
    '''

    if bandwidth == 'median':
        from sklearn.metrics.pairwise import euclidean_distances

        def sub(feats, n): return feats[np.random.choice(
            feats.shape[0], min(feats.shape[0], n), replace=False)]
        Z = np.r_[sub(X, median_samples // 2), sub(Y, median_samples // 2)]
        D2 = euclidean_distances(Z, squared=True)
        upper = D2[np.triu_indices_from(D2, k=1)]
        kernel_width = np.median(upper, overwrite_input=True)
        bandwidth = np.sqrt(kernel_width / 2)
        # sigma = median / sqrt(2); works better, sometimes at least
        del Z, D2, upper
    else:
        kernel_width = 2 * bandwidth ** 2

    mmd = sg.QuadraticTimeMMD()
    mmd.set_p(sg.RealFeatures(X.T.astype(np.float64)))
    mmd.set_q(sg.RealFeatures(Y.T.astype(np.float64)))
    mmd.set_kernel(sg.GaussianKernel(cache_size, kernel_width))

    mmd.set_num_null_samples(null_samples)
    samps = mmd.sample_null()
    stat = mmd.compute_statistic()

    p_val = np.mean(stat <= samps)
    return p_val, stat, samps, bandwidth

def load_data(dataset):
    return TUDataset(dataset)

def main(args):
    data = load_data(args.dataset)
    init_features(args, data)
    import pdb; pdb.set_trace()
    train_data = data.dataset_train
    test_data = data.dataset_test

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
