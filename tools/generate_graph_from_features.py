from dataloader import CitationDataloader, DefaultDataloader
import torch 
import torch.nn.functional as F
import sys
import pdb
import argparse
import numpy as np
import os
from shutil import copyfile

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", required=True, help="Dataset name")
    parser.add_argument("--type", choices=["sigmoid", "knn"], default="sigmoid")
    parser.add_argument("--out_dir", required=True)

    # If type = sigmoid
    parser.add_argument("--threshold", default=None, type=float, 
        help="If None, threshold = scores.mean().")
    # If type = knn
    parser.add_argument("--k", default=10, type=int)
    return parser.parse_args()

def load_data(dataset):
    data_name = dataset.split('/')[-1]
    if data_name in ["citeseer", "pubmed"]:
        return CitationDataloader(dataset, random_split=False)
    # elif data_name == "reddit":
    #     return RedditDataset(self_loop=False)
    # elif data_name == "reddit_self_loop":
    #     return RedditDataset(self_loop=True)
    # elif data_name == "NELL":
    #     return NELLDataloader(dataset)
    else:
        # cora bc flickr wiki youtube homo-sapiens
        return DefaultDataloader(dataset, random_split=False)

def generate_graph(features, kind="sigmoid", threshold=None, k=5, 
    out_dir="./generated_graph"):
    features_norm = F.normalize(features, dim=1)
    scores = features_norm.mm(features_norm.t())
    print(f"Generate graph using {kind}")
    if kind == "sigmoid":
        scores = torch.sigmoid(scores)
        if threshold is None:
            threshold = scores.mean()
        print(f"Scores range: {scores.min()}-{scores.max()}")
        print("Threshold: ", threshold)
        adj = scores > threshold
        adj = adj.int()
        edge_index = adj.nonzero().cpu().numpy()
    elif kind == "knn":
        print(f"Knn k = {k}")
        sorted_scores = torch.argsort(-scores, dim=1)[:, :k]
        edge_index = np.zeros((len(scores)*k, 2))
        N = len(scores)
        for i in range(k):
            edge_index[i*N:(i+1)*N, 0] = np.arange(N)
            edge_index[i*N:(i+1)*N, 1] = sorted_scores[:, i]
    else:
        raise NotImplementedError
    
    print("Number of edges: ", edge_index.shape[0])
    return edge_index

def dict2arr(dictt, graph):
    """
    Note: always sort graph nodes
    """
    dict_arr = torch.FloatTensor([dictt[int(x)] for x in graph.nodes()])
    return dict_arr

def arr2dict(arr, graph):
    dictt = {}
    for i in range(len(arr)):
        dictt[i] = arr[i]
    return dictt

if __name__ == '__main__':
    args = parse_args()
    data = load_data("data/" + args.dataname)
    features_path = "data/" + args.dataname + "/features.npz"
    features = np.load(features_path, allow_pickle=True)["features"][()]
    features = dict2arr(features, data.graph)
    edge_index = generate_graph(features, args.type, 
        threshold=args.threshold, k=args.k, out_dir=args.out_dir)

    out_dir = args.out_dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    np.savetxt(out_dir+"/edgelist.txt", edge_index, delimiter=",")

    copyfile(features_path, f"{out_dir}/features.npz")
    copyfile("data/" + args.dataname + "/labels.txt", f"{out_dir}/labels.txt")
    
    print("Graph has been saved to", out_dir)
