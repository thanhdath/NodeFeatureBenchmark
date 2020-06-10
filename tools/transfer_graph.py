"""
    for seed in 100 101 102 103 104
    do
        echo $seed 
        python -u tools/transfer_graph.py --graph-method knn --seed $seed --epochs 100 > logs/knn-sigmoid/knn-seed$seed.log
        python -u tools/transfer_graph.py --graph-method sigmoid --seed $seed --epochs 100 > logs/knn-sigmoid/sigmoid-seed$seed.log
        python -u tools/transfer_graph.py --graph-method knn --seed $seed \
            --transfer-from gin-sigmoid-seed$seed.pkl --epochs 0 > logs/knn-sigmoid/knn-from-sigmoid-seed$seed.log
        python -u tools/transfer_graph.py --graph-method sigmoid --seed $seed \
            --transfer-from gin-knn-seed$seed.pkl --epochs 0 > logs/knn-sigmoid/sigmoid-from-knn-seed$seed.log
    done

    for i in knn sigmoid
    do
        PYTHONPATH="." python -u tools/transfer_graph.py $i > logs/transfer-gin/$i.log
    done

    for i in knn sigmoid
    do
        for j in knn sigmoid
        do
            PYTHONPATH="." python -u tools/transfer_graph.py $i gin-$j.pkl > logs/transfer-gin/$i-transfer-from-$j.log
        done
    done

    Noise:
    PYTHONPATH="." python -u tools/transfer_graph.py knn

    mkdir logs
    mkdir logs/transfer-gin
    for t in $(seq 1 20)
    do 
        rm *.pkl
        mkdir logs/transfer-gin/$t
        python tools/torus_sphere.py

        for i in 0.0 0.0001 0.001 0.01
        do
            PYTHONPATH="." python -u tools/transfer_graph.py knn-n$i > logs/transfer-gin/$t/knn-n$i.log
        done

        for i in 0.0 0.0001 0.001 0.01
        do
            PYTHONPATH="." python -u tools/transfer_graph.py knn-n0.0 gin-knn-n$i.pkl > logs/transfer-gin/$t/knn-transfer-from-knn-n$i.log
        done

        for i in 0.0001 0.001 0.01
        do
            PYTHONPATH="." python -u tools/transfer_graph.py knn-n$i gin-knn-n0.0.pkl > logs/transfer-gin/$t/knn-n$i-transfer-from-knn.log
        done
    done
    
"""

from dataloader.tu_dataset import TUDataset
# %load_ext autoreload
# %autoreload 2
import os 
import sys 
import argparse
import numpy as np
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--graph-method', default='knn')
parser.add_argument('--transfer-from', default=None)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()
print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

graph_method = args.graph_method
transfer_from = args.transfer_from
data = TUDataset(f"tools/data/torus_vs_sphere-{graph_method}-n0.0-seed{args.seed}", ratio=[.8, .1, .1])


from algorithms.graph_embedding import *
from types import SimpleNamespace
from normalization import lookup as lookup_normalizer
import numpy as np

def init_features(data):
    print("Init features: Original , node attributes")
    for idx_g, g in enumerate(data.graph_lists):
        idxs = list(g.nodes())
        features = data.node_attr[idxs, :]
        nodes = [x.item() for x in g.nodes()]
        features_dict = {x: features[i] for i, x in enumerate(nodes)}
#         features_dict = lookup_normalizer["standard"].norm(features_dict, g.to_networkx(), verbose=False)
        g.ndata['feat'] = np.array([features_dict[int(x)] for x in g.nodes()])

if transfer_from is None:
    model_name = f"gin-{graph_method}-seed{args.seed}.pkl"
else:
    model_name = f"gin-{graph_method}-transfer-from-{transfer_from.split('.')[0]}-seed{args.seed}.pkl"
gin_params = SimpleNamespace(
    dataset=data,
    batch_size=32,
    cuda=True,
    net='gin',
    num_layers=2, # 5
    num_mlp_layers=2,
    hidden_dim=64,
    graph_pooling_type="sum",
    neighbor_pooling_type="sum",
    learn_eps=False,
    degree_as_tag=False,
    epochs=args.epochs,
    lr=0.01,
    final_dropout=0.5,
    model_name=model_name,
    transfer_from=transfer_from
)
init_features(data)

gin_api(gin_params)
