"""
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
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

graph_method = sys.argv[1]
transfer_from = None
if len(sys.argv) >= 3:
    transfer_from = sys.argv[2]
data = TUDataset(f"data/torus_vs_sphere-{graph_method}", ratio=[.8, .2, .0])


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
    model_name = f"gin-{graph_method}.pkl"
else:
    model_name = f"gin-{graph_method}-transfer-from-{transfer_from}.pkl"
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
    epochs=200,
    lr=0.01,
    final_dropout=0.5,
    model_name=model_name,
    transfer_from=transfer_from
)
init_features(data)

gin_api(gin_params)
