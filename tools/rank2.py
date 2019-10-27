f1s="""
0.126+-0.000	0.002+-0.000	0.506
0.206+-0.015	0.108+-0.006	0.326
0.844+-0.004	0.609+-0.005	0
0.729+-0.000	0.317+-0.000	6.59
0.741+-0.000	0.374+-0.001	6.896
0.685+-0.005	0.282+-0.005	0
0.806+-0.003	0.587+-0.007	0
OOM		
0.026+-0.000	0.001+-0.000	3.747
0.200+-0.000	0.007+-0.000	17.744
0.126+-0.000	0.002+-0.000	35.329
0.126+-0.000	0.002+-0.000	26.977
0.179+-0.000	0.008+-0.000	0.383
0.026+-0.000	0.001+-0.000	7324.36
paperspace	nan+-nan	nan
0.810+-0.002	0.466+-0.003	4.279
0.580+-0.000	0.180+-0.000	9.962
0.833+-0.000	0.603+-0.000	16.808
0.206+-0.015	0.108+-0.006	0
0.155+-0.003	0.005+-0.001	0
0.953+-0.000	0.855+-0.000	0.404
"""
inits="""
degree - standard
random uniform
DeepWalk
SVD (alpha = 0.5)
SVD (alpha = 1)
HOPE
LINE
GraphFactorization
triangles - standard
k-core numbers - standard
egonet edges number - standard
pagerank - standard
local coloring number - standard
largest clique number - standard
Identity (feat_dim=n_nodes)
Real feature 
Real feature - rowsum
Real feature - standard
Learnable random uniform
Graphwave
Node label one hot
Degree one hot
Struc2vec
""".strip().split('\n')
inits = [x.strip() for x in inits]
# convert to init mmd 
convert = {
    "degree - standard": "degree-standard",
    # "SVD (alpha = 0.5)": "SVD (alpha=0.5)",
    # "SVD (alpha = 1)": "SVD (alpha=1)",
    "triangles - standard": "triangle-standard",
    "k-core numbers - standard": "kcore-standard",
    "egonet edges number - standard": "egonet-standard",
    "pagerank - standard": "pagerank-standard",
    "local coloring number - standard": "coloring-standard",
    "largest clique number - standard": "clique-standard",
    "Graphwave": "GraphWave"
}
inits = [x if x not in convert else convert[x] for x in inits]
print(inits)

inits_mmd = """
SVD (alpha = 1)
SVD (alpha = 0.5)
DeepWalk
GraphWave
HOPE
Real feature
degree-standard 
triangle-standard
kcore-standard
egonet-standard
pagerank-standard
coloring-standard
clique-standard""".strip().split('\n')
inits_mmd = [x.strip() for x in inits_mmd]

f1s = f1s.strip().split('\n')
f1s = [x.split('\t')[0].split('+')[0] for x in f1s]


consider = {}
import numpy as np
f1micros = []

for init in inits_mmd:
    idx_init = inits.index(init)
    # consider[init] = f1s[idx_init]
    try:
        f1micros.append(float(f1s[idx_init]))
    except:
        f1micros.append(0)
ranks = np.argsort(-np.array(f1micros))

for i in range(len(inits_mmd)):
    print("{}\t{}\t{}".format(inits_mmd[i], f1micros[i], ranks.tolist().index(i)+1))
    # print()
