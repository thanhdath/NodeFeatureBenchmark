import re 
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def parse(init, seed):
    logfile = "log/{}/{}-{}-seed{}".format(alg, data, init, seed)
    content = open(logfile).read()
    f1 = re.findall("micro-macro: [0-9\.\s]+", content)[0]
    f1 = f1.replace("micro-macro: ", "")
    micro, macro = [float(x) for x in f1.split()]

    time_inits = re.findall("Time init features: [0-9\.]+", content)   
    time_inits = [x.replace("Time init features: ", "") for x in time_inits]
    time_init = sum(map(float, time_inits))
    return micro, macro, time_init

data = sys.argv[1]
try:
    alg = sys.argv[2]
except:
    alg = "sgc"

if data in "bc flickr wiki".split():
    if alg == "nope":
        inits = "deepwalk hope line gf".split()
    else:
        inits = "degree-standard uniform deepwalk ssvd0.5 ssvd1 hope line gf triangle-standard kcore-standard egonet-standard pagerank-standard coloring-standard clique-standard identity".split()
elif "ppi" in data or "inductive" in data:
    inits = "degree-standard uniform deepwalk ssvd0.5 ssvd1 hope line gf triangle-standard kcor-standard egonet-standard pagerank-standard coloring-standard clique-standard identity ori ori-rowsum ori-standard learnable".split()
else:
    if alg == "nope":
        inits = "ori ori-rowsum ori-standard deepwalk hope node2vec line gf \
            deepwalk-standard hope-standard node2vec-standard line-standard gf-standard graphwave".split()
    elif alg in ["diffpool", "gin"]:
        inits = "degree-standard uniform deepwalk node2vec ssvd0.5 ssvd1 hope line \
            gf triangle-standard kcore-standard egonet-standard pagerank-standard coloring-standar \
            clique-standard graphlet identity ori ori-rowsum ori-standard label label-standard graphwave".split()
    else:
        inits = "degree-standard uniform deepwalk node2vec ssvd0.5 ssvd1 hope \
                line gf deepwalk-standard node2vec-standard ssvd0.5-standard ssvd1-standard hope-standard line-standard \
                gf-standard \
                triangle-standard kcore-standard egonet-standard pagerank-standard coloring-standard \
                clique-standard graphlet identity ori ori-rowsum ori-standard label graphwave".split()
    
print("Check ordered init methods:")
for i, init in enumerate(inits):
    print(i+1, init)

for init in inits:
    micros = []
    macros = []
    times = []
    for seed in range(40, 50):
        try:
            micro, macro, time_init = parse(init, seed)
            micros.append(micro)
            macros.append(macro)
            times.append(time_init)
        except:
            pass
    micro = np.mean(micros)
    micro_std = np.std(micros)
    macro = np.mean(macros)
    macro_std = np.std(macros)
    time_init = np.mean(times)
    # print("Data: {} - Init: {}".format(data, init))
    print("{:.3f}+-{:.3f}\t{:.3f}+-{:.3f}\t{:.3f}".format(
        micro, micro_std, macro, macro_std, time_init))