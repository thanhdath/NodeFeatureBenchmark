import re 
import sys
import numpy as np

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

for init in "degree uniform deepwalk node2vec ssvd0.5 ssvd1 hope triangle kcore egonet pagerank coloring clique identity ori label".split():
    micros = []
    macros = []
    times = []
    for seed in range(40, 51):
        try:
            micro, macro, time_init = parse(init, seed)
            micros.append(micro)
            macros.append(macro)
            times.append(time_init)
        except:
            pass
    micro = np.mean(micros)
    macro = np.mean(macros)
    time_init = np.mean(times)
    print("Data: {} - Init: {}".format(data, init))
    print("\tMicro-Macro-Time: {:.3f}\t{:.3f}\t{:.3f}".format(micro, macro, time_init))