import re 
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def parse(init, seed):
    logfile = "log/{}-{}-mmd-seed{}".format(data, init, seed)
    content = open(logfile).read()
    pval = re.findall("p_val: [-0-9\.\s]+", content)[0]
    pval = pval.replace("p_val: ", "")
    pval = float(pval)

    stats = re.findall("stats: [-0-9\.\s]+", content)[0]
    stats = stats.replace("stats: ", "")
    stats = float(stats)

    bandwidth = re.findall("bandwidth: [-0-9\.\s]+", content)[0]
    bandwidth = bandwidth.replace("bandwidth: ", "")
    bandwidth = float(bandwidth)

    try:
        samps = re.findall("samps: [-0-9\.\s]+", content)[0]
        samps = samps.replace("samps: ", "")
        samps = float(samps)
    except:
        samps = 0
    return pval, stats, bandwidth, samps

data = sys.argv[1]

inits = "ssvd1 ssvd0.5 deepwalk graphwave hope ori degree-standard triangle-standard kcore-standard egonet-standard pagerank-standard coloring-standard clique-standard".split()

print("Check ordered init methods:")
for i, init in enumerate(inits):
    print(i+1, init)

for init in inits:
    micros = []
    macros = []
    times = []
    sampss = []
    for seed in range(40, 50):
        try:
            pval, stats, bandwidth, samps = parse(init, seed)
            micros.append(pval)
            macros.append(stats)
            times.append(bandwidth)
            sampss.append(samps)
        except:
            pass
    micro = np.mean(micros)
    micro_std = np.std(micros)
    macro = np.mean(macros)
    macro_std = np.std(macros)
    time_init = np.mean(times)
    bandwidth_std = np.std(times)
    samps = np.mean(sampss)
    samps_std = np.std(samps)
    # print("Data: {} - Init: {}".format(data, init))
    print("{:.3f}+-{:.3f}\t{:.3f}+-{:.3f}\t{:.3f}+-{:.3f}\t{:.3f}+-{:.3f}".format(
        micro, micro_std, macro, macro_std, time_init, bandwidth_std, samps, samps_std))
