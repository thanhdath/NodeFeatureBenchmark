import re 
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def parse(init, seed):
    logfile = "log/{}/{}-{}-mmd-seed{}".format(alg, data, init, seed)
    content = open(logfile).read()
    pval = re.findall("pval: [0-9\.\s]+", content)[0]
    pval = pval.replace("pval: ", "")
    pval = float(pval)

    stats = re.findall("stats: [0-9\.\s]+", content)[0]
    stats = stats.replace("stats: ", "")
    stats = float(stats)

    bandwidth = re.findall("bandwidth: [0-9\.\s]+", content)[0]
    bandwidth = bandwidth.replace("bandwidth: ", "")
    bandwidth = float(bandwidth)
    return pval, stats, bandwidth

data = sys.argv[1]
try:
    alg = sys.argv[2]
except:
    alg = "sgc"

inits = "ssvd1 ssvd0.5 deepwalk graphwave hope ori".split()

print("Check ordered init methods:")
for i, init in enumerate(inits):
    print(i+1, init)

for init in inits:
    micros = []
    macros = []
    times = []
    for seed in range(40, 50):
        try:
            pval, stats, bandwidth = parse(init, seed)
            micros.append(pval)
            macros.append(stats)
            times.append(bandwidth)
        except:
            pass
    micro = np.mean(micros)
    micro_std = np.std(micros)
    macro = np.mean(macros)
    macro_std = np.std(macros)
    time_init = np.mean(times)
    bandwidth_std = np.std(times)
    # print("Data: {} - Init: {}".format(data, init))
    print("{:.3f}+-{:.3f}\t{:.3f}+-{:.3f}\t{:.3f}+-{:.3f}".format(
        micro, micro_std, macro, macro_std, time_init, bandwidth_std))
