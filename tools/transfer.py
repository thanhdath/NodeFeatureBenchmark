import re
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def parse(init, seed):
    logfile = "log/transfer/{}/{}-{}-seed{}-loadfrom-{}".format(
        alg, data, init, seed, load_from)
    if with_muse:
        logfile += "-withmuse"
    content = open(logfile).read()
    f1 = re.findall("micro-macro: [0-9\.\s]+", content)[0]
    f1 = f1.replace("micro-macro: ", "")
    micro, macro = [float(x) for x in f1.split()]
    return micro, macro


data = sys.argv[1]
alg = sys.argv[2]
load_from = sys.argv[3]
try:
    sys.argv[4]
    with_muse = True 
except:
    with_muse = False
print("Parse result transfer with muse")

inits = "ssvd1 ssvd0.5 hope deepwalk graphwave degree-standard ori".split()

print("Check ordered init methods:")
for i, init in enumerate(inits):
    print(i+1, init)

for init in inits:
    micros = []
    macros = []
    times = []
    for seed in range(40, 50):
        try:
            micro, macro = parse(init, seed)
            micros.append(micro)
            macros.append(macro)
        except:
            pass
    micro = np.mean(micros)
    micro_std = np.std(micros)
    macro = np.mean(macros)
    macro_std = np.std(macros)
    # print("Data: {} - Init: {}".format(data, init))
    print("{:.3f}+-{:.3f}\t{:.3f}+-{:.3f}".format(micro, micro_std, macro, macro_std))
