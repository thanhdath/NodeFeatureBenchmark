import re 
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def parse(init, seed, feature_size):
    logfile = "log/{}/{}-{}-dim{}-seed{}".format(alg, data, init, feature_size, seed)
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

inits = "deepwalk ssvd0.5 ssvd1 hope graphwave struc2vec".split()
features_size = [32, 64, 128, 256, 512]
print("Check orderd dimension", features_size)

print("Check ordered init methods:")
for i, init in enumerate(inits):
    print(i+1, init)

for init in inits:
    # print("Init ====== {} ======".format(init))
    for feature_size in features_size:
        micros = []
        macros = []
        times = []
        for seed in range(40, 50):
            try:
                micro, macro, time_init = parse(init, seed, feature_size)
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
    # print()
