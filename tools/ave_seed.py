import re 
import sys
import numpy as np

def parse(init, seed):
    logfile = "log/{}/{}-{}-seed{}".format(alg, data, init, seed)
    content = open(logfile).read()
    micro = re.findall("Micro: [0-9\.]+", content)[0]
    micro = micro.replace("Micro: ", "")
    micro = float(micro)

    macro = re.findall("Macro: [0-9\.]+", content)[0]
    macro = macro.replace("Macro: ", "")
    macro = float(macro)

    time_init = re.findall("Time init features: [0-9\.]+", content)[0]
    time_init = time_init.replace("Time init features: ", "")
    time_init = float(time_init)
    return micro, macro, time_init

data = sys.argv[1]
try:
    alg = sys.argv[2]
except:
    alg = "sgc"

for init in ["degree", "uniform", "identity"]:
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
    print("\tMicro: {:.3f}\tMacro: {:.3f}\tTime: {:.3f}".format(micro, macro, time_init))