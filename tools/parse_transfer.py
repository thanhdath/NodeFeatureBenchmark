
import os 
import glob 
import re

files = os.listdir("logs/transfer-gin/1/")

file2res = {file: {"no-finetuning": [], "best-val-acc": []} for file in files}

for seed in range(1, 21):
    for file in files:
        filepath = f"logs/transfer-gin/{seed}/{file}"
        content = open(filepath).read()
        nofinetuning = re.findall("== No finetuning - val acc:.+", content)[0]
        nofinetuning = float(nofinetuning.replace("== No finetuning - val acc: ", ""))
        file2res[file]["no-finetuning"].append(nofinetuning)

        bestval = re.findall("Best val acc:.+", content)[-1]
        bestval = float(bestval.replace("Best val acc: ", ""))
        file2res[file]["best-val-acc"].append(bestval)

import numpy as np
for file in files:
    print("========")
    print(file )
    print("no fine tuning: {:.3f} +- {:.3f}".format(np.mean(file2res[file]["no-finetuning"]),
        np.std(file2res[file]["no-finetuning"])))
    print("best-val: {:.3f} +- {:.3f}".format(np.mean(file2res[file]["best-val-acc"]),
        np.std(file2res[file]["best-val-acc"])))