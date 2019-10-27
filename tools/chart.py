import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


for data in "cora citeseer pubmed".split():
    for init in "deepwalk ssvd0.5 ssvd1 hope graphwave ori".split():
        transfer_from = [x for x in "cora citeseer pubmed".split() if x != data]

        data1file = "tools/acc/run-graphsage-{}-{}-40-tag-val_acc.csv".format(data, init)
        data2file = "tools/acc/run-graphsage-{}-{}-40-from-{}-{}-40-tag-val_acc.csv".format(data, init, transfer_from[0], init)
        data3file = "tools/acc/run-graphsage-{}-{}-40-from-{}-{}-40-tag-val_acc.csv".format(data, init, transfer_from[1], init)
        data1 = pd.read_csv(data1file).values
        data2 = pd.read_csv(data2file).values
        data3 = pd.read_csv(data3file).values
        steps = data1[:,1]
        acc1 = data1[:,2]
        acc2 = data2[:,2]
        acc3 = data3[:,2]
        acc1 = smooth(acc1, 0.6)
        acc2 = smooth(acc2, 0.6)
        acc3 = smooth(acc3, 0.6)

        # parse transfer from data
        # transfer_from = re.findall("from-[a-z]+-", data2file)[0].replace("from-", "")
        title = re.findall("run-graphsage-[a-z]+-[a-z]+", data2file)[0].replace("run-graphsage-", "")

        fig = plt.figure(figsize=(11,8))
        ax1 = fig.add_subplot(111)

        ax1.plot(steps, acc1, label='Without transfer')
        ax1.plot(steps, acc2, label='Transfer from {}'.format(transfer_from[0]))
        ax1.plot(steps, acc3, label='Transfer from {}'.format(transfer_from[1]))

        plt.xticks(list(range(0,200,20)))
        plt.xlabel('Epochs')
        plt.ylabel('Validation accuracy (%)')
        plt.title(title)

        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.85,0.15))
        ax1.grid('on')

        plt.savefig('{}-{}.png'.format(data, init))