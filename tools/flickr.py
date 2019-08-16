import numpy as np 

labels = {}
with open('data/flickr/labels.txt') as fp:
    for line in fp:
        elms = line.strip().split()
        labels[int(elms[0])] = elms[1:]

nodes = sorted(set([x for x in labels.keys()]))
id2idx = {node: i for i, node in enumerate(nodes)}
# edgelist = []
# with open('data/flickr/edgelist.txt') as fp:
#     for line in fp:
#         src,trg = line.strip().split()
#         src = int(src)
#         trg = int(trg)
#         if src in id2idx and trg in id2idx:
#             edgelist.append([src,trg])


# new_edgelist = []
# with open('data/flickr/edgelist2.txt', 'w+') as fp:
#     for src, trg in edgelist:
#         # new_edgelist.append(id2idx[src], id2idx[trg])
#         fp.write("{} {}\n".format(id2idx[src], id2idx[trg]))
# with open('data/flickr/labels2.txt', 'w+') as fp:
#     for node in labels.keys():
#         fp.write("{} {}\n".format(id2idx[node], " ".join(labels[node])))
