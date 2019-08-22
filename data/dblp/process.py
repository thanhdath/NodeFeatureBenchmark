labelfp = open('raw_labels.txt', 'w+')
with open('com-dblp.all.cmty.txt') as fp:
    for line in fp:
        elms = line.split()
        if len(elms) >= 3:
            labelfp.write(line)
labelfp.close()

node2labels = {}
with open('raw_labels.txt') as fp:
    for i, line in enumerate(fp):
        elms = line.strip().split()
        for elm in elms:
            node2labels[elm] = node2labels.get(elm, []) + [i]
print("Node has labels: ", len(node2labels))

id2idx = {k: i for i, k in enumerate(node2labels.keys())}

edgelistfp = open('edgelist.txt', 'w+')
with open('com-dblp.ungraph.txt') as fp:
    for line in fp:
        if "#" in line: continue
        src, trg = line.strip().split()
        if src in node2labels and trg in node2labels:
            edgelistfp.write("{} {}\n".format(id2idx[src], id2idx[trg]))
edgelistfp.close()
"""
remove community with has less than 3 nodes -> remains 6millions community
remove nodes which do not belongs to any community

94450158 edges
6288363 classes
2307085 nodes
"""
with open('labels.txt', 'w+') as fp:
    for node in node2labels:
        fp.write("{} {}\n".format(id2idx[node], ' '.join(map(str, node2labels[node]))))