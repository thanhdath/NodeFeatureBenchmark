mmd="""
0.000+-0.000	9.539+-0.001	0.997+-0.000	0.002+-0.000
0.000+-0.000	8.089+-0.004	0.170+-0.000	-0.003+-0.000
0.000+-0.000	171.953+-9.399	3.321+-0.035	0.003+-0.000
0
0.000+-0.000	11.851+-0.012	1.081+-0.002	-0.004+-0.000
0.551+-0.023	-0.028+-0.001	8.226+-0.014	0.001+-0.000
0.001+-0.000	2.448+-0.000	0.016+-0.000	-0.007+-0.000
			error kernel width = 0.00000
1.000+-0.000	-0.706+-0.000	0.373+-0.000	2.110+-0.000
0.002+-0.001	2.448+-0.000	0.023+-0.000	0.002+-0.000
0.012+-0.003	1.516+-0.018	0.011+-0.000	0.007+-0.000
			error kernel width = 0.00000
init too slow	nan+-nan	nan+-nan	nan+-nan
"""
inits_mmd = """
SVD (alpha = 1)
SVD (alpha = 0.5)
DeepWalk
GraphWave
HOPE
Real feature
degree-standard 
triangle-standard
kcore-standard
egonet-standard
pagerank-standard
coloring-standard
clique-standard""".strip().split('\n')

import numpy as np
mmd = mmd.strip().split('\n')
stats = np.array([float(x.split('\t')[1].split('+')[0]) for x in mmd])
# f1s = f1.strip().split('\n')
# f1micro = []
# for x in f1s:
#     x = x.split('\t')[0].split('+')[0]
#     f1micro.append(x)
ranks = np.argsort(-stats)
for i in range(len(stats)):
    print("{}\t{}\t{}".format(inits_mmd[i], stats[i], ranks.tolist().index(i)+1))
    print()

