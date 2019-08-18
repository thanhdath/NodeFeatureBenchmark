import re 
import sys 

logfile = sys.argv[1]

inits = "degree-standard uniform deepwalk ssvd0.5 ssvd1 hope line gf triangle-standard kcore-standard egonet-standard pagerank-standard coloring-standard clique-standard identity ori ori-rowsum ori-standard".split()

content = open(logfile).read()

for i, init in enumerate(inits):
    print(i, init)

for init in inits:
    times = re.findall("Time init features {} : [0-9\.]+".format(init), content)
    times = [float(x.split(":")[-1].strip()) for x in times]
    try:
        time_init = sum(times) / len(times)
    except:
        time_init = -1
    print("{:.3f}".format(time_init))
    