import dgl
import json
import os
import multiprocessing

def add_to_dict(neibs, nodes, adj):
    neibs[node] = adj[i].nonzero()[1]

class DGLGraph(dgl.DGLGraph):
    def __init__(self, adj, readonly=False):
        super(DGLGraph, self).__init__(adj, readonly=readonly)
        self.adj = adj
        self.nodes_ = [int(x) for x in super(DGLGraph, self).nodes()] 

    def build_neibs_dict(self):
        neibs_file = "neibs-reddit.json"
        if not os.path.isfile(neibs_file):
            self.neibs = {}
            n_process = multiprocessing.cpu_count()//2
            pool = multiprocessing.Pool(processes=n_process)
            n_node = len(self.nodes()) // n_process + 1
            params = [(self.neibs, self.nodes()[i*n_node:(i+1)*n_node], self.adj) for i in range(n_process)]
            pool.map(add_to_dict, params)
            with open(neibs_file, "w+") as fp:
                fp.write(json.dumps(neibs_file))
        else:
            self.neibs = json.load(open("neibs-reddit.json"))

    def nodes(self):
        return self.nodes_
