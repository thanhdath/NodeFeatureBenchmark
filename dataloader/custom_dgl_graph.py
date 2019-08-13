import dgl
import json
import os
import multiprocessing
import numpy as np
from dgl.base import DGLError, ALL
from dgl.view import NodeDataView, NodeSpace

def add_to_dict(neibs, nodes, adj):
    for i, node in enumerate(nodes):
        neibs[node] = adj[i].nonzero()[1]

class NodeView(object):
    """A NodeView class to act as G.nodes for a DGLGraph.

    Can be used to get a list of current nodes and get and set node data.

    See Also
    --------
    dgl.DGLGraph.nodes
    """
    __slots__ = ['_graph']

    def __init__(self, graph):
        self._graph = graph

    def __len__(self):
        return self._graph.number_of_nodes()

    def __getitem__(self, nodes):
        if isinstance(nodes, slice):
            # slice
            if not (nodes.start is None and nodes.stop is None
                    and nodes.step is None):
                raise DGLError('Currently only full slice ":" is supported')
            return NodeSpace(data=NodeDataView(self._graph, ALL))
        else:
            return NodeSpace(data=NodeDataView(self._graph, nodes))

    def __call__(self):
        """Return the nodes."""
        return np.arange(0, len(self))

class DGLGraph(dgl.DGLGraph):
    def __init__(self, adj, readonly=False):
        super(DGLGraph, self).__init__(adj, readonly=readonly)
        self.adj = adj
        # self.nodes_ = [int(x) for x in super(DGLGraph, self).nodes()] 

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
    
    @property
    def nodes(self):
        return NodeView(self)
