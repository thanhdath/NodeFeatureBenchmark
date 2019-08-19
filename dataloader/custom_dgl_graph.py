import dgl
import json
import os
import multiprocessing
import numpy as np
from dgl.base import DGLError, ALL
from dgl.view import NodeDataView, NodeSpace

def add_to_dict(com):
    neibs, nodes, adj = com
    for i, node in enumerate(nodes):
        neibs[node] = [int(x) for x in adj[i].nonzero()[1]]
    return neibs

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
        return list(range(0, len(self)))

class DGLGraph(dgl.DGLGraph):
    def __init__(self, adj, suffix="", readonly=False):
        super(DGLGraph, self).__init__(adj, readonly=readonly)
        self.adj = adj.tocsr().astype(np.float32)
        self.suffix = suffix
        # self.nodes_ = [int(x) for x in super(DGLGraph, self).nodes()] 
        # self.edges = self.edges()

    def build_neibs_dict(self):
        neibs_file = "neibs-reddit{}.json".format(self.suffix)
        if not os.path.isfile(neibs_file):
            self.neibs = {}
            n_process = multiprocessing.cpu_count()//2
            pool = multiprocessing.Pool(processes=n_process)
            n_node = len(self.nodes()) // n_process + 1
            params = [({}, self.nodes()[i*n_node:(i+1)*n_node], self.adj) for i in range(n_process)]
            res = pool.map(add_to_dict, params)
            [self.neibs.update(r) for r in res]
            with open(neibs_file, "w+") as fp:
                fp.write(json.dumps(self.neibs))
        else:
            self.neibs = json.load(open(neibs_file))
            self.neibs = {int(k): v for k, v in self.neibs.items()}
    
    @property
    def nodes(self):
        return NodeView(self)

    def degree(self, node):
        return self.adj[node].sum()

    def is_directed(self):
        return True

    def edges(self):
        src, trg = super(DGLGraph, self).edges()
        return [(int(s), int(t)) for s, t in zip(src, trg)]

    def is_multigraph(self):
        return self._graph.is_multigraph()

    def is_directed(self):
        return False
    
    def __iter__(self):
        yield from self.nodes()
