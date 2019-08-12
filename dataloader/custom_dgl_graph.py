import dgl

class CustomDGLGraph(dgl.DGLGraph):
    def __init__(self, adj, readonly=False):
        super(CustomDGLGraph, self).__init__(adj, readonly=readonly)
        self.adj = adj
        self.nodes = [int(x) for x in super(CustomDGLGraph, self).nodes()] 

    def build_neibs_dict(self):
        self.neibs = {}
        for src, trg in zip(super(CustomDGLGraph, self).edges()):
            src = int(src)
            trg = int(trg)
            self.neibs[src] = self.neibs.get(src, []) + [trg]
            self.neibs[trg] = self.neibs.get(trg, []) + [src]

    @property
    def nodes(self):
        return self.nodes 
