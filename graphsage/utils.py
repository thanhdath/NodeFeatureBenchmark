import networkx as nx
import torch.nn as nn

class IdentityLayer(nn.Module):
    def __init__(self):
        """ Example of mapping -- doesn't do anything """
        super(IdentityLayer, self).__init__()
    
    def forward(self, inputs):
        return inputs 

def print_graph_detail(G):
    print(nx.info(G))
    if(nx.is_connected(G)):        
        print("Diameter: " + str(nx.diameter(G)))
    else:
        print("Diameter: N/A")
    print("Avg. clustering coefficient: " + str(nx.average_clustering(G)))
    print("# Triangles: " + str(sum(nx.triangles(G).values()) / 3))

    # from utils import print_graph_detail
    # print_graph_detail(G)