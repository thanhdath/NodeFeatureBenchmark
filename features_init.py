import time
import numpy as np
from embed_algs import deepwalk, node2vec, HOPE
import networkx as nx
import pdb

class FeatureInitialization():
    def __init__(self):
        pass 
    def generate(self, graph, dim_size, inplace=False):
        # wrapper function for generate()
        print('Start generate feature')
        stime = time.time()
        features = self._generate(graph, dim_size)
        etime = time.time()
        print("Time init features: {:.3f}s".format(etime-stime))
        if inplace:
            for node in graph.nodes():
                graph.node[node]['feature'] = features[node]
        print("== Done generate features")
        return features
    def _generate(self, graph, dim_size):
        return {}

class NodeDegreesFeature(FeatureInitialization):
    def __init__(self):
        super(NodeDegreesFeature).__init__()
    def _generate(self, graph, dim_size):
        prep_dict = {}
        for idx, node in enumerate(graph.nodes()):
            prep_dict[node] = np.array([graph.degree(node)]+[1.]*(dim_size-1))
        return prep_dict

class RandomUniformFeature(FeatureInitialization):
    def __init__(self):
        super(RandomUniformFeature).__init__()
    def _generate(self, graph, dim_size):
        prep_dict = {}
        for idx, node in enumerate(graph.nodes()):
            prep_dict[node] = np.random.uniform(0.,1.,size=(dim_size))
        return prep_dict

class IdentityFeature(FeatureInitialization):
    def __init__(self):
        super(IdentityFeature).__init__()
    def _generate(self, graph, dim_size=None):
        features = np.identity(len(graph.nodes()))
        prep_dict = {}
        for idx, node in enumerate(graph.nodes()):
            prep_dict[node] = features[idx]
        return prep_dict

class NeighborhoodFeature(FeatureInitialization):
    def __init__(self):
        super(NeighborhoodFeature).__init__()
    def _generate(self, graph, dim_size):
        prep_dict = {}
        n_edges = len(graph.edges())
        for idx, node in enumerate(graph.nodes()):
            ego_graph = nx.ego_graph(graph, node, radius=1)
            n_within_edges = len(ego_graph.edges())
            n_external_edges = n_edges - n_within_edges
            feature = np.ones((dim_size))
            feature[0] = graph.degree(node)
            feature[1] = n_within_edges
            feature[2] = n_external_edges
            prep_dict[node] = feature
        return prep_dict

class RecursiveFeature(FeatureInitialization):
    def __init__(self):
        super(RecursiveFeature).__init__()
    def _generate(self, graph, dim_size):
        raise NotImplementedError

class DeepWalkFeature(FeatureInitialization):
    def __init__(self):
        super(DeepWalkFeature).__init__()
    def _generate(self, graph, dim_size):
        features = deepwalk(graph, dim_size) 
        return features

class Node2VecFeature(FeatureInitialization):
    def __init__(self):
        super(Node2VecFeature).__init__()
    def _generate(self, graph, dim_size):
        features = node2vec(graph, dim_size)
        return features

class HOPEFeature(FeatureInitialization):
    def __init__(self):
        super(HOPEFeature).__init__()
    def _generate(self, graph, dim_size):
        features = HOPE(graph, dim_size)
        return features

class TriangleFeature(FeatureInitialization):
    def __init__(self):
        super(TriangleFeature).__init__()
    def _generate(self, graph, dim_size):
        triangles = nx.triangles(graph)
        prep_dict = {}
        for idx, node in enumerate(graph.nodes()):
            prep_dict[node] = np.array([triangles[node]]+[1.]*(dim_size-1))
        return prep_dict

class EgonetFeature(FeatureInitialization):
    def __init__(self):
        """
        number of within-egonet edges
        number of external-egonet edges
        """
        super(EgonetFeature).__init__()
    def _generate(self, graph, dim_size):
        prep_dict = {}
        n_edges = len(graph.edges())
        for idx, node in enumerate(graph.nodes()):
            ego_graph = nx.ego_graph(graph, node, radius=1)
            n_within_edges = len(ego_graph.edges())
            n_external_edges = n_edges - n_within_edges
            feature = np.ones((dim_size))
            feature[0] = n_within_edges
            feature[1] = n_external_edges
            prep_dict[node] = feature
        return prep_dict

class KCoreNumberFeature(FeatureInitialization):
    def __init__(self):
        """
        k-core number
        """
        super(KCoreNumberFeature).__init__()
    def _generate(self, graph, dim_size):
        prep_dict = {}
        kcore = nx.core_number(graph)
        for idx, node in enumerate(graph.nodes()):
            feature = np.ones((dim_size))
            feature[0] = kcore[node]
            prep_dict[node] = feature
        return prep_dict

class PageRankFeature(FeatureInitialization):
    def __init__(self):
        super(PageRankFeature).__init__()
    def _generate(self, graph, dim_size):
        prep_dict = {}
        pr = nx.pagerank(graph)
        for idx, node in enumerate(graph.nodes()):
            feature = np.ones((dim_size))
            feature[0] = pr[node]
            prep_dict[node] = feature
        return prep_dict

class LocalColoringColorFeature(FeatureInitialization):
    def __init__(self):
        super(LocalColoringColorFeature).__init__()
    def _generate(self, graph, dim_size):
        prep_dict = {}
        ncolor = nx.algorithms.coloring.greedy_color(graph)
        for idx, node in enumerate(graph.nodes()):
            feature = np.ones((dim_size))
            feature[0] = ncolor[node]
            prep_dict[node] = feature
        return prep_dict

class NodeCliqueNumber(FeatureInitialization):
    def __init__(self):
        """ Returns the size of the largest maximal clique containing given node.
        """
        super(NodeCliqueNumber).__init__()
    def _generate(self, graph, dim_size):
        prep_dict = {}
        cn = nx.node_clique_number(graph)
        for idx, node in enumerate(graph.nodes()):
            feature = np.ones((dim_size))
            feature[0] = cn[node]
            prep_dict[node] = feature
        return prep_dict


lookup = {
    "degree": NodeDegreesFeature,
    "uniform": RandomUniformFeature,
    "identity": IdentityFeature,
    "degree+egonet": NeighborhoodFeature,
    # "recursive": RecursiveFeature,
    "deepwalk": DeepWalkFeature,
    "node2vec": Node2VecFeature,
    "hope": HOPEFeature,
    "triangle": TriangleFeature,
    "egonet": EgonetFeature,
    "kcore": KCoreNumberFeature,
    "pagerank": PageRankFeature,
    "coloring": LocalColoringColorFeature,
    "clique": NodeCliqueNumber
}

