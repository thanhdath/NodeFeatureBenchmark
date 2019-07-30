import time
import numpy as np

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
        super(IdentityFeature).__init__()
    def _generate(self, graph, dim_size):
        prep_dict = {}
        for idx, node in enumerate(graph.nodes()):
            feature = np.zeros((dim_size))
            feature[0] = G.degree(node)

            prep_dict[node] = feature
        return prep_dict

class RecursiveFeature(FeatureInitialization):
    def __init__(self):
        super(RecursiveFeature).__init__()
    def _generate(self, graph, dim_size):
        raise NotImplementedError

lookup = {
    "degree": NodeDegreesFeature,
    "uniform": RandomUniformFeature,
    "identity": IdentityFeature,
    "neighbor": NeighborhoodFeature,
    "recursive": RecursiveFeature
}