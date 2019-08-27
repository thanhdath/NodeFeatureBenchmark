import time
import numpy as np
from embed_algs import *
import networkx as nx
import pdb
from scipy.sparse.linalg import svds
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, OneHotEncoder
from normalization import lookup as lookup_normalizer
import json
import os
from scipy.sparse import vstack, csr_matrix, hstack
try:
    from networkit import *
except:
    print("Warning: cannot import networkit. Install by command: pip install networkit")

def log_verbose(msg, v):
    if v >= 1:
        print(msg)

def check_is_multiclass(features_arr):
    for feature in features_arr:
        if len(feature) > 1: return True 
    return False

def transform_onehot_or_multiclass(features_arr, is_multiclass=False):
    if is_multiclass:
        encoder = MultiLabelBinarizer(sparse_output=False)
        encoder.fit_transform(features_arr)
    else:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        encoder.fit_transform(features_arr)
    features_arr = encoder.transform(features_arr)
    return features_arr

class FeatureInitialization():
    def __init__(self):
        pass 
    def generate(self, graph, dim_size, inplace=False, 
        normalizer="pass", verbose=1, shuffle=False):
        # wrapper function for generate()
        log_verbose('Start generate feature', verbose)
        stime = time.time()
        features = self._generate(graph, dim_size)
        features = lookup_normalizer[normalizer].norm(features, graph, verbose=verbose)
        if shuffle:
            print("Shuffle features")
            features_arr = np.array([features[x] for x in graph.nodes()])
            np.random.shuffle(features_arr)
            features = {node: features_arr[i] for i, node in enumerate(graph.nodes())}

        etime = time.time()
        log_verbose("Time init features: {:.3f}s".format(etime-stime), verbose)
        if inplace:
            for node in graph.nodes():
                graph.node[node]['feature'] = features[node]
        return features
    def _generate(self, graph, dim_size):
        return {}

class NodeDegreesFeature(FeatureInitialization):
    def __init__(self, **kwargs):
        super(NodeDegreesFeature).__init__()
    def _generate(self, graph, dim_size):
        prep_dict = {}
        for idx, node in enumerate(graph.nodes()):
            prep_dict[node] = np.array([graph.degree(node)]+[1.]*(dim_size-1))
            # prep_dict[node] = graph.degree(node)
        # features_arr = [[prep_dict[x]] for x in graph.node()]
        # features_arr = transform_onehot_or_multiclass(features_arr)
        # prep_dict = {node: features_arr[i] for i, node in enumerate(graph.nodes())}
        return prep_dict

class RandomUniformFeature(FeatureInitialization):
    def __init__(self, **kwargs):
        super(RandomUniformFeature).__init__()
    def _generate(self, graph, dim_size):
        prep_dict = {}
        for idx, node in enumerate(graph.nodes()):
            prep_dict[node] = np.random.uniform(-1.,1.,size=(dim_size))
        return prep_dict

class IdentityFeature(FeatureInitialization):
    def __init__(self, **kwargs):
        super(IdentityFeature).__init__()
    def _generate(self, graph, dim_size=None):
        features = np.identity(len(graph.nodes()))
        prep_dict = {}
        for idx, node in enumerate(graph.nodes()):
            prep_dict[node] = features[idx]
        return prep_dict

class NeighborhoodFeature(FeatureInitialization):
    def __init__(self, **kwargs):
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
    def __init__(self, **kwargs):
        super(RecursiveFeature).__init__()
    def _generate(self, graph, dim_size):
        raise NotImplementedError

class DeepWalkFeature(FeatureInitialization):
    def __init__(self, **kwargs):
        super(DeepWalkFeature).__init__()
    def _generate(self, graph, dim_size):
        features = deepwalk(graph, dim_size) 
        return features

class Node2VecFeature(FeatureInitialization):
    def __init__(self, **kwargs):
        super(Node2VecFeature).__init__()
    def _generate(self, graph, dim_size):
        features = node2vec(graph, dim_size)
        return features

class HOPEFeature(FeatureInitialization):
    def __init__(self, **kwargs):
        super(HOPEFeature).__init__()
    def _generate(self, graph, dim_size):
        features = HOPE(graph, dim_size)
        return features

class TriangleFeature(FeatureInitialization):
    def __init__(self, **kwargs):
        super(TriangleFeature).__init__()
        try:
            self.use_networkit = kwargs["use_networkit"]
        except:
            self.use_networkit = False
    def _generate(self, graph, dim_size):
        if self.use_networkit: # graph must be an instance of networkit
            execute = sparsification.ChibaNishizekiTriangleEdgeScore(graph)
            graph.indexEdges()
            execute.run()
            triangles = {}
            for i, (src, trg) in enumerate(graph.edges()):
                triangles[src] = triangles.get(src, 0) + execute.scores()[i]
                triangles[trg] = triangles.get(trg, 0) + execute.scores()[i]
        else:
            if nx.is_directed(graph) and "DGLGraph" not in graph.__class__.__name__:
                graph = nx.to_undirected(graph)
            triangles = nx.triangles(graph)
        
        prep_dict = {}
        for idx, node in enumerate(graph.nodes()):
            prep_dict[node] = np.array([triangles[node]]+[1.]*(dim_size-1))
        return prep_dict

class EgonetFeature(FeatureInitialization):
    def __init__(self, **kwargs):
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
    def __init__(self, **kwargs):
        """
        k-core number
        """
        super(KCoreNumberFeature).__init__()
        try:
            self.use_networkit = kwargs["use_networkit"]
        except:
            self.use_networkit = False
    def _generate(self, graph, dim_size):
        if self.use_networkit: # graph must be an instance of networkit
            graph.removeSelfLoops()
            kcore = centrality.CoreDecomposition(graph)
            kcore.run()
            kcore = kcore.getPartition().getVector()
            kcore = {node: kcore[i] for i, node in enumerate(graph.nodes())}
        else:
            graph.remove_edges_from(nx.selfloop_edges(graph))
            kcore = nx.core_number(graph)

        prep_dict = {}
        for idx, node in enumerate(graph.nodes()):
            feature = np.ones((dim_size))
            feature[0] = kcore[node]
            prep_dict[node] = feature
        return prep_dict

class PageRankFeature(FeatureInitialization):
    def __init__(self, **kwargs):
        super(PageRankFeature).__init__()
        try:
            self.use_networkit = kwargs["use_networkit"]
        except:
            self.use_networkit = False
    def _generate(self, graph, dim_size):
        if self.use_networkit: # graph must be an instance of networkit
            pr = centrality.PageRank(graph, 1e-6)
            pr.run()
            pr = pr.ranking()
            pr = {x[0]: x[1] for x in pr}
        else:
            pr = nx.pagerank(graph)
        prep_dict = {}
        for idx, node in enumerate(graph.nodes()):
            feature = np.ones((dim_size))
            feature[0] = pr[node]
            prep_dict[node] = feature
        return prep_dict

class LocalColoringColorFeature(FeatureInitialization):
    def __init__(self, **kwargs):
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
    def __init__(self, **kwargs):
        """ Returns the size of the largest maximal clique containing given node.
        """
        super(NodeCliqueNumber).__init__()
    def _generate(self, graph, dim_size):
        if nx.is_directed(graph):
            graph = nx.to_undirected(graph)
        prep_dict = {}
        cn = nx.node_clique_number(graph)
        for idx, node in enumerate(graph.nodes()):
            feature = np.ones((dim_size))
            feature[0] = cn[node]
            prep_dict[node] = feature
        return prep_dict

class OriginalFeature(FeatureInitialization):
    def __init__(self, **kwargs):
        """ Returns the size of the largest maximal clique containing given node.
        """
        super(OriginalFeature).__init__()
        self.feature_path = kwargs["feature_path"]
    def read_node_features(self):
        if self.feature_path.endswith(".npy"):
            features = np.load(self.feature_path, allow_pickle=True)[()]
        elif self.feature_path.endswith(".npz"):
            npz = np.load(self.feature_path, allow_pickle=True)
            features = npz['feats'][()]
            try:
                nodes = npz['nodes'][()]
                features_dict = {node: features[i] for i, node in enumerate(nodes)}
            except:
                pass
        else:
            features = {}
            fin = open(self.feature_path, 'r')
            for l in fin.readlines():
                vec = l.split()
                features[int(vec[0])] = np.array([float(x) for x in vec[1:]])
            fin.close()
        return features
    def _generate(self, graph, dim_size):
        features = self.read_node_features()
        return features

class NodeLabelFeature(FeatureInitialization):
    def __init__(self, **kwargs):
        super(NodeLabelFeature).__init__()
        self.label_path = kwargs["label_path"]
    def read_node_labels(self):
        if self.label_path.endswith(".npz"):
            npz = np.load(self.label_path, allow_pickle=True)
            features = npz['labels'][()]
            try:
                is_multiclass = npz['is_multiclass'][()]
            except:
                values = list(features.values())
                try:
                    len(values[0])
                    for v in values:
                        if len(v) > 1: 
                            is_multiclass = True 
                            break
                except:
                    is_multiclass = False
                    features = {k: [v] for k, v in features.items()}
                
        else:
            features = {}
            is_multiclass = False
            fin = open(self.label_path, 'r')
            for l in fin.readlines():
                vec = l.split()
                if len(vec) > 2:
                    is_multiclass = True
                features[int(vec[0])] = np.array([float(x) for x in vec[1:]])
            fin.close()
        return features, is_multiclass
    def _generate(self, graph, dim_size):
        features, is_multiclass = self.read_node_labels()
        features_arr = [features[x] for x in graph.nodes()]
        features_arr = transform_onehot_or_multiclass(features_arr, is_multiclass=is_multiclass)
        features = {node: features_arr[i] for i, node in enumerate(graph.nodes())}
        return features

class SymmetricSVD(FeatureInitialization):
    def __init__(self, **kwargs):
        super(SymmetricSVD).__init__()
        self.alpha = kwargs['alpha']
    def _generate(self, graph, dim_size):
        if "DGLGraph" in graph.__class__.__name__:
            adj = graph.adj 
        else:
            adj = nx.to_scipy_sparse_matrix(graph, dtype=np.float32)
        if adj.shape[0] <= dim_size:
            padding_row = csr_matrix((dim_size-adj.shape[0]+1, adj.shape[1]), dtype=adj.dtype)
            adj = vstack((adj, padding_row))
            padding_col = csr_matrix((adj.shape[0], dim_size-adj.shape[1]+1), dtype=adj.dtype)
            adj = hstack((adj, padding_col))

        U, X,_ = svds(adj, k = dim_size)
        embedding = U*(X**(self.alpha))
        features = {node: embedding[i] for i, node in enumerate(graph.nodes())}
        return features

class LINEFeature(FeatureInitialization):
    def __init__(self, **kwargs):
        super(LINEFeature).__init__()
    def _generate(self, graph, dim_size):
        features = LINE(graph, dim_size)
        return features

class GraphFactorizationFeature(FeatureInitialization):
    def __init__(self, **kwargs):
        super(GraphFactorizationFeature).__init__()
    def _generate(self, graph, dim_size):
        features = graph_factorization(graph, dim_size)
        return features

class GraphWaveFeature(FeatureInitialization):
    def __init__(self, **kwargs):
        super(GraphWaveFeature).__init__()
    def _generate(self, graph, dim_size):
        features = graphwave(graph, dim_size)
        return features

class Struc2VecFeature(FeatureInitialization):
    def __init__(self, **kwargs):
        super(Struc2VecFeature).__init__()
    def _generate(self, graph, dim_size):
        features = struc2vec(graph, dim_size)
        return features

class DeepwalkNetMFFeature(FeatureInitialization):
    def __init__(self, **kwargs):
        super(DeepwalkNetMFFeature).__init__()
    def _generate(self, graph, dim_size):
        from helpers.netmf import netmf_small, netmf_large
        features = netmf_small(graph, dim_size)
        return features

class ProposedFeature(FeatureInitialization):
    def __init__(self, **kwargs):
        super(ProposedFeature).__init__()
    def _generate(self, graph, dim_size):
        features1 = SymmetricSVD(alpha=1)._generate(graph, dim_size-3)
        features2 = NodeDegreesFeature().generate(graph, dim_size=1, inplace=False, normalizer="standard")
        features = {node: np.hstack([features1[node], features2[node]]) 
            for node in features1}
        return features


lookup = {
    "ori": OriginalFeature,
    "label": NodeLabelFeature,
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
    "clique": NodeCliqueNumber,
    "ssvd": SymmetricSVD,
    "ssvd0.5": SymmetricSVD,
    "ssvd1": SymmetricSVD,
    "line": LINEFeature,
    "gf": GraphFactorizationFeature,
    "graphwave": GraphWaveFeature,
    "struc2vec": Struc2VecFeature,
    "netmf": DeepwalkNetMFFeature,
    "propose": ProposedFeature
}

