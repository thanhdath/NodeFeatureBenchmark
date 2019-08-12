from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.sparse as sp


class Normalizer():
    def __init__(self, name):
        pass

    def norm(self, features):
        pass


class StandardNormalizer(Normalizer):
    @staticmethod
    def norm(features, graph, verbose=1):
        if verbose > 0:
            print("Feature Normalizer: Standard")
        scaler = StandardScaler()
        features_arr = np.array([features[int(x)] for x in graph.nodes()])
        scaler.fit(features_arr)
        features_arr = scaler.transform(features_arr)
        features = {int(node): features_arr[i] for i, node in enumerate(graph.nodes())}
        return features


class RowSumNormalizer(Normalizer):
    @staticmethod
    def norm(features, graph, verbose=1):
        mx = np.array([features[int(x)] for x in graph.nodes()])
        if verbose > 0:
            print("Feature Normalizer: RowSum = 1")
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        features = {int(node): mx[i] for i, node in enumerate(graph.nodes())}
        return features


class PassNormalizer(Normalizer):
    @staticmethod
    def norm(features, graph, verbose=1):
        if verbose > 0:
            print("Feature Normalizer: No")
        return features


lookup = {
    "standard": StandardNormalizer,
    "rowsum": RowSumNormalizer,
    "pass": PassNormalizer
}
