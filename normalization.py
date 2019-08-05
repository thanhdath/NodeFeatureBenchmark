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
    def norm(features, graph):
        print("Feature Normalizer: ")
        scaler = StandardScaler()
        features_arr = np.array([features[x] for x in graph.nodes()])
        scaler.fit(features_arr)
        features_arr = scaler.transform(features_arr)
        features = {node: features_arr[i] for i, node in enumerate(graph.nodes())}
        return features


class RowSumNormalizer(Normalizer):
    @staticmethod
    def norm(mx, **kwargs):
        print("Feature Normalizer: RowSum = 1")
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class PassNormalizer(Normalizer):
    @staticmethod
    def norm(features, **kwargs):
        print("Feature Normalizer: No")
        return features


lookup = {
    "standard": StandardNormalizer,
    "rowsum": RowSumNormalizer,
    "pass": PassNormalizer
}
