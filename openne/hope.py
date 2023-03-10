import pdb
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from . import graph as g
import torch
from sklearn.preprocessing import normalize
from scipy.sparse import vstack, csr_matrix, hstack

def matrix_multiplication_chunk(m, chunk):
    res = sp.csr_matrix((m.shape))
    n_iters = int(np.ceil(m.shape[0]/chunk))
    for i in range(n_iters):
        for j in range(n_iters):
            res[i*chunk:(i+1)*chunk, j*chunk:(j+1)*chunk] = m[i*chunk:(i+1)*chunk,:].dot(m[:,j*chunk:(j+1)*chunk])
    return res

class HOPE(object):
    def __init__(self, graph, d):
        '''
          d: representation vector dimension
        '''
        self._d = d
        self._graph = graph.G
        self.g = graph
        self._node_num = graph.node_size
        self.learn_embedding()

    # def learn_embedding(self):

    #     graph = self.g.G
    #     A = nx.to_numpy_matrix(graph)

    #     # self._beta = 0.0728

    #     # M_g = np.eye(graph.number_of_nodes()) - self._beta * A
    #     # M_l = self._beta * A

    #     M_g = np.eye(graph.number_of_nodes())
    #     M_l = np.dot(A, A)

    #     S = np.dot(np.linalg.inv(M_g), M_l)
    #     # s: \sigma_k
    #     u, s, vt = lg.svds(S, k=self._d // 2)
    #     sigma = np.diagflat(np.sqrt(s))
    #     X1 = np.dot(u, sigma)
    #     X2 = np.dot(vt.T, sigma)
    #     # self._X = X2
    #     self._X = np.concatenate((X1, X2), axis=1)

    def learn_embedding(self):
        graph = self.g.G
        if "DGLGraph" in graph.__class__.__name__:
            A = graph.adj 
        else:
            A = nx.to_scipy_sparse_matrix(graph, dtype=np.float32)
        if A.shape[0] <= self._d:
            padding_row = csr_matrix((self._d-A.shape[0]+1, A.shape[1]), dtype=A.dtype)
            A = vstack((A, padding_row))
            padding_col = csr_matrix((A.shape[0], self._d-A.shape[1]+1), dtype=A.dtype)
            A = hstack((A, padding_col))
        #M_l = beta*A
        #M_g = sp.eye(graph.number_of_nodes(), dtype=np.float32) - M_l

        #M_g = sp.eye(graph.number_of_nodes(), dtype=np.float32)
        #M_l = A.dot(A)

        #S = lg.inv(M_g).dot(M_l)
        # s: \sigma_k
        u, s, vt = lg.svds(A, k=self._d // 2)
        sigma = sp.diags(s).todense()

        X1 = np.dot(u, sigma)
        X2 = np.dot(vt.T, sigma)
        # self._X = X2
        self._X = np.asarray(np.concatenate((X1, X2), axis=1))
        self._X = self._X[:graph.number_of_nodes()]

    @property
    def vectors(self):
        vectors = {}
        look_back = self.g.look_back_list
        for i, embedding in enumerate(self._X):
            vectors[look_back[i]] = embedding
        return vectors

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self._d))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
