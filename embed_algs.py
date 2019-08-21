import multiprocessing
from gensim.models import Word2Vec
from openne.walker import BasicWalker
from openne.node2vec import Node2vec
from openne.hope import HOPE as HOPE_Openne
from openne.graph import Graph
from types import SimpleNamespace

def deepwalk(G, dim_size, number_walks=20, walk_length=10, 
    workers=multiprocessing.cpu_count()//3):
    walk = BasicWalker(G, workers=workers)
    sentences = walk.simulate_walks(num_walks=number_walks, walk_length=walk_length, num_workers=workers)
    # for idx in range(len(sentences)):
    #     sentences[idx] = [str(x) for x in sentences[idx]]

    print("Learning representation...")
    word2vec = Word2Vec(sentences=sentences, min_count=0, workers=workers,
                            size=dim_size, sg=1)
    vectors = {}
    for word in G.nodes():
        vectors[word] = word2vec.wv[str(word)]
    return vectors

def node2vec(G, dim_size, number_walks=20, walk_length=10, 
    workers=multiprocessing.cpu_count(), p=4.0, q=1.0):
    graph = Graph()
    graph.read_g(G)
    n2v = Node2vec(graph, walk_length, number_walks, dim_size, p, q, workers=workers)
    return n2v.vectors

def HOPE(G, dim_size):
    graph = Graph()
    graph.read_g(G)
    hh = HOPE_Openne(graph, dim_size)
    return hh.vectors

def LINE(G, dim_size):
    from openne.line import LINE as LINE_Openne
    graph = Graph()
    graph.read_g(G)
    line = LINE_Openne(graph, rep_size=dim_size, order=3)
    return line.vectors

def graph_factorization(G, dim_size):
    from openne.gf import GraphFactorization
    graph = Graph()
    graph.read_g(G)
    gf = GraphFactorization(graph, rep_size=dim_size)
    return gf.vectors

def graphwave(G, dim_size):
    from helpers.GraphWave.spectral_machinery import WaveletMachine
    settings = SimpleNamespace(
        mechanism='exact',
        heat_coefficient=1000.0,
        sample_number=50,
        approximation=100,
        step_size=20,
        switch=dim_size,
        node_label_type=int
    )
    machine = WaveletMachine(G, settings)
    machine.create_embedding()
    embeds = machine.real_and_imaginary
    indexes = list(machine.index)
    vectors = {indexes[i]: embeds[i] for i in range(G.number_of_nodes())}
    return vectors