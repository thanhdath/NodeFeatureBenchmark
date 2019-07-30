from __future__ import print_function
from pathlib import Path

import numpy as np
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

class Dataset:

    N_WALKS = 50
    WALK_LEN = 5

    def __init__(self):
        self.prefix = None
        self.G = None
        self.num_total_node = None
        self.feats = None
        self.id_map = None #Mapping id of nodes to indices
        self.idx2id = None #Mapping idx of nodes to ids
        self.walks = None
        self.class_map = None
        self.labels = None
        self.num_class = None
        self.conversion = None
        self.nodes_ids = []
        self.nodes = []
        self.train_nodes = []
        self.val_nodes = []
        self.test_nodes = []
        self.train_nodes_ids = []
        self.val_nodes_ids = []
        self.test_nodes_ids = []
        self.train_edges = []
        self.val_edges = []
        self.test_edges = []
        self.adj = None
        self.train_adj = None
        self.deg = None
        self.train_deg = None
        self.max_degree = None
        self.multiclass = None
        self.train_all_edge = None

    def load_data(self, prefix, normalize=True, supervised=True,
                    max_degree = 25, multiclass=False, load_adj_dir = None, use_random_walks = True, load_walks=True, num_walk=50, walk_len=5, train_all_edge=False):

        self.multiclass = multiclass
        self.prefix = prefix
        if(max_degree != None):
            self.max_degree = max_degree
            print("Set max degree to {0}".format(max_degree))

        self.read_data(prefix)
        self.preprocess_data(prefix, normalize, load_adj_dir, use_random_walks, load_walks, num_walk, walk_len, supervised, train_all_edge)

        return self.G, self.feats, self.id_map, self.walks, self.class_map

    def read_data(self, prefix):

        print("-----------------------------------------------")
        print("Loading data:")

    ## Load graph data
        print("Loading graph data from {0}".format(prefix + "-G.json"))
        G_data = json.load(open(prefix + "-G.json"))
        G = json_graph.node_link_graph(G_data)

        if isinstance(G.nodes()[0], int):
            def conversion(n): return int(n)
        else:
            def conversion(n): return n

        ## Remove all nodes that do not have val/test annotations
        ## (necessary because of networkx weirdness with the Reddit data)
        broken_count = 0
        self.num_total_node = 0

        for node in G.nodes():
            if not 'val' in G.node[node] or not 'test' in G.node[node]:
                G.remove_node(node)
                broken_count += 1

        print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))
        self.G = G
        self.conversion = conversion
        self.num_total_node = len(G.nodes())
        print("File loaded successfully")

        ## Load feature data
        print("Loading feature from {0}".format(prefix + "-G.json"))
        if os.path.exists(prefix + "-feats.npy"):
            self.feats = np.load(prefix + "-feats.npy")
            print("File loaded successfully")
        else:
            print("No features present.. Only identity features will be used.")

        id_map = json.load(open(prefix + "-id_map.json"))
        id_map = {conversion(k): int(v) for k, v in id_map.items()}
        idx2id = {v: k for k, v in id_map.items()}
        self.id_map = id_map
        self.idx2id = idx2id

        ## Load classmap
        class_map_file = Path(prefix + "-class_map.json")
        if class_map_file.is_file():
            print("Loading classmap data from {0}".format(prefix + "-class_map.json"))
            class_map = json.load(open(prefix + "-class_map.json"))
            if isinstance(list(class_map.values())[0], list):
                def lab_conversion(n): return n
            else:
                def lab_conversion(n): return int(n)

            class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}
            self.class_map = class_map

            if self.multiclass:
                self.labels = [None]*len(self.class_map.keys())
                for idx in self.class_map.keys():
                    self.labels[self.id_map[idx]] = self.class_map[idx]
                self.labels = np.array(self.labels, dtype=np.int64)
            else:
                self.labels = np.zeros((self.feats.shape[0], 1), dtype=np.int64)
                for idx in self.class_map.keys():
                    self.labels[self.id_map[idx]] = np.argmax(self.class_map[idx])

            self.num_class = len(self.class_map[G.nodes()[0]])
            print("File loaded successfully")
        else:
            print("Class map doesn't exist")

    def preprocess_data(self, prefix, normalize=True, load_adj_dir = None, use_random_walks = True, load_walks=False, num_walk = 50, walk_len = 5, supervised=True, train_all_edge=False):

        G = self.G
        if G == None:
            raise Exception("Data hasn't been load")

        print("Loaded data.. now preprocessing..")

        # Categorize train, val and test nodes
        # Using id_maps.keys to control the node index
        self.nodes_ids = np.array([n for n in G.node.keys()])

        if not train_all_edge:
            self.train_nodes_ids = np.array([n for n in self.nodes_ids if not G.node[n]['val'] and not G.node[n]['test']])
            self.val_nodes_ids = np.array([n for n in self.nodes_ids if G.node[n]['val']])
            self.test_nodes_ids = np.array([n for n in self.nodes_ids if G.node[n]['test']])
        else:
            self.train_nodes_ids = np.array([n for n in self.nodes_ids])
            self.val_nodes_ids = np.array([n for n in self.nodes_ids])
            self.test_nodes_ids = np.array([n for n in self.nodes_ids])

        self.nodes = np.array([self.id_map[n] for n in self.nodes_ids])
        self.train_nodes = np.array([self.id_map[n] for n in self.train_nodes_ids])
        self.val_nodes = np.array([self.id_map[n] for n in self.val_nodes_ids])
        self.test_nodes = np.array([self.id_map[n] for n in self.test_nodes_ids])

        ## Make sure the graph has edge train_removed annotations
        ## (some datasets might already have this..)
        for edge in G.edges():
            if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
                G[edge[0]][edge[1]]['train_removed'] = True
            else:
                G[edge[0]][edge[1]]['train_removed'] = False

        #Remove isolated train nodes after remove "train_remove" edge from train graph
        # and val nodes and test nodes from original graph
        if not train_all_edge:
            self.remove_isolated_node()

        #Construct train_deg and deg, deg[i] is degree of node that have idx i, train_deg consider "train_remove" edge
        if not train_all_edge:
            self.construct_train_val_deg()
        else:
            self.construct_all_deg()

        #Construct train_adj and adj, adj is matrix of Uniformly samples neighbors of nodes
        if load_adj_dir is not None:
            self.train_adj = np.load(load_adj_dir + "train_adj.npy")
            self.adj = np.load(load_adj_dir + "adj.npy")
        else:
            if not train_all_edge:
                self.construct_train_val_adj()
            else:
                self.construct_all_adj()

        if normalize and not self.feats is None:
            from sklearn.preprocessing import StandardScaler
            train_feats = self.feats[self.train_nodes]
            scaler = StandardScaler()
            scaler.fit(train_feats)
            self.feats = scaler.transform(self.feats)

        if not supervised:
            if use_random_walks:
                if load_walks and os.path.exists(prefix + "-walks.txt"):
                    walks = []
                    with open(prefix + "-walks.txt") as fp:
                        for line in fp:
                            walks.append(map(self.conversion, line.split()))
                    self.walks = walks
                    if len(walks) == 0:
                        raise Exception("Empty walks file at {0}".format(prefix + "-walks.txt"))
                else:
                    if load_walks:
                        print("Walks file not exist, run random walk with num_walk {0} and len_walk {1}".format(num_walk, walk_len))
                    else:
                        print("Run random walk with num_walk {0} and len_walk {1}".format(num_walk, walk_len))
                    self.walks = self.run_random_walks(out_file = self.prefix + "-walks.txt", num_walks = num_walk, walk_len = walk_len)

                print("Total walks edge: {0}".format(len(self.walks)))

            if not train_all_edge:
                self.construct_train_val_edge()
            else:
                self.construct_train_all_edge()

        print("Preprocessing finished, graph info:")
        print(nx.info(G))

    def remove_isolated_node(self):

        print("Removing isolated nodes")
        nodes_ids = []
        nodes = []
        num_rm_node = 0
        for nodeid in self.nodes_ids:
            neighbors = np.array([self.id_map[neighbor]
                for neighbor in self.G.neighbors(nodeid)
            ])
            deg = len(neighbors)
            if(deg > 0):
                nodes_ids.append(nodeid)
                nodes.append(self.id_map[nodeid])
            else:
                num_rm_node = num_rm_node + 1
        print("Removed {0} nodes".format(num_rm_node))
        self.nodes = np.array(nodes)
        self.nodes_ids = np.array(nodes_ids)

        print("Removing isolated train nodes")
        train_nodes_ids = []
        train_nodes = []
        num_rm_node = 0
        for nodeid in self.train_nodes_ids:
            neighbors = np.array([self.id_map[neighbor]
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])
            ])
            deg = len(neighbors)
            if(deg > 0):
                train_nodes_ids.append(nodeid)
                train_nodes.append(self.id_map[nodeid])
            else:
                num_rm_node = num_rm_node + 1
        print("Removed {0} nodes".format(num_rm_node))
        self.train_nodes = np.array(train_nodes)
        self.train_nodes_ids = np.array(train_nodes_ids)

        print("Removing isolated val nodes")
        val_nodes_ids = []
        val_nodes = []
        num_rm_node = 0
        for nodeid in self.val_nodes_ids:
            neighbors = np.array([self.id_map[neighbor]
                for neighbor in self.G.neighbors(nodeid)
            ])
            deg = len(neighbors)
            if(deg > 0):
                val_nodes_ids.append(nodeid)
                val_nodes.append(self.id_map[nodeid])
            else:
               num_rm_node = num_rm_node + 1
        print("Removed {0} nodes".format(num_rm_node))
        self.val_nodes = np.array(val_nodes)
        self.val_nodes_ids = np.array(val_nodes_ids)

        print("Removing isolated test nodes")
        test_nodes_ids = []
        test_nodes = []
        num_rm_node = 0
        for nodeid in self.test_nodes_ids:
            neighbors = np.array([self.id_map[neighbor]
                for neighbor in self.G.neighbors(nodeid)
            ])
            deg = len(neighbors)
            if(deg > 0):
                test_nodes_ids.append(nodeid)
                test_nodes.append(self.id_map[nodeid])
            else:
               num_rm_node = num_rm_node + 1
        print("Removed {0} nodes".format(num_rm_node))
        self.test_nodes = np.array(test_nodes)
        self.test_nodes_ids = np.array(test_nodes_ids)

    def construct_train_val_deg(self):
        self.deg = np.zeros((len(self.id_map),)).astype(int)
        self.train_deg = np.zeros((len(self.id_map),)).astype(int)

        #Deg
        for nodeid in self.nodes_ids:
            neighbors = np.array([self.id_map[neighbor]
                for neighbor in self.G.neighbors(nodeid)
            ])
            self.deg[self.id_map[nodeid]] = len(neighbors)
            if(len(neighbors) == 0):
                raise Exception("Node {0} is isolate node in deg".format(nodeid))

        #Train_deg
        for nodeid in self.train_nodes_ids:
            neighbors = np.array([self.id_map[neighbor]
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])
            ])
            self.train_deg[self.id_map[nodeid]] = len(neighbors)
            if(len(neighbors) == 0):
                raise Exception("Node {0} is isolate node in train deg".format(nodeid))

        if self.max_degree == None:
            self.max_degree = int(np.max(self.deg))
            print("Max degree initialized by {0}".format(self.max_degree))

    def construct_all_deg(self):
        self.deg = np.zeros((len(self.id_map),)).astype(int)
        self.train_deg = np.zeros((len(self.id_map),)).astype(int)

        #Deg
        for nodeid in self.nodes_ids:
            neighbors = np.array([self.id_map[neighbor]
                for neighbor in self.G.neighbors(nodeid)
            ])
            self.deg[self.id_map[nodeid]] = len(neighbors)
            self.train_deg[self.id_map[nodeid]] = len(neighbors)
            if(len(neighbors) == 0):
                raise Exception("Node {0} is isolate node in deg".format(nodeid))

        if self.max_degree == None:
            self.max_degree = int(np.max(self.deg))
            print("Max degree initialized by {0}".format(self.max_degree))

    def construct_train_val_adj(self):

        if self.max_degree == None:
            raise Exception("Max degree hasn't been initialized")

        self.adj = -1*np.ones((len(self.id_map)+1, self.max_degree)).astype(int)
        self.train_adj = -1*np.ones((len(self.id_map)+1, self.max_degree)).astype(int)

        #adj
        for nodeid in self.nodes_ids:
            neighbors = np.array([self.id_map[neighbor]
                for neighbor in self.G.neighbors(nodeid)
            ])

            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)

            self.adj[self.id_map[nodeid], :] = neighbors

        #Train_adj
        for nodeid in self.train_nodes_ids:
            neighbors = np.array([self.id_map[neighbor]
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])
            ])
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)

            self.train_adj[self.id_map[nodeid], :] = neighbors

    def construct_all_adj(self):

        if self.max_degree == None:
            raise Exception("Max degree hasn't been initialized")

        self.adj = -1*np.ones((len(self.id_map)+1, self.max_degree)).astype(int)
        self.train_adj = -1*np.ones((len(self.id_map)+1, self.max_degree)).astype(int)

        #adj
        for nodeid in self.nodes_ids:
            neighbors = np.array([self.id_map[neighbor]
                for neighbor in self.G.neighbors(nodeid)
            ])

            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)

            self.adj[self.id_map[nodeid], :] = neighbors
            self.train_adj[self.id_map[nodeid], :] = neighbors


    def construct_train_val_edge(self):

        #If walks == none then used G.edges as edge_adj
        if self.walks is not None:
            print("Use random walks as edges")
            edges = self.walks
        else:
            print("Use original edges")
            edges = self.G.edges()

        train_edges = []
        # val_edges = []
        # test_edges = []
        missing = 0
        print("Generate train edges")
        for n1, n2 in edges:
            if not n1 in self.G.node or not n2 in self.G.node:
                missing += 1
                continue
            if (self.id_map[n1] in self.train_nodes) and (self.id_map[n2] in self.train_nodes):
                train_edges.append((self.id_map[n1], self.id_map[n2]))
            # elif (self.id_map[n1] in self.train_nodes and self.id_map[n2] in self.val_nodes) \
            # or (self.id_map[n1] in self.val_nodes and self.id_map[n2] in self.train_nodes) \
            # or (self.id_map[n1] in self.val_nodes and self.id_map[n2] in self.val_nodes):
            #     val_edges.append((self.id_map[n1], self.id_map[n2]))
            # else:
            #     test_edges.append((self.id_map[n1], self.id_map[n2]))

        print("Unexpected missing:", missing)
        self.train_edges = np.array(train_edges)
        print("Number of training edges: {0}".format(self.train_edges.shape[0]))
        # self.val_edges = np.array(val_edges)
        # self.test_edges = np.array(test_edges)

    def construct_train_all_edge(self):

        #If walks == none then used G.edges as edge_adj
        if self.walks is not None:
            print("Use random walks as edges")
            edges = self.walks
        else:
            print("Use original edges")
            edges = self.G.edges()
        train_edges = []
        print("Generate train edges")
        for n1, n2 in edges:
            train_edges.append((self.id_map[n1], self.id_map[n2]))

        self.train_edges = np.array(train_edges)
        print("Number of training edges: {0}".format(self.train_edges.shape[0]))
        # self.val_edges = np.array(train_edges)
        # self.test_edges = np.array(train_edges)

    def run_random_walks(self, num_walks=N_WALKS, walk_len = WALK_LEN, out_file=None):
        if (self.G == None):
            print("Data has not been loaded, using load_data function first")
            return None
        nodes = self.train_nodes_ids
        pairs = []
        print("-----------------------------------------------")
        print("Random walk process")
        for count, node in enumerate(nodes):
            if self.G.degree(node) == 0:
                continue
            for i in range(num_walks):
                curr_node = node
                for j in range(walk_len):
                    next_node = np.random.choice(self.G.neighbors(curr_node))
                    curr_node = next_node
                    # self co-occurrences are useless
                    if curr_node != node:
                        pairs.append((node,curr_node))

            if (count+1) % 1000 == 0:
                print("Done walks for", count, "nodes")
        print("Done walks for", count + 1, "nodes")
        if(out_file != None):
            with open(out_file, "w") as fp:
                fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
                print("Walk info has been stored in: {0}".format(out_file))
        return pairs

if __name__ == "__main__":

    # data_dir = "/home/trunght/workspace/dataset/graph/facebook/graphsage/facebook"
    # data_dir = '../example_data/cora/graphsage/cora'
    data_dir = sys.argv[1]
    preprocessor = Dataset()
    preprocessor.load_data(prefix = data_dir, supervised=False)
    preprocessor.run_random_walks(out_file = data_dir + "_walks.txt")
