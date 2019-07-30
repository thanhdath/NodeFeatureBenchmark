from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np

from graphsage.models import SampleAndAggregate, SAGEInfo, Node2VecModel
from graphsage.minibatch import EdgeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import load_data
from openne.walker import BasicWalker

def construct_placeholders():
    # Define placeholders
    placeholders = {
        'batch1' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'batch2' : tf.placeholder(tf.int32, shape=(None), name='batch2'),
        # negative samples for all nodes in the batch
        'neg_samples': tf.placeholder(tf.int32, shape=(None,),
            name='neg_sample_size'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders

def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val = minibatch_iter.val_feed_dict(size)
    outs_val = sess.run([model.loss, model.ranks, model.mrr],
                        feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

class GraphSAGE():
    def __init__(self, graph, learning_rate=0.01, epochs=10, dropout=0.0, weight_decay=0.0, max_degree=25,
        samples=25, dim=128, neg_sample_size=20, batch_size=512,
        validate_iter=50, max_total_steps=1000000000,
        num_walks=20, walk_length=10, clf=0.8, prep_weight=None):
        self.G = graph.G
        self.graph = graph
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.max_degree = max_degree
        self.samples = samples
        self.dim = dim // 2
        self.neg_sample_size = neg_sample_size
        self.batch_size = batch_size
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.clf = clf
        self.validate_iter = validate_iter
        self.max_total_steps = max_total_steps
        self.prep_weight = prep_weight
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.train()
        tf.reset_default_graph()
        self.sess.close()

    def _run_random_walks(self):
        nodes = self.G.nodes()
        pairs = []
        print_every = len(nodes) // 2
        for count, node in enumerate(nodes):
            if self.G.degree(node) == 0:
                continue
            for i in range(self.num_walks):
                curr_node = node
                for j in range(self.walk_length):
                    next_node = np.random.choice([x for x in self.G.neighbors(curr_node)])
                    # self co-occurrences are useless
                    if curr_node != node:
                        pairs.append((node,curr_node))
                    curr_node = next_node
            if count % print_every == 0:
                print("Done walks for", count, "nodes")
        return pairs

    def build_features(self):
        # check whether graph has features
        features = None
        if 'feature' in self.G.node[list(self.G.nodes())[0]]:
            features = np.array([self.G.node[x]['feature'] for x in self.G.nodes()])
            features = np.vstack([features, np.zeros((features.shape[1],))])
        else:
            if self.prep_weight is None:
                raise Exception("If feature is None, prep must be given")
        return features

    def train(self):
        G = self.G
        features = self.build_features()
        context_pairs = self._run_random_walks()
        placeholders = construct_placeholders()
        minibatch = EdgeMinibatchIterator(G,
                placeholders,
                batch_size=self.batch_size,
                max_degree=self.max_degree,
                context_pairs=context_pairs,
                clf=self.clf)

        adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
        adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

        # self model GCN
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, self.samples, 2*self.dim)]

        model = SampleAndAggregate(placeholders,
            features,
            adj_info,
            minibatch.deg,
            layer_infos=layer_infos,
            aggregator_type="gcn",
            model_size='small',
            prep_weight=self.prep_weight,
            concat=False,
            logging=True)

        # Initialize session
        sess = self.sess
        # Init variables
        sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

        # Train model

        train_shadow_mrr = None
        shadow_mrr = None

        total_steps = 0
        avg_time = 0.0
        epoch_val_costs = []

        train_adj_info = tf.assign(adj_info, minibatch.adj)
        val_adj_info = tf.assign(adj_info, minibatch.test_adj)

        val_costs = []
        for epoch in range(self.epochs):
            minibatch.shuffle()

            iter = 0
            print('Epoch: %04d' % (epoch + 1))
            while not minibatch.end():
                # Construct feed dictionary
                feed_dict = minibatch.next_minibatch_feed_dict()
                feed_dict.update({placeholders['dropout']: self.dropout})

                t = time.time()
                # Training step
                outs = sess.run([model.opt_op, model.loss, model.ranks, model.aff_all,
                        model.mrr, model.outputs1], feed_dict=feed_dict)
                train_cost = outs[1]
                train_mrr = outs[4]
                if train_shadow_mrr is None:
                    train_shadow_mrr = train_mrr#
                else:
                    train_shadow_mrr -= (1-0.99) * (train_shadow_mrr - train_mrr)

                if iter % self.validate_iter == 0:
                    # Validation
                    sess.run(val_adj_info.op)
                    val_cost, ranks, val_mrr, duration  = evaluate(sess, model, minibatch, size=self.batch_size)
                    sess.run(train_adj_info.op)
                    val_costs.append(val_cost)

                if shadow_mrr is None:
                    shadow_mrr = val_mrr
                else:
                    shadow_mrr -= (1-0.99) * (shadow_mrr - val_mrr)

                # Print results
                avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

                if minibatch.end():
                    print("Iter:", '%04d' % iter,
                          "train_loss=", "{:.5f}".format(train_cost),
                          "train_mrr=", "{:.5f}".format(train_mrr),
                          "train_mrr_ema=", "{:.5f}".format(train_shadow_mrr), # exponential moving average
                          "val_loss=", "{:.5f}".format(val_cost),
                          "val_mrr=", "{:.5f}".format(val_mrr),
                          "val_mrr_ema=", "{:.5f}".format(shadow_mrr), # exponential moving average
                          "time=", "{:.5f}".format(avg_time))

                iter += 1
                total_steps += 1

                if total_steps > self.max_total_steps:
                    break

            # if epoch > self.early_stopping and val_costs[-1] > np.mean(val_costs[-(self.early_stopping+1):-1]):
            #     print("Early stopping...")
            #     break

            if total_steps > self.max_total_steps:
                break

        self.get_embedding(sess, model, minibatch)

    def get_embedding(self, sess, model, minibatch):
        vectors = {}
        finished = False
        iter_num = 0
        while not finished:
            feed_dict_val, finished, edges = minibatch.incremental_embed_feed_dict(self.batch_size, iter_num)
            iter_num += 1
            outs_val = sess.run([model.loss, model.mrr, model.outputs1], feed_dict=feed_dict_val)
            for i, edge in enumerate(edges):
                vectors[str(edge[0])] = outs_val[-1][i,:]
        self.vectors = vectors
