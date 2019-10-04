
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics

from .supervised_models import SupervisedGraphsage
from .models import SAGEInfo
from .minibatch import NodeMinibatchIterator
from .neigh_samplers import UniformNeighborSampler
import tensorboardX

def calc_f1(y_true, y_pred, sigmoid=False):
    if not sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

# Define model evaluation function


def evaluate(sess, model, minibatch_iter, size=None, sigmoid=False):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss], feed_dict=feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0], sigmoid)
    return node_outs_val[1], mic, mac, (time.time() - t_test)


def incremental_evaluate(sess, model, minibatch_iter, size, test=False, sigmoid=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _ = minibatch_iter.incremental_node_val_feed_dict(
            size, iter_num, test=test)
        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds, sigmoid=sigmoid)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)


def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch': tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders


class Graphsage():
    def __init__(self, data, features, batch_size=128, max_degree=100, model='mean',
                 samples_1=25, samples_2=10, samples_3=10, dim_1=128, dim_2=128, lr=0.01,
                 model_size="small", identity_dim=0, epochs=400, dropout=0.2, train_features=False,
                 load_model=None, suffix=""):
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.model = model
        self.samples_1 = samples_1
        self.samples_2 = samples_2
        self.samples_3 = samples_3
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.model_size = model_size
        self.identity_dim = identity_dim
        self.epochs = epochs
        self.dropout = dropout
        self.train_features = train_features

        self.data = data
        self.features = features
        self.sigmoid = self.data.multiclass
        self.load_model = load_model
        self.suffix = suffix
        self.lr = lr
        if self.load_model:
            self.epochs = self.epochs // 2

    def train(self):
        train_data = self.data 
        G = train_data.graph
        features = self.features.numpy()
        class_map = {node: train_data.labels[i].numpy().tolist() for i, node in enumerate(G.nodes())}
        num_classes = train_data.n_classes

        if not features is None:
            # pad with dummy zero vector
            features = np.vstack([features, np.zeros((features.shape[1],))])

        placeholders = construct_placeholders(num_classes)

        train_indices = np.argwhere(train_data.train_mask).flatten()
        val_indices = np.argwhere(train_data.val_mask).flatten()
        test_indices = np.argwhere(train_data.test_mask).flatten()
        train_nodes = np.array(train_data.graph.nodes())[train_indices]
        val_nodes = np.array(train_data.graph.nodes())[val_indices]
        test_nodes = np.array(train_data.graph.nodes())[test_indices]

        minibatch = NodeMinibatchIterator(G,
                                          train_nodes, val_nodes, test_nodes,
                                          placeholders,
                                          class_map,
                                          num_classes,
                                          batch_size=self.batch_size,
                                          max_degree=self.max_degree,
                                          context_pairs=None)
        adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
        adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

        if self.model == 'mean':
            # Create model
            sampler = UniformNeighborSampler(adj_info)
            if self.samples_3 != 0:
                layer_infos = [SAGEInfo("node", sampler, self.samples_1, self.dim_1),
                               SAGEInfo("node", sampler,
                                        self.samples_2, self.dim_2),
                               SAGEInfo("node", sampler, self.samples_3, self.dim_2)]
            elif self.samples_2 != 0:
                layer_infos = [SAGEInfo("node", sampler, self.samples_1, self.dim_1),
                               SAGEInfo("node", sampler, self.samples_2, self.dim_2)]
            else:
                layer_infos = [
                    SAGEInfo("node", sampler, self.samples_1, self.dim_1)]

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos,
                                        model_size=self.model_size,
                                        sigmoid_loss=self.sigmoid,
                                        identity_dim=self.identity_dim,
                                        logging=True,
                                        train_features=self.train_features,
                                        learning_rate=self.lr,
                                        weight_decay=0.0)
        elif self.model == 'gcn':
            # Create model
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.samples_1, 2*self.dim_1)]
            n_layers = 1 
            for i in range(n_layers-1):
                layer_infos.append(SAGEInfo("node", sampler, self.samples_2, 2*self.dim_2))

            model = SupervisedGraphsage(num_classes, placeholders, 
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos=layer_infos, 
                                        aggregator_type="gcn",
                                        model_size=self.model_size,
                                        concat=False,
                                        sigmoid_loss = self.sigmoid,
                                        identity_dim = self.identity_dim,
                                        logging=True,
                                        weight_decay=0.0)
        elif self.model == 'maxpool':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.samples_1, self.dim_1),
                                SAGEInfo("node", sampler, self.samples_2, self.dim_2)]

            model = SupervisedGraphsage(num_classes, placeholders, 
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos=layer_infos, 
                                        aggregator_type="maxpool",
                                        model_size=self.model_size,
                                        sigmoid_loss = self.sigmoid,
                                        identity_dim = self.identity_dim,
                                        logging=True,
                                        weight_decay=0.0)
        elif self.model == 'graphsage_seq':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.samples_1, self.dim_1),
                                SAGEInfo("node", sampler, self.samples_2, self.dim_2)]

            model = SupervisedGraphsage(num_classes, placeholders, 
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos=layer_infos, 
                                        aggregator_type="seq",
                                        model_size=self.model_size,
                                        sigmoid_loss = self.sigmoid,
                                        identity_dim = self.identity_dim,
                                        logging=True,
                                        weight_decay=0.0)
        else:
            raise Exception('Error: model name unrecognized.')

        sigmoid = self.sigmoid
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = .8
        # config.allow_soft_placement = True

        # Initialize session
        sess = tf.Session(config=config)
        merged = tf.summary.merge_all()

        # Init variables
        sess.run(tf.global_variables_initializer(),
                 feed_dict={adj_info_ph: minibatch.adj})

        # Train model

        total_steps = 0
        avg_time = 0.0
        epoch_val_costs = []

        train_adj_info = tf.assign(adj_info, minibatch.adj)
        val_adj_info = tf.assign(adj_info, minibatch.test_adj)

        # save model
        vars2save = []
        for i in range(len(model.aggregators)):
            vars2save += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.aggregators[i].name_scope)
        saver = tf.train.Saver({x.name: x for x in vars2save})
        # restore pretrain model
        if self.load_model is not None:
            saver.restore(sess, self.load_model)
            print("Restore pretrained model: ", self.load_model)

            from_data = self.load_model.replace("graphsage-best-model-", "").replace("/model.ckpt", "")
            best_model_name = 'graphsage-best-model-{}-from-{}/model.ckpt'.format(self.suffix, from_data)
        else:
            best_model_name = 'graphsage-best-model-{}/model.ckpt'.format(self.suffix)

        # writer = tf.summary.FileWriter("logs", sess.graph)
        writer = tensorboardX.SummaryWriter("logs/"+best_model_name.replace("/model.ckpt", ""))
        best_val_acc = -1
        if self.load_model is not None: # finetune
            model.freeze_aggregators()
            print("Freeze aggregators , train classification layer.")
        for epoch in range(self.epochs):
            minibatch.shuffle()

            iter = 0
            print('Epoch: %04d' % (epoch + 1))
            epoch_val_costs.append(0)
            while not minibatch.end():
                # Construct feed dictionary
                feed_dict, labels = minibatch.next_minibatch_feed_dict()
                feed_dict.update({placeholders['dropout']: self.dropout})

                t = time.time()
                # Training step
                outs = sess.run([merged, model.opt_op, model.loss, model.preds], feed_dict=feed_dict)
                train_cost = outs[2]

                # Print results
                avg_time = (avg_time * total_steps +
                            time.time() - t) / (total_steps + 1)

                if iter == 0:
                    train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1], sigmoid=sigmoid)
                    print("Iter:", '%04d' % iter,
                          "train_loss=", "{:.5f}".format(train_cost),
                          "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                          "train_f1_mac=", "{:.5f}".format(train_f1_mac),
                          "time=", "{:.5f}".format(avg_time))
                    writer.add_scalar("train_f1", train_f1_mic, epoch)
                iter += 1 
                total_steps += 1
             # Validation
            sess.run(val_adj_info.op)
            val_cost, val_f1_mic, val_f1_mac, duration = evaluate(sess, model, minibatch, sigmoid=sigmoid)
            sess.run(train_adj_info.op)
            epoch_val_costs[-1] += val_cost
            print("val_loss=", "{:.5f}".format(val_cost),
                "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                "val_f1_mac=", "{:.5f}".format(val_f1_mac))
            writer.add_scalar("val_f1", val_f1_mic, epoch)
            if val_f1_mic > best_val_acc:
                saver.save(sess, best_model_name)
                best_val_acc = val_f1_mic
                
        # unfreeze aggregators and train the whole graph
        if self.load_model is not None:
            print("Train the whole graph with lr/10")
            model.switch_to_finetune_mode()
            for epoch in range(self.epochs):
                minibatch.shuffle()

                iter = 0
                print('Epoch: %04d' % (epoch + 1))
                epoch_val_costs.append(0)
                while not minibatch.end():
                    # Construct feed dictionary
                    feed_dict, labels = minibatch.next_minibatch_feed_dict()
                    feed_dict.update({placeholders['dropout']: self.dropout})

                    t = time.time()
                    # Training step
                    outs = sess.run([merged, model.opt_op, model.loss, model.preds], feed_dict=feed_dict)
                    train_cost = outs[2]

                    # Print results
                    avg_time = (avg_time * total_steps +
                                time.time() - t) / (total_steps + 1)

                    if iter == 0:
                        train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1], sigmoid=sigmoid)
                        print("Iter:", '%04d' % iter,
                            "train_loss=", "{:.5f}".format(train_cost),
                            "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                            "train_f1_mac=", "{:.5f}".format(train_f1_mac),
                            "time=", "{:.5f}".format(avg_time))
                        writer.add_scalar("train_f1", train_f1_mic, epoch+self.epochs)
                    iter += 1 
                    total_steps += 1
                # Validation
                sess.run(val_adj_info.op)
                val_cost, val_f1_mic, val_f1_mac, duration = evaluate(sess, model, minibatch, sigmoid=sigmoid)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost
                print("val_loss=", "{:.5f}".format(val_cost),
                    "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                    "val_f1_mac=", "{:.5f}".format(val_f1_mac))
                writer.add_scalar("val_f1", val_f1_mic, epoch+self.epochs)
                if val_f1_mic > best_val_acc:
                    saver.save(sess, best_model_name)
                    best_val_acc = val_f1_mic

        saver.restore(sess, best_model_name)
        save_path = saver.save(sess, best_model_name)
        print("Model saved in path: %s" % save_path)
        writer.close()
        print("Optimization Finished!")
        sess.run(val_adj_info.op)
        val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, self.batch_size, test=True, sigmoid=sigmoid)
        print("Test micro-macro: {:.3f}\t{:.3f}".format(val_f1_mic, val_f1_mac))


if __name__ == '__main__':
    tf.app.run()
