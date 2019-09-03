import tensorflow as tf

import algorithms.node_embedding.graphsagetf.models as models
import algorithms.node_embedding.graphsagetf.layers as layers
from .aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator


class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            placeholders, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean", 
            model_size="small", sigmoid_loss=False, identity_dim=0,
            learning_rate=0.01, weight_decay=5e-4, train_features=False,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        with tf.name_scope("features"):
            if features is None: 
                if identity_dim == 0:
                    raise Exception("Must have a positive value for identity feature dimension if no input features given.")
                self.features = self.embeds
            else:
                self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=train_features)
                if not self.embeds is None:
                    self.features = tf.concat([self.embeds, self.features], axis=1)
        # self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.trainable_varlist = None
        # two optimizer for finetuning
        self.optimizer_finetune = tf.train.AdamOptimizer(learning_rate=self.learning_rate/10)

        self.build()

    def freeze_aggregators(self):
        aggregator_vars = []
        for i in range(len(self.aggregators)):
            aggregator_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.aggregators[i].name_scope)
        self.trainable_varlist = [x for x in tf.trainable_variables()
            if x not in aggregator_vars]

    def switch_to_finetune_mode(self):
        self.optimizer = self.optimizer_finetune
        self.trainable_varlist = tf.trainable_variables()

    def build(self):
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                support_sizes1, concat=self.concat, model_size=self.model_size)
        dim_mult = 2 if self.concat else 1
        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)

        dim_mult = 2 if self.concat else 1
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes, 
                dropout=self.placeholders['dropout'],
                act=lambda x : x, weight_decay=self.weight_decay)
        # TF graph management
        self.node_preds = self.node_pred(self.outputs1)

        self._loss()
        if self.trainable_varlist is not None:
            grads_and_vars = self.optimizer.compute_gradients(self.loss, var_list=self.trainable_varlist)
        else:
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars) 
        self.preds = self.predict()

    def _loss(self):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += self.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)
       
        # classification loss
        if self.sigmoid_loss:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))
        else:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))

        tf.summary.scalar('loss', self.loss)

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)
