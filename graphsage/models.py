import torch.nn as nn

from graphsage.preps import prep_lookup

class SampleAndAggregate(nn.Module):

    def __init__(self, features, train_adj, adj, train_deg, deg, sampler, n_samples, agg_layers,
                 fc, identity_dim=0, weight_prep=None):

        """ Base model for GraphSAGE
            features: node features, datatype: torch tensor 
            train_adj: use in samplerSupervisedGraphSage in train mode, datatype: torch tensor 
            adj: use in samplerSupervisedGraphSage in val mode, datatype: torch tensor 
            sampler: neighborhood sampler for each node
            n_samplers: the number of node sampled at each layer
            agg_layers: list of aggregator
            fc: fully-connected layer (input: embedding scores, output: scores of each classes)
        """
        super(SampleAndAggregate, self).__init__()
        self.train_adj = train_adj
        self.adj = adj
        self.train_deg = train_deg
        self.deg = deg
        self.n_nodes = len(deg)
        self.features = features 
        self.sample_fn = sampler
        self.agg_layers = nn.Sequential(*agg_layers)
        self.num_aggs = len(agg_layers)
        self.n_samples = n_samples
        self.fc = fc
        self.mode = "train"

        if features is not None:
            input_dim = features.shape[1]
        else:
            input_dim = 0

        if identity_dim > 0:
            print("Use NodeEmbedding Preprocessing!")
            self.prep = prep_lookup['use_identity_features'](input_dim=input_dim,
                                                             n_nodes=deg.shape[0],
                                                             identity_dim=identity_dim,
                                                             weight=weight_prep)
        else:
            if input_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            print("Use Identity Preprocessing!")
            self.prep = prep_lookup['use_original_features'](input_dim=input_dim)

    def sample(self, inputs):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Output:
            samples: list of length K, samples[K-k-1] is the samples from k-hop neighbors of batch inputs            
        """
        samples = [inputs]
        for k in range(self.num_aggs):
            # Aggregator for k-hop neighbors is at layer K-k
            t = self.num_aggs - k - 1
            sampler = self.sample_fn
            nodes = sampler.sample(samples[k], self.n_samples[t])
            nodes = nodes.contiguous().view(-1) #Flatten
            samples.append(nodes)        
        return samples

    def aggregate(self, samples):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.            
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """
        has_feats = self.features is not None
        if has_feats:
            tmp_feats = [self.features[samples[i]] for i in range(len(samples))]
        else:
            tmp_feats = [None]*len(samples)
        hidden = [self.prep(samples[i], tmp_feats[i], hop_idx=len(samples) - i - 1) for i in range(len(samples))]
        layer = 0
        for aggregator in self.agg_layers.children():
            next_hidden = []
            for hop in range(self.num_aggs - layer):
                h = aggregator(hidden[hop], hidden[hop + 1])
                next_hidden.append(h)
            hidden = next_hidden
            layer = layer + 1
            
        return hidden[0]
        
    def loss(self, nodes, labels):
        raise Exception("Must use the loss function in child class")       
   