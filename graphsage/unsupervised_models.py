# Use in fixed_unigram_candidate_sampler
import numpy as np
import torch
import torch.nn.functional as F
import pdb

from graphsage.models import SampleAndAggregate
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.prediction import BipartiteEdgePredLayer

def fixed_unigram_candidate_sampler(num_sampled, unique, range_max, distortion, unigrams):
    weights = unigrams ** distortion
    prob = weights / weights.sum()
    sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
    return sampled


class UnsupervisedGraphSage(SampleAndAggregate):

    def __init__(self, features, train_adj, adj, train_deg, deg, sampler, n_samples, agg_layers, fc, identity_dim,
                 neg_sample_size, normalize_embedding=True, use_cuda=True, weight_prep=None):

        super(UnsupervisedGraphSage, self).__init__(features, train_adj, adj, train_deg, deg, sampler, n_samples,
                                                    agg_layers, fc, identity_dim, weight_prep=weight_prep)
        self.link_pred_layer = BipartiteEdgePredLayer(is_normalized_input=normalize_embedding)
        self.degrees = train_deg
        self.neg_sample_size = neg_sample_size
        self.normalize_embedding = normalize_embedding
        self.use_cuda = use_cuda

    def loss(self, inputs1, inputs2):
        batch_size = inputs1.size()[0]
        outputs1, outputs2, neg_outputs = self.forward(inputs1, inputs2)
        loss = self.link_pred_layer.loss(outputs1, outputs2, neg_outputs) / batch_size
        return loss, outputs1, outputs2, neg_outputs

    def forward(self, inputs1, inputs2, mode="train", input_samples=None):

        if mode != self.mode:
            if mode == 'train':
                self.sample_fn = UniformNeighborSampler(self.train_adj)
                self.degrees = self.train_deg
            else:
                self.sample_fn = UniformNeighborSampler(self.adj)
                self.degrees = self.deg
        if self.train_adj is None:
            self.degrees = self.deg

        neg = fixed_unigram_candidate_sampler(
            num_sampled=self.neg_sample_size,
            unique=False,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees
        )

        neg = torch.LongTensor(neg)
        if inputs1.is_cuda:
            neg = neg.cuda()

        # perform "convolution"
        if input_samples is not None:
            samples1, samples2, neg_samples = input_samples
        else:
            samples1 = self.sample(inputs1)
            samples2 = self.sample(inputs2)
            neg_samples = self.sample(neg)

        outputs1 = self.aggregate(samples1)
        outputs2 = self.aggregate(samples2)
        neg_outputs = self.aggregate(neg_samples)

        if mode == "save_embedding":
            return samples1, samples2, neg_samples, outputs1, outputs2, neg_outputs

        if self.normalize_embedding:
            outputs1 = F.normalize(outputs1, dim=1)
            outputs2 = F.normalize(outputs2, dim=1)
            neg_outputs = F.normalize(neg_outputs, dim=1)

        # Normalize
        if self.fc:
            outputs1 = self.fc(outputs1)
            outputs2 = self.fc(outputs2, dim=1)
            neg_outputs = self.fc(neg_outputs, dim=1)

        return outputs1, outputs2, neg_outputs

    def accuracy(self, outputs1, outputs2, neg_outputs):
        # shape: [batch_size]
        cuda = outputs1.is_cuda
        aff = self.link_pred_layer.affinity(outputs1, outputs2)
        batch_size = outputs1.size()[0]
        neg_sample_size = neg_outputs.size()[0]
        # shape : [batch_size x num_neg_samples]
        neg_aff = self.link_pred_layer.neg_cost(outputs1, neg_outputs)
        neg_aff = neg_aff.view(batch_size, neg_sample_size)
        _aff = aff.unsqueeze(1)
        aff_all = torch.cat([neg_aff, _aff], dim=1)
        size = aff_all.size()[1]
        _, indices_of_ranks = torch.topk(aff_all, k=size)
        _, ranks = torch.topk(-indices_of_ranks, k=size)

        # mrr = torch.mean(1.0/(ranks[:, -1] + 1))
        ranks = (ranks[:, -1] + 1).cpu().numpy()
        conv_ranks = 1.0 / ranks
        conv_ranks = torch.FloatTensor(conv_ranks)
        if cuda:
            conv_ranks = conv_ranks.cuda()
        mrr = torch.mean(conv_ranks)
        return mrr

    def forward_to_get_embedding(self, batch_nodes):
        batch_sample = self.sample(batch_nodes)
        batch_ouput = self.aggregate(batch_sample)
        batch_ouput = F.normalize(batch_ouput, dim=1)
        return batch_ouput

    def get_embedding(self):
        # Returns: embedding as lookup tables for all nodes
        nodes = np.arange(self.n_nodes)
        nodes = torch.LongTensor(nodes)
        if self.use_cuda:
            nodes = nodes.cuda()
        embedding = None
        BATCH_SIZE = 512

        for i in range(0, self.n_nodes, BATCH_SIZE):
            j = min(i + BATCH_SIZE, self.n_nodes)
            batch_nodes = nodes[i:j]
            if batch_nodes.shape[0] == 0: break
            batch_node_embeddings = self.forward_to_get_embedding(batch_nodes)
            if embedding is None:
                embedding = batch_node_embeddings
            else:
                embedding = torch.cat((embedding, batch_node_embeddings))

        return embedding
