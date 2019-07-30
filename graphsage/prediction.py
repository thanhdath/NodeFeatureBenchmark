import torch
import torch.nn as nn
import torch.nn.functional as F

class BipartiteEdgePredLayer(object):
    def __init__(self, loss_fn='xent', neg_sample_weights=1.0, is_normalized_input=True):
        """
        Basic class that applies skip-gram-like loss
        (i.e., dot product (normalize is True)/cosine similarity (normalize is False) of node+target and node and negative samples)
        Args:
            bilinear_weights: use a bilinear weight for affinity calculation: u^T A v. If set to
                false, it is assumed that input dimensions are the same and the affinity will be
                based on dot product.
        """
        # self.eps = 1e-7

        self.neg_sample_weights = neg_sample_weights
        self.is_normalized_input = is_normalized_input
        # self.dropout = dropout

        # output a likelihood term
        self.output_dim = 1
        if loss_fn == 'xent':
            self.loss_fn = self._xent_loss
        else:
            print("Not implemented yet.")

    def affinity(self, inputs1, inputs2):
        """ Affinity score between batch of inputs1 and inputs2.
        Args:
            inputs1: tensor of shape [batch_size x feature_size].
        """
        # shape: [batch_size, input_dim1]        
        if self.is_normalized_input:            
            result = torch.sum(inputs1 * inputs2, dim=1)
        else:
            result = F.cosine_similarity(inputs1, inputs2)
        return result

    def neg_cost(self, inputs1, neg_samples, hard_neg_samples=None):
        """ For each input in batch, compute the sum of its affinity to negative samples.

        Returns:
            Tensor of shape [batch_size x num_neg_samples]. For each node, a list of affinities to
                negative samples is computed.
        """
        if self.is_normalized_input:
            #Use dot product if inputs are normalized 
            neg_aff = inputs1.mm(neg_samples.t())    
        else:
            #Use cosine similarity if inputs are not normalized 
            batch_size = inputs1.shape[0]
            num_neg_samples = neg_samples.shape[0]            
            stack_input = torch.stack([inputs1] * num_neg_samples, 1)        
            stack_neg_samples = torch.stack([neg_samples] * batch_size)        
            neg_aff = F.cosine_similarity(stack_input, stack_neg_samples, dim = 2)
         
        return neg_aff

    def loss(self, inputs1, inputs2, neg_samples):
        """ negative sampling loss.
        Args:
            neg_samples: tensor of shape [num_neg_samples x input_dim2]. Negative samples for all
            inputs in batch inputs1.
        """
        return self.loss_fn(inputs1, inputs2, neg_samples)

    def sigmoid_cross_entropy_with_logits(self, labels, logits):
        sig_aff = torch.sigmoid(logits)
        loss = labels * -torch.log(sig_aff) + (1 - labels) * -torch.log(1 - sig_aff)
        return loss

    def _xent_loss(self, inputs1, inputs2, neg_samples, hard_neg_samples=None):        
        cuda = inputs1.is_cuda
        true_aff = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)
        true_labels = torch.ones(true_aff.shape)
        if cuda:
            true_labels = true_labels.cuda()
        true_xent = self.sigmoid_cross_entropy_with_logits(labels=true_labels, logits=true_aff)
        neg_labels = torch.zeros(neg_aff.shape)
        if cuda:
            neg_labels = neg_labels.cuda()
        neg_xent = self.sigmoid_cross_entropy_with_logits(labels=neg_labels, logits=neg_aff)
        loss = true_xent.sum() + self.neg_sample_weights * neg_xent.sum()
        return loss
