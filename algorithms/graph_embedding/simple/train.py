import numpy as np
import torch
from algorithms.logistic_regression import LogisticRegressionPytorch

class Simple():
    def __init__(self, operator='mean', l2_norm=False):
        self.operator = operator
        if operator == 'mean':
            self.func = lambda x: x.mean(axis=0)
        elif operator == 'sum':
            self.func = lambda x: x.sum(axis=0)
        elif operator == 'max':
            self.func = lambda x: x.max(axis=0)
        if l2_norm:
            self.norm = lambda x: x / np.sqrt(np.sum(x**2, axis=1))
        else:
            self.norm = lambda x: x

    def forward(self, node_embs):
        """
        node_embs: numpy array
        """
        graph_embs = self.norm(self.func(node_embs))
        return graph_embs

def gen_graph_embs(dataset, model_emb):
    graph_embeds = []
    graph_labels = []
    for graph, labels in dataset:
        graph_emb = model_emb.forward(graph.ndata['feat'])
        graph_embeds.append(graph_emb)
        graph_labels.append(graph_labels)
    graph_embeds = torch.FloatTensor(graph_embeds)
    graph_labels = np.array(graph_labels)
    if len(graph_labels.shape) == 2 and graph_labels.shape[1] > 1:
        multiclass = True
        graph_labels = torch.FloatTensor(graph_labels)
    else:
        multiclass = False
        graph_labels = torch.LongTensor(graph_labels)
    return graph_embeds, graph_labels, multiclass

def gen_mask(idx, l):
    mask = np.zeros((l))
    mask[idx] = 1
    mask = torch.ByteTensor(mask)
    return mask

def simple_api(args):
    dataset = args.dataset
    dataset_train = dataset.dataset_train
    dataset_val = dataset.dataset_val 
    dataset_test = dataset.dataset_test

    model_emb = Simple(operator=args.operator, l2_norm=args.l2_norm)
    train_embs, train_labels, multiclass = gen_graph_embs(dataset_train, model_emb)
    val_embs, val_labels, _ = gen_graph_embs(dataset_val, model_emb)
    test_embs, test_labels, _ = gen_graph_embs(dataset_test, model_emb)

    embs = torch.cat([train_embs, val_embs, test_embs], dim=0)
    labels = torch.cat([train_labels, val_labels, test_labels], dim=0)
    train_mask = gen_mask(range(0, len(train_embs)), len(embs))
    val_mask = gen_mask(range(len(train_embs), len(train_embs)+len(val_embs)), len(embs))
    test_mask = gen_mask(range(len(train_embs)+len(val_embs), len(embs)), len(embs))

    logreg = LogisticRegressionPytorch(embs, labels, train_mask, val_mask, test_mask,
        epochs=300, multiclass=multiclass, cuda=args.cuda)
    
