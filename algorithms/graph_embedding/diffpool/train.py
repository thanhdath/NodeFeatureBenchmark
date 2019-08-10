import os
import numpy as np
import torch
import dgl
import networkx as nx
import random
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import dgl.function as fn

from .model.encoder import DiffPool
from .data_utils import pre_process
import torch
# from main import get_feature_initialization
from utils import f1

def prepare_data(dataset, prog_args, train=False, pre_process=None):
    '''
    preprocess TU dataset according to DiffPool's paper setting and load dataset into dataloader
    '''
    if train:
        shuffle = True
    else:
        shuffle = False

    if pre_process:
        pre_process(dataset, prog_args)

    # dataset.set_fold(fold)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=prog_args.batch_size,
                                       shuffle=shuffle,
                                       collate_fn=collate_fn,
                                       drop_last=True,
                                       num_workers=prog_args.n_worker)


def graph_classify_task(prog_args):
    '''
    perform graph classification task
    '''

    # dataset = tu.TUDataset(name=prog_args.dataset)
    # dataset = TUDataset(prog_args.dataset, prog_args)
    # train_size = int(prog_args.train_ratio * len(dataset))
    # test_size = int(prog_args.test_ratio * len(dataset))
    # val_size = int(len(dataset) - train_size - test_size)

    # split train test
    # train_mask, val_mask, test_mask = split_train_test(len(dataset), seed=args.seed)
    dataset = prog_args.dataset
    dataset_train = dataset.dataset_train
    dataset_val = dataset.dataset_val
    dataset_test = dataset.dataset_test
    # 
    train_dataloader = prepare_data(dataset_train, prog_args, train=True)
    val_dataloader = prepare_data(dataset_val, prog_args, train=False)
    test_dataloader = prepare_data(dataset_test, prog_args, train=False)
    input_dim, label_dim, max_num_node = dataset.statistics()
    print("++++++++++STATISTICS ABOUT THE DATASET")
    print("dataset feature dimension is", input_dim)
    print("dataset label dimension is", label_dim)
    print("the max num node is", max_num_node)
    print("number of graphs is", len(dataset))
    # assert len(dataset) % prog_args.batch_size == 0, "training set not divisible by batch size"

    hidden_dim = prog_args.hidden_dim  # used to be 64
    embedding_dim = prog_args.output_dim

    # calculate assignment dimension: pool_ratio * largest graph's maximum
    # number of nodes  in the dataset
    assign_dim = int(max_num_node * prog_args.pool_ratio) * \
        prog_args.batch_size
    print("++++++++++MODEL STATISTICS++++++++")
    print("model hidden dim is", hidden_dim)
    print("model embedding dim for graph instance embedding", embedding_dim)
    print("initial batched pool graph dim is", assign_dim)
    activation = F.relu

    # initialize model
    # 'diffpool' : diffpool
    model = DiffPool(input_dim,
                     hidden_dim,
                     embedding_dim,
                     label_dim,
                     activation,
                     prog_args.gc_per_block,
                     prog_args.dropout,
                     prog_args.num_pool,
                     prog_args.linkpred,
                     prog_args.batch_size,
                     'meanpool',
                     assign_dim,
                     prog_args.pool_ratio)

    print("model init finished")
    if prog_args.cuda >= 0:
        model = model.cuda()

    logger = train(
        train_dataloader,
        model,
        prog_args,
        val_dataset=val_dataloader)
    micro, macro, _ = evaluate(test_dataloader, model, prog_args, logger)
    print('Test micro-macro: {:.3f}\t{:.3f}'.format(micro, macro))

def collate_fn(batch):
    '''
    collate_fn for dataset batching
    transform ndata to tensor (in gpu is available)
    '''
    graphs, labels = map(list, zip(*batch))
    #cuda = torch.cuda.is_available()

    # batch graphs and cast to PyTorch tensor
    for graph in graphs:
        for (key, value) in graph.ndata.items():
            graph.ndata[key] = torch.FloatTensor(value)
    batched_graphs = dgl.batch(graphs)

    # cast to PyTorch tensor
    batched_labels = torch.LongTensor(np.array(labels))

    return batched_graphs, batched_labels


def train(dataset, model, prog_args, same_feat=True, val_dataset=None):
    '''
    training function
    '''
    dataloader = dataset
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()), lr=0.001)
    early_stopping_logger = {"best_epoch": -1, "val_acc": -1}

    if prog_args.cuda > 0:
        torch.cuda.set_device(0)
    npt = 0
    for epoch in range(prog_args.epoch):
        begin_time = time.time()
        model.train()
        accum_correct = 0
        total = 0
        print("EPOCH ###### {} ######".format(epoch))
        computation_time = 0.0
        for (batch_idx, (batch_graph, graph_labels)) in enumerate(dataloader):
            if torch.cuda.is_available():
                for (key, value) in batch_graph.ndata.items():
                    batch_graph.ndata[key] = value.cuda()
                graph_labels = graph_labels.cuda()

            model.zero_grad()
            compute_start = time.time()
            ypred = model(batch_graph)
            indi = torch.argmax(ypred, dim=1)
            correct = torch.sum(indi == graph_labels).item()
            accum_correct += correct
            total += graph_labels.size()[0]
            loss = model.loss(ypred, graph_labels)
            loss.backward()
            batch_compute_time = time.time() - compute_start
            computation_time += batch_compute_time
            nn.utils.clip_grad_norm_(model.parameters(), prog_args.clip)
            optimizer.step()

        train_accu = accum_correct / total
        print("train accuracy for this epoch {} is {}%".format(epoch,
                                                               train_accu * 100))
        elapsed_time = time.time() - begin_time
        print("loss {} with epoch time {} s & computation time {} s ".format(
            loss.item(), elapsed_time, computation_time))
        if val_dataset is not None:
            _,_, result = evaluate(val_dataset, model, prog_args)
            print("validation  accuracy {}%".format(result * 100))
            if result >= early_stopping_logger['val_acc'] and result <=\
                    train_accu:
                early_stopping_logger.update(best_epoch=epoch, val_acc=result)
                torch.save(model.state_dict(), 'diffpool-best-model.pkl')
                npt = 0
            else:
                npt += 1
            if npt > 150:
                break
            print("best epoch is EPOCH {}, val_acc is {}%".format(early_stopping_logger['best_epoch'],
                                                                  early_stopping_logger['val_acc'] * 100))
        torch.cuda.empty_cache()
    model.load_state_dict(torch.load('diffpool-best-model.pkl'))
    return early_stopping_logger


def evaluate(dataloader, model, prog_args, logger=None):
    '''
    evaluate function
    '''
    model.eval()
    correct_label = 0
    ypreds = []
    labelss = []
    with torch.no_grad():
        for batch_idx, (batch_graph, graph_labels) in enumerate(dataloader):
            if torch.cuda.is_available():
                for (key, value) in batch_graph.ndata.items():
                    batch_graph.ndata[key] = value.cuda()
                graph_labels = graph_labels.cuda()
            ypred = model(batch_graph)
            ypreds.append(ypred)
            labelss.append(graph_labels)
            indi = torch.argmax(ypred, dim=1)
            correct = torch.sum(indi == graph_labels)
            correct_label += correct.item()
    acc = correct_label / (len(dataloader) * prog_args.batch_size)
    ypreds = torch.cat(ypreds, 0)
    labelss = torch.cat(labelss, 0)
    micro, macro = f1(ypreds, labelss)
    return micro, macro, acc


def diffpool_api(params):
    """
    params: types.SimpleNamespace
    """
    graph_classify_task(params)

if __name__ == "__main__":
    prog_args = arg_parse()
    print(prog_args)
    random.seed(prog_args.seed)
    np.random.seed(prog_args.seed)
    torch.manual_seed(prog_args.seed)
    graph_classify_task(prog_args)
