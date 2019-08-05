import os
import numpy as np
import torch
import dgl
import networkx as nx
import argparse
import random
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import tu

from model.encoder import DiffPool
from data_utils import pre_process
import torch
from main import get_feature_initialization
from SGC.metrics import f1
from utils import split_train_test

from dgl.data.utils import download, extract_archive, get_download_dir
class TUDataset(object):
    """
    TUDataset contains lots of graph kernel datasets for graph classification.
    Use provided node feature by default. If no feature provided, use one-hot node label instead.
    If neither labels provided, use constant for node feature.

    :param name: Dataset Name, such as `ENZYMES`, `DD`, `COLLAB`
    :param use_pandas: Default: False.
        Numpy's file read function has performance issue when file is large,
        using pandas can be faster.
    :param hidden_size: Default 10. Some dataset doesn't contain features.
        Use constant node features initialization instead, with hidden size as `hidden_size`.

    """

    _url = r"https://ls11-www.cs.uni-dortmund.de/people/morris/graphkerneldatasets/{}.zip"

    def __init__(self, name, args, use_pandas=False, hidden_size=10):

        self.name = name
        self.hidden_size = hidden_size
        # self.extract_dir = self._download()
        self.extract_dir = "data"
        if use_pandas:
            import pandas as pd
            DS_edge_list = self._idx_from_zero(
                pd.read_csv(self._file_path("A"), delimiter=",", dtype=int, header=None).values)
        else:
            DS_edge_list = self._idx_from_zero(
                np.genfromtxt(self._file_path("A"), delimiter=",", dtype=int))

        DS_indicator = self._idx_from_zero(
            np.genfromtxt(self._file_path("graph_indicator"), dtype=int))
        DS_graph_labels = self._idx_from_zero(
            np.genfromtxt(self._file_path("graph_labels"), dtype=int))

        g = dgl.DGLGraph()
        g.add_nodes(int(DS_edge_list.max()) + 1)
        g.add_edges(DS_edge_list[:, 0], DS_edge_list[:, 1])

        node_idx_list = []
        for idx in range(np.max(DS_indicator) + 1):
            node_idx = np.where(DS_indicator == idx)
            node_idx_list.append(node_idx[0])
        self.graph_lists = g.subgraphs(node_idx_list)
        self.graph_labels = DS_graph_labels

        stime = time.time()
        if args.init == "ori": # use node attributes
            print("Init features: Original , node attributes")
            DS_node_attr = np.loadtxt(self._file_path("node_attributes"), delimiter=",")
            for idxs, g in zip(node_idx_list, self.graph_lists):
                g.ndata['feat'] = DS_node_attr[idxs, :]
            if args.norm_features:
                g.ndata['feat'] = (g.ndata['feat'] - g.ndata['feat'].mean())/g.ndata['feat'].std()
        elif args.init == "label": # use node label as node features
            print("Init features: node labels")
            DS_node_labels = self._idx_from_zero(np.loadtxt(self._file_path("node_labels"), dtype=int))
            g.ndata['node_label'] = DS_node_labels
            one_hot_node_labels = self._to_onehot(DS_node_labels)
            for idxs, g in zip(node_idx_list, self.graph_lists):
                g.ndata['feat'] = one_hot_node_labels[idxs, :]
            if args.norm_features:
                g.ndata['feat'] = (g.ndata['feat'] - g.ndata['feat'].mean())/g.ndata['feat'].std()
        else:
            print("Init features:", args.init)
            for graph in self.graph_lists:
                features = get_feature_initialization(args, graph.to_networkx(), inplace=False)
                graph.ndata['feat'] = np.array([features[int(x)] for x in graph.nodes()])
        print("Time init features: {:.3f}s".format(time.time()-stime))

        # try:
            # DS_node_labels = self._idx_from_zero(np.loadtxt(self._file_path("node_labels"), dtype=int))
            # g.ndata['node_label'] = DS_node_labels
            # one_hot_node_labels = self._to_onehot(DS_node_labels)
            # for idxs, g in zip(node_idx_list, self.graph_lists):
            #     g.ndata['feat'] = one_hot_node_labels[idxs, :]
        # except IOError:
        #     print("No Node Label Data")
        # try:
            # DS_node_attr = np.loadtxt(self._file_path("node_attributes"), delimiter=",")
            # for idxs, g in zip(node_idx_list, self.graph_lists):
            #     g.ndata['feat'] = DS_node_attr[idxs, :]
        # except IOError:
        #     print("No Node Attribute Data")
        # if 'feat' not in g.ndata.keys():
        #     for idxs, g in zip(node_idx_list, self.graph_lists):
        #         g.ndata['feat'] = np.ones((g.number_of_nodes(), hidden_size))
        #     print("Use Constant one as Feature with hidden size {}".format(hidden_size))

    def __getitem__(self, idx):
        """Get the i^th sample.
        Paramters
        ---------
        idx : int
            The sample index.
        Returns
        -------
        (dgl.DGLGraph, int)
            DGLGraph with node feature stored in `feat` field and node label in `node_label` if available.
            And its label.
        """
        g = self.graph_lists[idx]
        return g, self.graph_labels[idx]

    def __len__(self):
        return len(self.graph_lists)

    def _file_path(self, category):
        return os.path.join(self.extract_dir, self.name, "{}_{}.txt".format(self.name, category))

    @staticmethod
    def _idx_from_zero(idx_tensor):
        return idx_tensor - np.min(idx_tensor)

    @staticmethod
    def _to_onehot(label_tensor):
        label_num = label_tensor.shape[0]
        assert np.min(label_tensor) == 0
        one_hot_tensor = np.zeros((label_num, np.max(label_tensor) + 1))
        one_hot_tensor[np.arange(label_num), label_tensor] = 1
        return one_hot_tensor

    def statistics(self):
        input_dim = self.graph_lists[0].ndata['feat'].shape[1]
        label_dim = self.graph_labels.max() + 1
        max_num_nodes = max([len(x.nodes()) for x in self.graph_lists])
        return input_dim, label_dim, max_num_nodes


def arg_parse():
    '''
    argument parser
    '''
    parser = argparse.ArgumentParser(description='DiffPool arguments')
    parser.add_argument('--data', dest='data', help='Input Dataset')
    parser.add_argument(
        '--pool_ratio',
        dest='pool_ratio',
        type=float,
        help='pooling ratio')
    parser.add_argument(
        '--num_pool',
        dest='num_pool',
        type=int,
        help='num_pooling layer')
    parser.add_argument('--no_link_pred', dest='linkpred', action='store_false',
                        help='switch of link prediction object')
    parser.add_argument('--cuda', dest='cuda', type=int, help='switch cuda')
    parser.add_argument('--lr', dest='lr', type=float, help='learning rate')
    parser.add_argument(
        '--clip',
        dest='clip',
        type=float,
        help='gradient clipping')
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        help='batch size')
    parser.add_argument('--epochs', dest='epoch', type=int,
                        help='num-of-epoch')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='ratio of trainning dataset split')
    parser.add_argument('--test-ratio', dest='test_ratio', type=float,
                        help='ratio of testing dataset split')
    parser.add_argument('--num_workers', dest='n_worker', type=int,
                        help='number of workers when dataloading')
    parser.add_argument('--gc-per-block', dest='gc_per_block', type=int,
                        help='number of graph conv layer per block')
    parser.add_argument('--bn', dest='bn', action='store_const', const=True,
                        default=True, help='switch for bn')
    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='dropout rate')
    parser.add_argument('--bias', dest='bias', action='store_const',
                        const=True, default=True, help='switch for bias')
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='model saving directory: SAVE_DICT/DATASET')
    parser.add_argument('--load_epoch', dest='load_epoch', help='load trained model params from\
                         SAVE_DICT/DATASET/model-LOAD_EPOCH')
    parser.add_argument('--data_mode', dest='data_mode', help='data\
                        preprocessing mode: default, id, degree, or one-hot\
                        vector of degree number', choices=['default', 'id', 'deg',
                                                           'deg_num'])


    parser.add_argument('--init', type=str, default="ori", help="Features initialization method")
    parser.add_argument('--feature_size', type=int, default=128, help="Features dimension")
    parser.add_argument('--norm_features', action='store_true', help="norm features by standard scaler.")
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--verbose', type=int, default=0)
    parser.set_defaults(dataset='ENZYMES',
                        pool_ratio=0.15,
                        num_pool=1,
                        cuda=1,
                        lr=1e-3,
                        clip=2.0,
                        batch_size=20,
                        epoch=4000,
                        train_ratio=0.7,
                        test_ratio=0.2,
                        n_worker=1,
                        gc_per_block=3,
                        dropout=0.0,
                        method='diffpool',
                        bn=True,
                        bias=True,
                        save_dir="./model_param",
                        load_epoch=-1,
                        data_mode='default')
    return parser.parse_args()


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
    dataset = TUDataset(prog_args.dataset, prog_args)
    train_size = int(prog_args.train_ratio * len(dataset))
    test_size = int(prog_args.test_ratio * len(dataset))
    val_size = int(len(dataset) - train_size - test_size)

    # split train test
    train_mask, val_mask, test_mask = split_train_test(len(dataset), seed=args.seed)
    dataset_train = dataset[np.argwhere(train_mask==1).flatten()]
    dataset_val = dataset[np.argwhere(val_mask == 1).flatten()]
    dataset_test = dataset[np.argwhere(test_mask == 1).flatten()]
    # 

    train_dataloader = prepare_data(dataset_train, prog_args, train=True,
                                    pre_process=pre_process)
    val_dataloader = prepare_data(dataset_val, prog_args, train=False,
                                  pre_process=pre_process)
    test_dataloader = prepare_data(dataset_test, prog_args, train=False,
                                   pre_process=pre_process)
    input_dim, label_dim, max_num_node = dataset.statistics()
    print("++++++++++STATISTICS ABOUT THE DATASET")
    print("dataset feature dimension is", input_dim)
    print("dataset label dimension is", label_dim)
    print("the max num node is", max_num_node)
    print("number of graphs is", len(dataset))
    # assert len(dataset) % prog_args.batch_size == 0, "training set not divisible by batch size"

    hidden_dim = 64  # used to be 64
    embedding_dim = 64

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

    if prog_args.load_epoch >= 0 and prog_args.save_dir is not None:
        model.load_state_dict(torch.load(prog_args.save_dir + "/" + prog_args.dataset
                                         + "/model.iter-" + str(prog_args.load_epoch)))

    print("model init finished")
    print("MODEL:::::::", prog_args.method)
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
    dir = prog_args.save_dir + "/" + prog_args.dataset
    if not os.path.exists(dir):
        os.makedirs(dir)
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
    # if logger is not None and prog_args.save_dir is not None:
    #     model.load_state_dict(torch.load(prog_args.save_dir + "/" + prog_args.dataset
    #                                      + "/model.iter-" + str(logger['best_epoch'])))
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


def main():
    '''
    main
    '''
    prog_args = arg_parse()
    print(prog_args)
    random.seed(prog_args.seed)
    np.random.seed(prog_args.seed)
    torch.manual_seed(prog_args.seed)

    graph_classify_task(prog_args)


if __name__ == "__main__":
    main()
