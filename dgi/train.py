import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgi.dgi import DGI, Classifier
from dataset import Dataset
from main import get_feature_initialization
from SGC.metrics import f1
import random
from SGC.normalization import row_normalize
from utils import split_train_test

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    data = Dataset(args.data)
    labels = data.labels
    # features
    features = get_feature_initialization(args, data.graph, inplace=False)
    if args.init == "ori":
        features = np.array([features[data.idx2id[node]] for node in sorted(data.graph.nodes())])
    else:
        features = np.array([features[node] for node in sorted(data.graph.nodes())])
    features = row_normalize(features)
    features = torch.FloatTensor(features)

    train_mask, val_mask, test_mask = split_train_test(len(labels), seed=args.seed)

    in_feats = features.shape[1]
    n_classes = data.n_classes
    n_edges = data.graph.number_of_edges()

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # graph preprocess
    g = data.graph
    # add self loop
    # if args.self_loop:
    #     g.remove_edges_from(g.selfloop_edges())
    #     g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = DGLGraph(g)
    n_edges = g.number_of_edges()

    # create DGI model
    dgi = DGI(g,
              in_feats,
              args.n_hidden,
              args.n_layers,
              nn.PReLU(args.n_hidden),
              args.dropout)

    if cuda:
        dgi.cuda()

    dgi_optimizer = torch.optim.Adam(dgi.parameters(),
                                     lr=args.dgi_lr,
                                     weight_decay=args.weight_decay)

    # train deep graph infomax
    cnt_wait = 0
    best = 1e9
    best_t = 0
    dur = []
    for epoch in range(args.n_dgi_epochs):
        
        dgi.train()
        if epoch >= 3:
            t0 = time.time()

        dgi_optimizer.zero_grad()
        loss = dgi(features)
        loss.backward()
        dgi_optimizer.step()

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(dgi.state_dict(), 'dgi-best-model.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        if epoch % 20 == 0:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f}".format(epoch, np.mean(dur), loss.item()))
    print('Loading {}th epoch'.format(best_t))
    dgi.load_state_dict(torch.load('dgi-best-model.pkl'))

    # create classifier model
    classifier = Classifier(args.n_hidden, n_classes)
    if cuda:
        classifier.cuda()

    classifier_optimizer = torch.optim.Adam(classifier.parameters(),
                                            lr=args.classifier_lr,
                                            weight_decay=args.weight_decay)
    # train classifier
    embeds = dgi.encoder(features, corrupt=False)
    embeds = embeds.detach()
    dur = []
    best_val_acc = 0
    for epoch in range(args.n_classifier_epochs):
        classifier.train()
        if epoch >= 3:
            t0 = time.time()

        classifier_optimizer.zero_grad()
        preds = classifier(embeds)
        loss = F.nll_loss(preds[train_mask], labels[train_mask])
        loss.backward()
        classifier_optimizer.step()
        
        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(classifier, embeds, labels, val_mask)
        if epoch % 20 == 0:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f}".format(epoch, np.mean(dur), loss.item(), acc))
        if acc > best_val_acc:
            best_val_acc = acc 
            torch.save(classifier.state_dict(), 'dgi-best-model.pkl')
            print("== Epoch {} - Best val acc: {:.3f}".format(epoch, acc))
    classifier.load_state_dict(torch.load('dgi-best-model.pkl'))
    classifier.eval()
    with torch.no_grad():
        logits = classifier(embeds)
        logits = logits[test_mask]
        labels = labels[test_mask]
        micro, macro = f1(logits, labels)
        print('Test micro-macro: {:.3f}\t{:.3f}'.format(micro, macro))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGI')
    # register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--dgi-lr", type=float, default=1e-3,
                        help="dgi learning rate")
    parser.add_argument("--classifier-lr", type=float, default=0.2,
                        help="classifier learning rate")
    parser.add_argument("--n-dgi-epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=512,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=20,
                        help="early stop patience condition")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")

    parser.add_argument('--data', type=str)
    parser.add_argument('--init', type=str, default="ori", help="Features initialization method")
    parser.add_argument('--feature_size', type=int, default=128, help="Features dimension")
    parser.add_argument('--norm_features', action='store_true', help="norm features by standard scaler.")
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--verbose', type=int, default=1)

    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
