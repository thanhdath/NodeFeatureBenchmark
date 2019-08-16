import time
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
from utils import f1, accuracy
import torch

class LogisticRegressionInductive():
    def __init__(self, train_embs, val_embs, test_embs, 
        train_labels, val_labels, test_labels,
        epochs=200, lr=0.2, weight_decay=5e-6, bias=True, cuda=True, multiclass=False):
        """
        embs: np array, embedding of nodes
        labels: LongTensor for single-label, FloatTensor for multilabel 
        """
        if multiclass:
            print("Train logistic regression for multiclass")
        self.multiclass = multiclass
        self.train_embs = train_embs
        self.val_embs = val_embs
        self.test_embs = test_embs

        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels

        if not self.multiclass:
            self.n_classes = int(self.train_labels.max() + 1)
        else:
            self.n_classes = self.train_labels.shape[1]
            self.loss_multiclass = nn.BCEWithLogitsLoss()

        self.epochs = epochs
        self.cuda = cuda

        self.model = nn.Linear(self.train_embs.shape[1], self.n_classes, bias=bias)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.train()

    def train(self):
        train_embs, val_embs, test_embs = self.train_embs, self.val_embs, self.test_embs
        train_labels, val_labels, test_labels = self.train_labels, self.val_labels, self.test_labels

        stime = time.time()
        if self.cuda:
            train_labels = train_labels.cuda()
            train_embs = train_embs.cuda()
            val_labels = val_labels.cuda()
            val_embs = val_embs.cuda()
            self.model.cuda()
        best_val_acc = 0
        npt = 0
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(train_embs)
            if self.multiclass:
                loss_train = self.loss_multiclass(output, train_labels) # multiple classes
            else:
                loss_train = F.cross_entropy(output, train_labels)
            loss_train.backward()
            self.optimizer.step()
            if epoch % 20 == 0:
                print('Epoch {} - loss {}'.format(epoch, loss_train.item()))
                with torch.no_grad():
                    self.model.eval()
                    output = self.model(val_embs)
                    acc = accuracy(output, val_labels, multiclass=self.multiclass)
                    if acc > best_val_acc:
                        best_val_acc = acc
                        torch.save(self.model.state_dict(), 'logistic-best-model.pkl')
                        print('== Epoch {} - Best val acc: {:.3f}'.format(epoch, acc.item()))
                        npt = 0
                    else:
                        npt += 1
                    if npt > 3: 
                        print("Early stopping")
                        break
        train_time = time.time() - stime
        print('Train time: {:.3f}'.format(train_time))
        self.model.load_state_dict(torch.load('logistic-best-model.pkl'))
        if self.cuda:
            train_labels = train_labels.cpu()
            train_embs = train_embs.cpu()
            val_labels = val_labels.cpu()
            val_embs = val_embs.cpu()
            test_labels = test_labels.cuda()
            test_embs = test_embs.cuda()

        with torch.no_grad():
            self.model.eval()
            output = self.model(test_embs)
            micro, macro = f1(output, test_labels, multiclass=self.multiclass)
            print('Test micro-macro: {:.3f}\t{:.3f}'.format(micro, macro))
            