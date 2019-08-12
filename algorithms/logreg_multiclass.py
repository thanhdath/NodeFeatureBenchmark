import time
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
from utils import accuracy, f1_multiple_classes
import torch

class LogisticRegressionPytorch():
    def __init__(self, embs, labels, train_mask, val_mask, test_mask, 
        epochs=200, lr=0.2, weight_decay=5e-6, bias=True, cuda=True):
        """
        embs: np array, embedding of nodes
        labels: np array, multiple class, binary encoder, eg: [[0,1,0], [1,0,1], [1,1,0]]
        """
        self.embs = embs
        self.labels = torch.FloatTensor(labels)
        
        self.n_classes = int(self.labels.max() + 1)
        self.epochs = epochs
        self.cuda = cuda
        self.train_indices = np.argwhere(train_mask).flatten()
        self.val_indices = np.argwhere(val_mask).flatten()
        self.test_indices = np.argwhere(test_mask).flatten()

        self.model = nn.Linear(self.embs.shape[1], self.n_classes, bias=bias)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train()

    def train(self):
        train_features = self.embs[self.train_indices]
        train_labels = self.labels[self.train_indices]
        val_features = self.embs[self.val_indices]
        val_labels = self.labels[self.val_indices]
        test_features = self.embs[self.test_indices]
        test_labels = self.labels[self.test_indices]

        stime = time.time()
        if self.cuda:
            train_labels = train_labels.cuda()
            train_features = train_features.cuda()
            val_labels = val_labels.cuda()
            val_features = val_features.cuda()
            self.model.cuda()
        best_val_acc = 0
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(train_features)
            loss_train = self.loss_fn(output, train_labels) # multiple classes
            loss_train.backward()
            self.optimizer.step()
            if epoch % 20 == 0:
                print('Epoch {} - loss {}'.format(epoch, loss_train.item()))
            with torch.no_grad():
                self.model.eval()
                output = self.model(val_features)
                acc = accuracy(output, val_labels)
                if acc > best_val_acc:
                    best_val_acc = acc
                    torch.save(self.model.state_dict(), 'logistic-best-model.pkl')
                    print('== Epoch {} - Best val acc: {:.3f}'.format(epoch, acc.item()))
        train_time = time.time() - stime
        print('Train time: {:.3f}'.format(train_time))
        self.model.load_state_dict(torch.load('logistic-best-model.pkl'))
        if self.cuda:
            train_labels = train_labels.cpu()
            train_features = train_features.cpu()
            val_labels = val_labels.cpu()
            val_features = val_features.cpu()
            test_labels = test_labels.cuda()
            test_features = test_features.cuda()

        with torch.no_grad():
            self.model.eval()
            output = self.model(test_features)
            micro, macro = f1_multiple_classes(output, test_labels)
            print('Test micro-macro: {:.3f}\t{:.3f}'.format(micro, macro))
            