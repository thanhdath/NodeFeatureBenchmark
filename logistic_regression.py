import time
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from SGC.metrics import f1, accuracy
import torch

class LogisticRegressionPytorch():
    def __init__(self, G, labels, epochs=100, ratio=[.7, .1, .2], lr=0.2, cuda=True):
        self.G = G
        self.labels = labels
        self.ratio = ratio
        self.epochs = epochs
        self.cuda = cuda
        self._process_features()
        self._convert_labels_to_binary()
        self._split_train_test()
        self.model = nn.Linear(self.features.shape[1], self.n_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train()

    def _process_features(self):
        features = []
        for node in self.G.nodes():
            features.append(self.G.node[node]['feature'])
        features = np.array(features)
        self.features = torch.FloatTensor(features)
    
    def _convert_labels_to_binary(self):
        labels_arr = []
        for node in self.G.nodes():
            labels_arr.append(self.labels[str(node)])
        self.binarizer = MultiLabelBinarizer(sparse_output=False)
        self.binarizer.fit(labels_arr)
        self.labels = self.binarizer.transform(labels_arr)
        self.labels = torch.LongTensor(self.labels).argmax(dim=1)
        self.n_classes = int(self.labels.max() + 1)

    def _split_train_test(self):
        n_train = int(len(self.features)*self.ratio[0])
        n_val = int(len(self.features)*self.ratio[1])
        indices = np.random.permutation(np.arange(len(self.features)))
        self.train_indices = indices[:n_train]
        self.val_indices = indices[n_train:n_val+n_train]
        self.test_indices = indices[n_train+n_val:]

    def train(self):
        train_features = self.features[self.train_indices]
        train_labels = self.labels[self.train_indices]
        val_features = self.features[self.val_indices]
        val_labels = self.labels[self.val_indices]
        test_features = self.features[self.test_indices]
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
            loss_train = F.cross_entropy(output, train_labels)
            # loss_train = self.loss_fn(output, train_labels) # multiple classes
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
            micro, macro = f1(output, test_labels)
            print('Test micro-macro: {:.3f}\t{:.3f}'.format(micro, macro))
