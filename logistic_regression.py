import time
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from SGC.metrics import f1
import torch

class LogisticRegressionPytorch():
    def __init__(self, G, labels, epochs=10, ratio=[.8, .2], lr=0.1, cuda=True):
        self.G = G
        self.labels = labels
        self.ratio = ratio
        self.epochs = epochs
        self.cuda = cuda
        self._process_features()
        self._convert_labels_to_binary()
        self.split_train_test()
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

    def split_train_test(self):
        indices = np.random.permutation(np.arange(len(self.features)))
        n_train = int(len(self.features)*self.ratio[0])
        self.train_features = self.features[indices[:n_train]]
        self.train_labels = self.labels[indices[:n_train]]
        self.test_features = self.features[indices[n_train:]]
        self.test_labels = self.labels[indices[n_train:]]

    def train(self):
        if self.cuda:
            self.train_labels = self.train_labels.cuda()
            self.model.cuda()
            self.train_features = self.train_features.cuda()
        t = time.time()
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.train_features)
            loss_train = F.cross_entropy(output, self.train_labels)
            loss_train.backward()
            self.optimizer.step()
            if epoch%20 == 0:
                print("Epoch {} - train loss: {:.3f}".format(epoch, loss_train.item()))
        train_time = time.time()-t
        print("Train time: {:.3f}s".format(train_time))

        if self.cuda:
            self.train_features = self.train_features.cpu()
            self.train_labels = self.train_labels.cpu()
            self.test_labels = self.test_labels.cuda()
            self.test_features = self.test_features.cuda()
        with torch.no_grad():
            self.model.eval()
            output = self.model(self.test_features)
            micro, macro = f1(output, self.test_labels)
            print('Test micro-macro: {:.3f}\t{:.3f}'.format(micro, macro))
