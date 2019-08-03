from sklearn.metrics import f1_score
import numpy as np

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro, macro

def f1_multiple_classes(preds, labels):
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro, macro

def acc_multiple_classes(preds, labels):
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    preds = preds.cpu().detach().numpy().astype(np.int32)
    labels = labels.cpu().detach().numpy().astype(np.int32)
    acc = (preds == labels).sum() / len(labels.flatten())
    return acc
