import torch 
import numpy as np

def sample_mask(idx, length):
    mask = np.zeros((length))
    mask[idx] = 1
    return torch.ByteTensor(mask)

def split_train_test(length, ratio=[.7, .1, .2], seed=40):
    state = np.random.get_state()
    np.random.seed(seed)
    n_train = int(length*ratio[0])
    n_val = int(length*ratio[1])
    indices = np.random.permutation(np.arange(length))
    train_mask = sample_mask(indices[:n_train], length)
    val_mask = sample_mask(indices[n_train:n_train+n_val], length)
    test_mask = sample_mask(indices[n_train+n_val:], length)
    np.random.set_state(state)
    return train_mask, val_mask, test_mask


from sklearn.metrics import f1_score
import numpy as np

def accuracy(output, labels, multiclass=False):
    if not multiclass:
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)
    else:
        # preds = torch.sigmoid(output)
        # preds[preds > 0.5] = 1
        # preds[preds <= 0.5] = 0
        # preds = preds.cpu().detach().numpy().astype(np.int32)
        probs = torch.sigmoid(output)
        top_k_list = [int(x.sum()) for x in labels]
        for i, k in enumerate(top_k_list):
            preds = probs[i].argsort()[-k:]
            probs[i,:] = 0
            probs[i,preds] = 1
        probs = probs.cpu().detach().numpy().astype(np.int32)
        labels = labels.cpu().detach().numpy().astype(np.int32)
        correct = (probs == labels).mean()
        return correct

def f1(output, labels, multiclass=False):
    if not multiclass:
        preds = output.max(1)[1]
        preds = preds.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        micro = f1_score(labels, preds, average='micro')
        macro = f1_score(labels, preds, average='macro')
        return micro, macro
    else:
        probs = torch.sigmoid(output)
        top_k_list = [int(x.sum()) for x in labels]
        for i, k in enumerate(top_k_list):
            preds = probs[i].argsort()[-k:]
            probs[i,:] = 0
            probs[i,preds] = 1
        probs = probs.cpu().detach().numpy().astype(np.int32)
        labels = labels.cpu().detach().numpy().astype(np.int32)
        micro = f1_score(labels, probs, average='micro')
        macro = f1_score(labels, probs, average='macro')
        return micro, macro

