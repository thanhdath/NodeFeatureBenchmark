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
