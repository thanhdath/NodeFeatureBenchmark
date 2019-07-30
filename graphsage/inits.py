import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import numpy as np

#Initializer for tensor
#Output: a parameter with value initialized
class Initializer:

    @staticmethod
    def uniform(shape, scale=0.05):
        """Uniform init."""
        """Fills the input Tensor with values drawn from the uniform distribution U(a,b)."""
        w = nn.Parameter(torch.FloatTensor(torch.Size(shape)))
        init.uniform(w, a=-scale, b=scale)
        return w

    @staticmethod
    def xavier_uniform(shape):
        """Glorot & Bengio (AISTATS 2010) init."""
        """The resulting tensor will have values sampled from U(−a,a) where a = np.sqrt(6.0/(shape[0]+shape[1]))"""
        w = nn.Parameter(torch.FloatTensor(torch.Size(shape)))
        nn.init.xavier_uniform(w, gain=nn.init.calculate_gain('relu'))
        return w

    @staticmethod
    def zeros(shape):
        """All zeros."""
        w = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        return w

    @staticmethod
    def ones(shape):
        """All ones."""
        w = nn.Parameter(torch.ones(shape, dtype=torch.float32))
        return w
