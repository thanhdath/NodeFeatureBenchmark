from .sgc.sgc import SGC
from .dgi.train import DGIAPI

__all__ = ['SGC', 'DGIAPI']

import torch
class Nope():
    def __init__(self, features):
        self.features = features
    def train(self):
        return torch.FloatTensor(self.features)
    
