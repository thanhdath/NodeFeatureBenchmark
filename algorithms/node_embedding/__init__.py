from .sgc.sgc import SGC

__all__ = ['SGC']

import torch
class Nope():
    def __init__(self, features):
        self.features = features
    def train(self):
        return torch.FloatTensor(self.features)
    
