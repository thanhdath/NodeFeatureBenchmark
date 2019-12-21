from .sgc.sgc import SGC
from .dgi.train import DGIAPI
from .graphsage.graphsage import GraphsageAPI
from .graphsage.graphsage_inductive import GraphsageInductive
from .gat.train import GATAPI
import torch

class Nope():
    def __init__(self, features):
        self.features = features
    def train(self):
        return torch.FloatTensor(self.features)
    
__all__ = ['SGC', 'DGIAPI', 'GraphsageAPI', 'Nope', 'GATAPI']
