from .sgc.sgc import SGC

__all__ = ['SGC']

class Nope():
    def __init__(self, features):
        self.features = features
    def train(self):
        return self.features
