import torch
import torch.nn as nn
import numpy as np
import time
from torch.autograd import Variable

"""
Set of modules for aggregating embeddings of neighbors.
"""

class Aggregator(nn.Module):
    def __init__(self, input_dim, output_dim, activation, concat=True, dropout=0.0,
                 fc_x_weight=None, fc_neib_weight=None):

        super(Aggregator, self).__init__()

        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)
        if fc_x_weight is not None:
            self.fc_x.weight = nn.Parameter(torch.FloatTensor(fc_x_weight))
            self.fc_x.weight.requires_grad = True
        if fc_neib_weight is not None:
            self.fc_neib.weight = nn.Parameter(torch.FloatTensor(fc_neib_weight))
            self.fc_neib.weight.requires_grad = True

        self.activation = activation
        self.concat = concat
        self.dropout = nn.Dropout(p=dropout)

        if self.concat:
            self.output_dim = output_dim * 2
        else:
            self.output_dim = output_dim

class MeanAggregator(Aggregator):

    def __init__(self, input_dim, output_dim, activation, concat=True, dropout=0.0,
                 fc_x_weight=None, fc_neib_weight=None):
        super(MeanAggregator, self).__init__(input_dim, output_dim, activation, concat, dropout,
                                             fc_x_weight=fc_x_weight, fc_neib_weight=fc_neib_weight)

    def forward(self, x, neibs): 
        neibs = self.dropout(neibs)
        x = self.dropout(x)

        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib = agg_neib.mean(dim=1)  

        if self.concat:
            out = torch.cat([self.fc_x(x), self.fc_neib(agg_neib)], dim=1)
        else:
            out = self.fc_x(x) + self.fc_neib(agg_neib) 

        if self.activation:
            out = self.activation(out)

        return out
       
class MaxPoolAggregator(Aggregator):
    def __init__(self, input_dim, output_dim, activation, concat=True, dropout=0.0, hidden_dim=256):
        super(MaxPoolAggregator, self).__init__(input_dim, output_dim, activation, concat, dropout)
        
        self.mlp = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU()
        ])
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(hidden_dim, output_dim, bias=False)                
            
    def forward(self, x, neibs):
        x = self.dropout(x)
        neibs = self.dropout(neibs)        
        h_neibs = self.mlp(neibs)
        agg_neib = h_neibs.view(x.size(0), -1, h_neibs.size(1))
        agg_neib = agg_neib.max(dim=1)[0]
        
        if self.concat:
            out = torch.cat([self.fc_x(x), self.fc_neib(agg_neib)], dim=1)
        else:
            out = self.fc_x(x) + self.fc_neib(agg_neib) 

        if self.activation:
            out = self.activation(out)
        
        return out

class MeanPoolAggregator(Aggregator):
    def __init__(self, input_dim, output_dim, activation, concat=True, dropout=0.0, hidden_dim=256):
        super(MeanPoolAggregator, self).__init__(input_dim, output_dim, activation, concat, dropout)
        
        self.mlp = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU()
        ])
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(hidden_dim, output_dim, bias=False)                
            
    def forward(self, x, neibs):
        x = self.dropout(x)
        neibs = self.dropout(neibs)        
        h_neibs = self.mlp(neibs)
        agg_neib = h_neibs.view(x.size(0), -1, h_neibs.size(1))
        agg_neib = agg_neib.mean(dim=1)
        
        if self.concat:
            out = torch.cat([self.fc_x(x), self.fc_neib(agg_neib)], dim=1)
        else:
            out = self.fc_x(x) + self.fc_neib(agg_neib) 

        if self.activation:
            out = self.activation(out)
        
        return out

# class MaxPoolAggregator(PoolAggregator):
#     def __init__(self, input_dim, output_dim, activation, hidden_dim=256, concat=True, dropout=0.0):
#         super(MaxPoolAggregator, self).__init__(**{
#             "input_dim"     : input_dim,
#             "output_dim"    : output_dim,
#             "pool_fn"       : lambda x: x.max(dim=1)[0],
#             "activation"    : activation,
#             "hidden_dim"    : hidden_dim,
#             "concat"        : concat,
#             "dropout"       : dropout
#         })


# class MeanPoolAggregator(PoolAggregator):
#     def __init__(self, input_dim, output_dim, activation, hidden_dim=256, concat=True, dropout=0.0):
#         super(MeanPoolAggregator, self).__init__(**{
#             "input_dim"     : input_dim,
#             "output_dim"    : output_dim,
#             "pool_fn"       : lambda x: x.mean(dim=1),
#             "activation"    : activation,
#             "hidden_dim"    : hidden_dim,
#             "concat"        : concat,
#             "dropout"       : dropout          
#         })

class LSTMAggregator(Aggregator):
    def __init__(self, input_dim, output_dim, activation, concat=True, dropout=0.0, bidirectional = False, hidden_dim=256):    
        super(LSTMAggregator, self).__init__(input_dim, output_dim, activation, concat, dropout)
        self.lstm = nn.LSTM(input_dim, hidden_dim // (1 + bidirectional), bidirectional=bidirectional, batch_first=True)
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(hidden_dim, output_dim, bias=False)        
                
    def forward(self, x, neibs):
        
        x = self.dropout(x)
        neibs = self.dropout(neibs)                
        
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib, _ = self.lstm(agg_neib)
        agg_neib = agg_neib[:,-1,:] # !! Taking final state
        
        if self.concat:
            out = torch.cat([self.fc_x(x), self.fc_neib(agg_neib)], dim=1)
        else:
            out = self.fc_x(x) + self.fc_neib(agg_neib) 

        if self.activation:
            out = self.activation(out)
        
        return out

