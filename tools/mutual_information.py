"""
mkdir logs
mkdir logs/mi
for data in cora citeseer pubmed NELL
do 
    for init in hope deepwalk
    do
        python -u tools/mutual_information.py $data $init > logs/mi/$data-$init
    done
done  
"""


import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from tqdm import tqdm
import matplotlib.pyplot as plt

def dict2arr(dictt):
    """
    Note: always sort graph nodes
    """
    dict_arr = np.array([dictt[int(x)] for x in range(len(dictt))])
    return dict_arr
# load embedding and node features
data = sys.argv[1] # cora
init = sys.argv[2] # deepwalk
feature_file = f"/home/datht/nodefeature/data/{data}/features.npz"
embed_unsup_file = f"/home/datht/nodefeature/feats/{data}-{init}-seed40-dim128.npz"
figure_file = f"figures/{data}-original-vs-{init}.png"
features = np.load(feature_file, allow_pickle=True)["features"][()]
embeds = np.load(embed_unsup_file, allow_pickle=True)["features"][()]

x = dict2arr(features)
y = dict2arr(embeds)
print(x.shape, y.shape)

class Mine(nn.Module):
    def __init__(self, x_size=2, y_size=2, hidden_size=100):
        super().__init__()
        self.fcx = nn.Linear(x_size, hidden_size)
        self.fcy = nn.Linear(y_size, hidden_size)
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1)
        )
        
        nn.init.normal_(self.fcx.weight, std=0.02)
        nn.init.constant_(self.fcx.bias, 0)
        nn.init.normal_(self.fcy.weight, std=0.02)
        nn.init.constant_(self.fcy.bias, 0)
        nn.init.normal_(self.transform[0].weight, std=0.02)
        nn.init.constant_(self.transform[0].bias, 0)
        nn.init.normal_(self.transform[-1].weight, std=0.02)
        nn.init.constant_(self.transform[-1].bias, 0)
        
    def forward(self, x, y):
        x = F.elu(self.fcx(x))
        y = F.elu(self.fcy(y))
        output = self.transform(x+y)
        return output

def mutual_information(joint_x, joint_y, marginal_x, marginal_y, mine_net):
    t = mine_net(joint_x, joint_y)
    et = torch.exp(mine_net(marginal_x, marginal_y))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et

def learn_mine(batch, mine_net, mine_net_optim,  ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint , marginal = batch
    joint_x, joint_y = joint
    marginal_x, marginal_y = marginal
    joint_x = torch.autograd.Variable(torch.FloatTensor(joint_x)).cuda()
    joint_y = torch.autograd.Variable(torch.FloatTensor(joint_y)).cuda()
    marginal_x = torch.autograd.Variable(torch.FloatTensor(marginal_x)).cuda()
    marginal_y = torch.autograd.Variable(torch.FloatTensor(marginal_y)).cuda()
    
    mi_lb , t, et = mutual_information(joint_x, joint_y, marginal_x, marginal_y, mine_net)
    ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
    
    # unbiasing use moving average
    loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
    # use biased estimator
#     loss = - mi_lb
    
    mine_net_optim.zero_grad()
    autograd.backward(loss)
    mine_net_optim.step()
    return mi_lb, ma_et

def sample_batch(x, y, batch_size=100, sample_mode='joint'):
    if sample_mode == 'joint':
        index = np.random.choice(range(x.shape[0]), size=batch_size, replace=False)
        batch = x[index], y[index]
    else:
        joint_index = np.random.choice(range(x.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(x.shape[0]), size=batch_size, replace=False)
        batch = x[joint_index], y[marginal_index]
    return batch

def train(data, mine_net,mine_net_optim, batch_size=100, iter_num=int(5e+3), log_freq=int(1e+3)):
    # data is x or y
    result = list()
    ma_et = 1.
    for i in range(iter_num):
        if i%1000 == 0: print("Iter", i)
        batch_joint = sample_batch(x, y, batch_size=batch_size)
        batch_marginal = sample_batch(x, y, batch_size=batch_size,sample_mode='marginal')

        mi_lb, ma_et = learn_mine((batch_joint, batch_marginal), mine_net, mine_net_optim, ma_et)
        result.append(mi_lb.detach().cpu().numpy())
        if (i+1)%(log_freq)==0:
            print("MI: ", result[-1])
    return result

def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]

mine_net_indep = Mine(x_size=x.shape[1], y_size=y.shape[1]).cuda()
mine_net_optim_indep = optim.Adam(mine_net_indep.parameters(), lr=0.01)
result_indep = train(x,mine_net_indep,mine_net_optim_indep, batch_size=512, iter_num = 20000)

figure_dir = figure_file.split("/")[0]

if not os.path.isdir(figure_dir):
    os.makedirs(figure_dir)

result_indep = ma(result_indep)
print("Smoothed result indep: ", result_indep[-1])
plt.figure()
plt.plot(result_indep, linewidth=.4)
plt.savefig(figure_file)
