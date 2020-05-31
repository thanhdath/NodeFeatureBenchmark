import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import f1_score
from torch_geometric.nn import DenseSAGEConv



class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 normalize=False,
                 add_loop=False,
                 lin=True):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        # self.conv1 = SAGESum(in_channels, hidden_channels, normalize)
        # self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        # self.conv2 = SAGESum(hidden_channels, hidden_channels, normalize)
        # self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        # self.conv3 = SAGESum(hidden_channels, out_channels, normalize)
        # self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        num_nodes, in_channels = x.size()
        x = x.view(1, num_nodes, in_channels)
        adj = adj.view(1, adj.shape[0], adj.shape[1])

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask, self.add_loop)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask, self.add_loop)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask, self.add_loop)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x.view(num_nodes, -1)

class Net(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, n_classes):
        super(Net, self).__init__()
        self.transform = torch.nn.Linear(feature_dim, hidden_dim)
        self.gnn = GNN(hidden_dim, hidden_dim*2, n_classes, lin=True)

    def forward(self, x, edge_index):
        # A = torch.zeros((len(x), len(x))).to(device)
        # src, trg = edge_index
        # A[src, trg] = 1

        x = self.transform(x)
        # x = F.normalize(x, dim=1)
        Aabs = x.mm(x.t())
        # linkpred_loss = torch.norm(A - Aabs, p=2) / len(Aabs)

        # S = torch.softmax(x, dim=1)
        # Aabs = S.t().mm(A).mm(S)
        # Xabs = S.t().mm(X)
        # edge_index = Aabs.nonzero().t()
        x = self.gnn(x, Aabs)
        return x, Aabs



path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')
# val_dataset = PPI(path, split='val')
# test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X, edge_index = None, None
Y = None 
for i, data in enumerate(train_loader):
    if i > 0: break
    X = data.x.to(device)
    edge_index = data.edge_index.to(device)
    Y = data.y.to(device)

model = Net(X.shape[1], X.shape[1]*2, train_dataset.num_classes).to(device)
print(model)
loss_op = torch.nn.BCEWithLogitsLoss()
loss2 = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def test():
    model.eval()

    ys, preds = [], []
    # for data in loader:
    ys.append(Y)
    with torch.no_grad():
        out, _ = model(X, edge_index)
    preds.append((out > 0).float())

    y, pred = torch.cat(ys, dim=0).cpu().numpy(), torch.cat(preds, dim=0).cpu().numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

def train():
    model.train()
    # for data in train_loader:
    optimizer.zero_grad()
    out, Aabs = model(X, edge_index)
    loss = loss_op(out, Y)

    A = torch.zeros((len(X), len(X))).to(device)
    src, trg = edge_index
    A[src, trg] = 1
    linkpred_loss = loss2(A, Aabs)

    loss += linkpred_loss

    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(1, 301):
    loss = train()
    train_f1 = test()
    # test_f1 = test(test_loader)
    print('Epoch: {:02d}, Loss: {:.4f}, Train: {:.4f}'.format(
        epoch, loss, train_f1))







