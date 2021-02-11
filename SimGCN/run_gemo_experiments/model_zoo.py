import torch
from torch_geometric.nn import GCNConv, SGConv, ChebConv, GATConv, SAGEConv, GINConv
import torch.nn.functional as F
import torch.nn  as nn


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, g, x):
        edge_index = torch.cat([g.edges()[0].unsqueeze(0), g.edges()[1].unsqueeze(0)]).cuda()
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class SGC(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SGC, self).__init__()
        self.conv1 = SGConv(in_dim, out_dim, K=2, cached=True)

    def forward(self, g, x):
        edge_index = torch.cat([g.edges()[0].unsqueeze(0), g.edges()[1].unsqueeze(0)]).cuda()
        x = F.dropout(x, training=self.training)
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


class ChebNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_dim, hidden_dim, K=2)
        self.conv2 = ChebConv(hidden_dim, out_dim, K=2)

    def forward(self, g, x):
        edge_index = torch.cat([g.edges()[0].unsqueeze(0), g.edges()[1].unsqueeze(0)]).cuda()
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class SAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, g, x):
        edge_index = torch.cat([g.edges()[0].unsqueeze(0), g.edges()[1].unsqueeze(0)]).cuda()
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=8, concat=True)
        self.conv2 = GATConv(hidden_dim * 8, out_dim, heads=1, concat=False)

    def forward(self, g, x):
        edge_index = torch.cat([g.edges()[0].unsqueeze(0), g.edges()[1].unsqueeze(0)]).cuda()
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GIN(torch.nn.Module):
    # Test results:

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GIN, self).__init__()
        self.conv1 = GINConv(nn.Linear(in_dim, hidden_dim))
        self.conv2 = GINConv(nn.Linear(hidden_dim, out_dim))
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, g, x):
        edge_index = torch.cat([g.edges()[0].unsqueeze(0), g.edges()[1].unsqueeze(0)]).cuda()
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class SimGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SimGCN, self).__init__()
        self.act = nn.LeakyReLU()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

        self.conv1 = GCNConv(out_dim, out_dim)
        self.conv2 = GCNConv(out_dim, out_dim)

        self.alpha = torch.nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.Tensor([0.9]), requires_grad=True)

    def simAGG(self, x):
        adj = torch.matmul(x, x.transpose(0, 1))
        zero_vec = -9e15 * torch.ones_like(adj)
        a = torch.where(adj > 0, adj, zero_vec)
        sim1 = F.softmax(a, dim=1)
        x = torch.matmul(sim1, x)
        return x

    def forward(self, g, x, epoch=-1):
        x = F.dropout(x, training=self.training)
        x = self.lin1(x)
        x = self.act(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = self.alpha * self.simAGG(x) + self.beta * x
        return F.log_softmax(x, dim=1)


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.LeakyReLU()

    def forward(self, g, x):
        x = F.dropout(x, training=self.training)
        x1 = self.lin1(x)
        x2 = self.act(x1)
        x1 = F.dropout(x2, training=self.training)
        x2 = self.lin2(x1)
        return F.log_softmax(x2, dim=1)
