import torch.nn as nn
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SGConv, SAGEConv, ChebConv, GINConv  # noqa
import numpy as np
import random

# # R3
# link_ratio=0.015
# mean_center=0.25

# R2
# link_ratio=0.016
# mean_center=0.3

# R1
link_ratio = 0.015
mean_center = 0.2


def assortative_rate(nx_g, label):
    '''
    :param nx_g: networkx graphs
    :param label: labels of each node
    :return: assortative rate
    '''
    graph_rate = []
    label_dict = {}
    for ix, y in enumerate(label.numpy()):
        label_dict[ix] = y

    for vi in nx_g.nodes():
        node_rate = []
        for vj in nx_g.neighbors(vi):
            if label_dict[vi] == label_dict[vj]:
                node_rate.append(1)
            else:
                node_rate.append(0)
        graph_rate.append(np.mean(node_rate))
    return np.mean(graph_rate)


def build_random_graph(num_nodes=2000, ratio=link_ratio):
    g_nx = nx.gnp_random_graph(num_nodes, ratio, seed=1234)
    print(g_nx.number_of_edges())
    g_nx.add_nodes_from(list(range(num_nodes)))
    edge_index = torch.from_numpy(np.array(g_nx.edges).T).long()
    return edge_index, g_nx


def shuffle(x, y):
    examples = np.concatenate([x, y], axis=1)
    np.random.shuffle(examples)
    return examples[:, :-1], examples[:, -1]


def feature_by_gaussian(mean, num_nodes, num_dim):
    center = np.ones([num_dim]) * mean * mean_center
    cov = np.identity(num_dim, dtype=float) * 0.10
    feature = np.random.multivariate_normal(center, cov, num_nodes)
    return feature


def generate_feature_label(num_nodes, num_class):
    x = []
    y = []
    for i in range(num_class):
        x.append(feature_by_gaussian(i, num_nodes // num_class, num_dim))
        y.append(np.ones([num_nodes // num_class, 1]) * i)
    x = np.concatenate(x, axis=0)  # (N,F)
    y = np.concatenate(y, axis=0)  # (N,1)
    x, y = shuffle(x, y)
    return torch.from_numpy(x).float(), torch.from_numpy(y).long()


def generate_mask(rate=[1, 2, 7]):
    train_mask = torch.zeros(num_nodes)
    val_mask = torch.zeros(num_nodes)
    test_mask = torch.zeros(num_nodes)
    train_num, val_num, test_num = (num_nodes * np.array(rate) * 0.1).astype(int)
    for i in range(train_num):
        train_mask[i] = 1
    for j in range(train_num, train_num + val_num):
        val_mask[j] = 1
    for k in range(train_num + val_num, num_nodes):
        test_mask[k] = 1
    mask = {'train_mask': train_mask.long(), 'val_mask': val_mask.long(), 'test_mask': test_mask.long()}
    return mask


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_dim, 16)
        self.conv2 = GCNConv(16, num_class)

    def forward(self, x, edge_index=None, train_mode=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class SGC(torch.nn.Module):
    def __init__(self):
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_dim, num_class, K=2, cached=True)

    def forward(self, x, edge_index=None, train_mode=None):
        x = F.dropout(x, training=self.training)
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


class ChebNet(torch.nn.Module):
    def __init__(self):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(num_dim, 16, K=2)
        self.conv2 = ChebConv(16, num_class, K=2)

    def forward(self, x, edge_index=None, train_mode=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class SAGE(torch.nn.Module):
    def __init__(self):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(num_dim, 16)
        self.conv2 = SAGEConv(16, num_class)

    def forward(self, x, edge_index=None, train_mode=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_dim, 16, heads=8, concat=True)
        self.conv2 = GATConv(16 * 8, num_class, heads=1, concat=False)

    def forward(self, x, edge_index=None, train_mode=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GIN(torch.nn.Module):
    # Test results:
    # R1: test_acc:0.2714
    # R2: test_acc:0.2743
    # R3: test_acc:0.2686
    def __init__(self):
        super(GIN, self).__init__()
        hidden = 16
        self.conv1 = GINConv(nn.Linear(num_dim, hidden))
        self.conv2 = GINConv(nn.Linear(hidden, num_class))

    def forward(self, x, edge_index, train_mode=False):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class SimGCN(torch.nn.Module):
    '''
    test_acc:
    R1: test_acc:74.57
    R1: test_acc:86.57
    R1: test_acc:81.43
    '''

    def __init__(self):
        super(SimGCN, self).__init__()
        self.hidden_dim = 16
        self.act = nn.LeakyReLU()
        self.lin1 = nn.Linear(num_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin3 = nn.Linear(self.hidden_dim, num_class)

    def simAGG(self, x):
        adj = torch.matmul(x, x.transpose(0, 1))
        zero_vec = -9e15 * torch.ones_like(adj)
        a = torch.where(adj > 0, adj, zero_vec)
        sim1 = F.softmax(a, dim=1)
        x = torch.matmul(sim1, x)
        return x

    def forward(self, x, edge_index=None, train_mode=False):
        x = F.dropout(x, training=self.training)
        x = self.lin1(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, training=self.training)
        x = self.lin3(x)
        x = self.simAGG(x)
        return F.log_softmax(x, dim=1)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden_dim = 16
        self.lin1 = nn.Linear(num_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin3 = nn.Linear(self.hidden_dim, num_class)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index=None, train_mode=False):
        x = F.dropout(x, training=self.training)
        x = self.lin1(x)
        x = self.act(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = self.act(x)
        x = F.dropout(x, training=self.training)
        x = self.lin3(x)

        return F.log_softmax(x, dim=1)


EPS = 1e-15


def lp_loss(z, pos_edges, neg_edges):
    pos_value = (z[pos_edges[0]] * z[pos_edges[1]]).sum(dim=1)
    neg_value = (z[neg_edges[0]] * z[neg_edges[1]]).sum(dim=1)
    pos_loss = -torch.log(pos_value.sigmoid() + EPS).mean()
    neg_loss = -torch.log(1 - neg_value.sigmoid() + EPS).mean()
    return pos_loss + neg_loss


def train(mask):
    model.train()
    optimizer.zero_grad()
    logit = model(x, edge_index, train_mode=False)
    loss = F.nll_loss(logit[mask == 1], y[mask == 1])
    pred = logit[mask == 1].max(1)[1]
    acc = pred.eq(y[mask == 1]).sum().item() / mask.sum().item()
    loss.backward()
    optimizer.step()
    return loss.item(), acc


@torch.no_grad()
def evaluation(mask):
    model.eval()
    logits = model(x, edge_index)
    pred = logits[mask == 1].max(1)[1]
    acc = pred.eq(y[mask == 1]).sum().item() / mask.sum().item()
    return acc


if __name__ == '__main__':
    # Experiments for RG
    print("generating random graph")
    setup_seed(100)
    num_class = 4
    num_dim = 10
    num_nodes = 500
    print('>> build random graph ...')
    edge_index, nx_g = build_random_graph(num_nodes)
    print('>> generate features and label ...')
    x, y = generate_feature_label(num_nodes, num_class)
    # print('>> assortative rate: {:.4f}'.format(assortative_rate(nx_g, y)))
    print('>> generate masks')
    mask = generate_mask()
    print('>> training ...')
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    # model, data, y = GCN().to(device), x.to(device), y.to(device)
    # model, data, y = GAT().to(device), x.to(device), y.to(device)
    # model, data, y = SGC().to(device), x.to(device), y.to(device)
    # model, data, y = ChebNet().to(device), x.to(device), y.to(device)
    # model, data, y = SAGE().to(device), x.to(device), y.to(device)
    # model, data, y = MLP().to(device), x.to(device), y.to(device)
    # model, data, y = GIN().to(device), x.to(device), y.to(device)

    model, data, y = SimGCN().to(device), x.to(device), y.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = test_acc = 0
    for epoch in range(1, 401):
        train_loss, train_acc = train(mask['train_mask'])
        val_acc = evaluation(mask['val_mask'])
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            test_acc = evaluation(mask['test_mask'])
        print(
            'Epoch:{:3d} train_loss:{:.4f}, train_acc:{:.4f}, val_acc:{:.4f} best_val_acc:{:.4f} test_acc:{:.4f}'.format(
                epoch, train_loss, train_acc, val_acc, best_val_acc, test_acc))
