import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential
import torch_geometric.nn as gnn
from torch_geometric.nn import aggr


class AggGCNConv(torch.nn.Module):
    def __init__(self, data, n_layers: int, n_hidden: int):
        super().__init__()
        n_features = data.x.shape[1]
        n_classes = int((data.y.max() + 1).item())

        self.conv1 = gnn.GCNConv(n_features, n_hidden, normalize=True)  # , aggr=['mean']
        self.conv2 = gnn.GCNConv(n_hidden, n_classes, normalize=True)
        self.global_pool = aggr.SortAggregation(k=4)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return F.log_softmax(x, dim=-1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__


class GCN(torch.nn.Module):
    def __init__(self, data, conv, n_layers: int, n_hidden: int):
        super().__init__()
        n_features = data.x.shape[1]
        n_classes = int((data.y.max() + 1).item())

        self.conv1 = conv(n_features, n_hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(n_layers - 1):
            self.convs.append(conv(n_hidden, n_hidden))
        self.lin1 = Linear(n_hidden, n_hidden)
        self.lin2 = Linear(n_hidden, n_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GCNConv(GCN):
    def __init__(self, data, n_layers: int, n_hidden: int):
        super().__init__(data=data, conv=gnn.GCNConv, n_layers=n_layers, n_hidden=n_hidden)


class GraphSAGE(GCN):
    def __init__(self, data, n_layers: int, n_hidden: int):
        super().__init__(data=data, conv=gnn.SAGEConv, n_layers=n_layers, n_hidden=n_hidden)


class GIN(torch.nn.Module):
    def __init__(self, data, n_layers: int, n_hidden: int):
        super().__init__()
        n_features = data.x.shape[1]
        n_classes = int((data.y.max() + 1).item())

        self.conv1 = gnn.GINConv(
            Sequential(
                Linear(n_features, n_hidden),
                ReLU(),
                BN(n_hidden),
                Linear(n_hidden, n_hidden),
                ReLU(),
                BN(n_hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(n_layers - 1):
            self.convs.append(
                gnn.GINConv(
                    Sequential(
                        Linear(n_hidden, n_hidden),
                        ReLU(),
                        BN(n_hidden),
                        Linear(n_hidden, n_hidden),
                        ReLU(),
                        BN(n_hidden),
                    ), train_eps=True))
        self.lin1 = Linear(n_hidden, n_hidden)
        self.lin2 = Linear(n_hidden, n_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
