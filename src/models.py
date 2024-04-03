import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, JumpingKnowledge, global_mean_pool


class GCN_2L(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self, data, n_layers: int, n_hidden: int):
        super().__init__()
        n_features = data.x.shape[1]
        n_classes = int((data.y.max() + 1).item())

        self.conv1 = GCNConv(n_features, n_hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(n_layers - 1):
            self.convs.append(GCNConv(n_hidden, n_hidden))
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
        # x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
