import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
import numpy as np


class GCN(torch.nn.Module):
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


class TrainerTorch:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self, data):
        # data = dataset[0].to(self.device)
        model = GCN(
            num_node_features=data.x.shape[1],
            num_hidden=16,
            num_classes=int((data.y.max() + 1).item())
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        """
        Train model.
        """
        model.train()
        for epoch in range(5000):
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print('Epoch {0}: {1}'.format(epoch, loss.item()))

        """
        Evaluate model.
        """
        model.eval()
        pred = model(data).argmax(dim=1)
        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        acc = int(correct) / int(data.val_mask.sum())
        print(f'Accuracy: {acc:.4f}')

        """
        Save prediction results.
        """
        # preds = pred[data.test_mask]
        # np.savetxt('submission.txt', preds, fmt='%d')
