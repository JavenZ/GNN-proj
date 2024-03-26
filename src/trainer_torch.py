import json
import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
import numpy as np
import logging


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

    def run(self, run_id, train_data, lr=0.01, weight_decay=5e-4, n_hidden=16, n_epochs=5000, lr_decay=0.8, lr_patience=50, epoch_patience=500):
        # graph convolutional network model
        model = GCN(
            num_node_features=train_data.x.shape[1],
            num_hidden=n_hidden,
            num_classes=int((train_data.y.max() + 1).item())
        ).to(self.device)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer = torch.optim.Adam([
        #         {'params': model.conv1.parameters(), 'weight_decay': weight_decay},
        #         {'params': model.conv2.parameters(), 'weight_decay': weight_decay}
        #     ], lr=lr
        # )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=lr_decay,
            patience=lr_patience
        )

        # logging
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fileHandler = logging.FileHandler(f"logs/{run_id}.log")
        logger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        logger.addHandler(consoleHandler)

        """
        Train Model.
        """
        vlss_mn = np.inf
        vacc_mx = 0.0
        state_dict_early_model = None
        curr_step = 0

        dur = []
        for epoch in range(n_epochs):
            t0 = time.time()

            # TRAIN
            model.train()
            train_logits = model(train_data)
            train_logp = F.log_softmax(train_logits, 1)
            train_loss = F.nll_loss(train_logp[train_data.train_mask], train_data.y[train_data.train_mask])
            train_pred = train_logp.argmax(dim=1)
            train_acc = torch.eq(train_pred[train_data.train_mask], train_data.y[train_data.train_mask]).float().mean().item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # VALIDATE
            model.eval()
            with torch.no_grad():
                val_logits = model(train_data)
                val_logp = F.log_softmax(val_logits, 1)
                val_loss = F.nll_loss(val_logp[train_data.val_mask], train_data.y[train_data.val_mask]).item()
                val_pred = val_logp.argmax(dim=1)
                val_acc = torch.eq(val_pred[train_data.val_mask], train_data.y[train_data.val_mask]).float().mean().item()

            lr_scheduler.step(val_loss)
            dur.append(time.time() - t0)
            logger.info(
                "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".format(
                    epoch, train_loss.item(), train_acc, val_loss, val_acc, sum(dur) / len(dur)))

            # Adapted from https://github.com/PetarV-/GAT/blob/master/execute_cora.py
            if val_acc >= vacc_mx or val_loss <= vlss_mn:
                if val_acc >= vacc_mx and val_loss <= vlss_mn:
                    state_dict_early_model = model.state_dict()
                vacc_mx = np.max((val_acc, vacc_mx))
                vlss_mn = np.min((val_loss, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= epoch_patience:
                    break

        """
        Evaluate Model.
        """
        model.load_state_dict(state_dict_early_model)
        model.eval()
        with torch.no_grad():
            eval_logits = model(train_data)
            eval_logp = F.log_softmax(eval_logits, 1)
            eval_pred = eval_logp.argmax(dim=1)
            return eval_pred
