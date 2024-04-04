import time
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
import numpy as np
import logging
from torch import tensor
from torch_geometric.utils import index_to_mask
from models import GCN, GCN_2L
from itertools import product


class TrainerTorch:
    def __init__(self, run_id, lr=0.01, weight_decay=5e-4, n_epochs=500, lr_decay=0.8, lr_patience=50, decay_steps=50):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # run parameters
        self.run_id = run_id
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        # self.epoch_patience = epoch_patience
        self.decay_steps = decay_steps
        self.n_folds = 10

        # cross-validation parameters
        self.layers = [2]  # 2, 3, 4, 5
        self.hiddens = [16, 32, 64, 128]  # [16, 32, 64, 128, 256]
        self.net = GCN

        # logging
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(f"logs/{run_id}/results.log"))
        self.logger.addHandler(logging.StreamHandler())

    def run(self, data):
        # model performance results
        results = []
        best_result = (float('inf'), 0, 0)  # (loss, acc, std)
        best_desc = ""
        best_pred = None

        # log run parameters
        self.logger.info(f"# Run({self.run_id}) parameters:"
                         f"\n\tn_epochs={self.n_epochs},"
                         f"\n\tlr={self.lr},"
                         f"\n\tlr_decay={self.lr_decay},"
                         f"\n\tdecay_steps={self.decay_steps},"
                         f"\n\tweight_decay={self.weight_decay},"
                         f"\n\tn_folds={self.n_folds},"
                         f"\n\tcv_layers={self.layers},"
                         f"\n\tcv_hiddens={self.hiddens},"
                         f"\n\tcv_models={self.net}\n"
                         )

        # run & evaluate each model configuration
        for model_idx, (n_layers, n_hidden) in enumerate(product(self.layers, self.hiddens)):
            # graph convolutional network model
            model = self.net(
                data=data,
                n_hidden=n_hidden,
                n_layers=n_layers,
            )
            self.logger.info(f"# Evaluating model({model_idx}): n_layers={n_layers}, n_hidden={n_hidden}")

            # cross-validate & train model
            loss, acc, std = self.cross_validation(
                data,
                model,
                n_folds=self.n_folds,
                n_epochs=self.n_epochs,
                lr=self.lr,
                lr_decay=self.lr_decay,
                lr_step_size=self.decay_steps,
                lr_patience=self.lr_patience,
                weight_decay=self.weight_decay,
            )

            # save best model results & predictions
            if loss < best_result[0]:
                best_result = (loss, acc, std)
                best_pred = self.predict(model, data)
                best_desc = f"Best Model({model_idx}) results: accuracy={best_result[1]:.3f}, std={best_result[2]:.3f}, n_layers={n_layers}, n_hidden={n_hidden}"

            desc = f"accuracy={acc:.3f}, std={std:.3f}, n_layers={n_layers}, n_hidden={n_hidden}"
            self.logger.info(f"Model({model_idx}) results: {desc}")
            self.logger.info(f'{best_desc}\n')
            results += [f'{model}({model_idx}): {desc}']

        # results from all models
        results = '\n'.join(results)
        self.logger.info(f'--\n{results}')
        return best_pred

    def cross_validation(self, data, model, n_epochs, n_folds, lr, weight_decay, lr_decay, lr_patience, lr_step_size):
        val_losses, accs, durations = [], [], []
        x_len = len(data.idx_train)

        # train model across k-folds
        for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*self.k_fold(data, n_folds))):

            # prepare fold data masks
            data.train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
            data.train_mask[data.idx_train] = index_to_mask(train_idx, size=x_len)

            data.val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
            data.val_mask[data.idx_train] = index_to_mask(val_idx, size=x_len)

            data.test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
            data.test_mask[data.idx_train] = index_to_mask(test_idx, size=x_len)

            # optimizer
            model.to(self.device).reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # TODO learning rate scheduler
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=lr_decay,
                patience=lr_patience
            )

            # train model for each epoch in this fold
            t_start = time.perf_counter()
            for epoch in range(1, n_epochs + 1):
                train_loss = self.train(model, optimizer, data)
                val_losses.append(self.eval_loss(model, data))
                accs.append(self.eval_accuracy(model, data))

                if epoch % lr_step_size == 0:
                    # decay learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_decay * param_group['lr']

                    # log epoch results
                    eval_info = {
                        'fold': fold,
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_losses[-1],
                        'test_acc': accs[-1],
                    }
                    self.logger.info(eval_info)

            # cuda optimization
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.synchronize()

            # fold duration
            t_end = time.perf_counter()
            durations.append(t_end - t_start)

        # final model performance
        loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
        loss, acc = loss.view(n_folds, n_epochs), acc.view(n_folds, n_epochs)
        loss, argmin = loss.min(dim=1)
        acc = acc[torch.arange(n_folds, dtype=torch.long), argmin]

        loss_mean = loss.mean().item()
        acc_mean = acc.mean().item()
        acc_std = acc.std().item()
        duration_mean = duration.mean().item()
        logging.info(f'Val Loss: {loss_mean:.4f}, Test Accuracy: {acc_mean:.3f} '
                     f'+/- {acc_std:.3f}, Duration: {duration_mean:.3f}')

        return loss_mean, acc_mean, acc_std

    def train(self, model, optimizer, data):
        """
        Train model.
        """
        model.train()
        optimizer.zero_grad()
        data = data.to(self.device)

        out = model(data)
        out_logp = F.log_softmax(out, 1)
        loss = F.nll_loss(out_logp[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_loss(self, model, data):
        """
        Validate model.
        """
        model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            out = model(data)
            out_logp = F.log_softmax(out, 1)
            loss = F.nll_loss(out_logp[data.val_mask], data.y[data.val_mask]).item()
            return loss

    def eval_accuracy(self, model, data):
        """
        Test model.
        """
        model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            out = model(data)
            out_logp = F.log_softmax(out, 1)
            pred = out_logp.argmax(dim=1)
            accuracy = torch.eq(pred[data.test_mask], data.y[data.test_mask]).float().mean().item()
            return accuracy

    def k_fold(self, data, n_folds: int):
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=12345)
        test_indices, train_indices = [], []
        x_len = len(data.idx_train)

        # test index folds
        for _, idx in skf.split(torch.zeros(x_len), data.y[data.idx_train]):
            test_indices.append(torch.from_numpy(idx).to(torch.long))

        # val index folds
        val_indices = [test_indices[i - 1] for i in range(n_folds)]

        # train index folds
        for i in range(n_folds):
            train_mask = torch.ones(x_len, dtype=torch.bool)
            train_mask[test_indices[i]] = 0
            train_mask[val_indices[i]] = 0
            train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

        return train_indices, test_indices, val_indices

    def predict(self, model, data):
        """
        Final best-model evaluation.
        """
        model.eval()
        with torch.no_grad():
            out = model(data)
            out_logp = F.log_softmax(out, 1)
            pred = out_logp.argmax(dim=1)
            return pred
