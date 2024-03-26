import numpy as np
import pandas as pd
import json
import neat
import torch
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from trainer_neat import TrainerNEAT
from trainer_torch import TrainerTorch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, mask_to_index, unbatch_edge_index, to_edge_index
from reporter import ResultsReporter
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit


def train_torch(labels, features, adj, adj_csr, idx_train, idx_test, cora_data):
    """
    Preprocess data.
    """
    x = features

    # dimensionality reduction
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)
    # pca = PCA(.95)
    # print(x.shape)
    # pca.fit(x)
    # x = pca.transform(x)
    # print(x.shape)

    # graph aggregated spacial filter
    # x = []
    # for i in range(len(features)):
    #     neigh_feats = features[adj_csr[i].astype(bool)]
    #     agg_feats = features[i] + neigh_feats.mean(axis=0)  # weighed-mean aggregation
    #     x.append(agg_feats)
    # x = np.array(x)
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    # format data
    x = torch.from_numpy(x).type(torch.float)
    y = torch.zeros(x.shape[0]).type(torch.long)
    y[idx_train] = torch.from_numpy(labels).type(torch.long)
    edges = from_scipy_sparse_matrix(adj)

    """
    Construct Torch data object(s).
    """
    train_data = Data(
        x=x[idx_train],
        y=y,
        edge_index=edges[0],
        train_mask=torch.zeros(x.shape[0]).type(torch.bool),
        val_mask=torch.zeros(x.shape[0]).type(torch.bool),
        test_mask=torch.zeros(x.shape[0]).type(torch.bool),
    )
    # create data splits
    transform = RandomNodeSplit(num_test=0, num_val=50)
    t = transform(train_data)
    train_data.x = x
    train_data.train_mask[idx_train] = t.train_mask
    train_data.val_mask[idx_train] = t.val_mask
    # train_data.test_mask[idx_test] = t.test_mask

    """
    Synthesize additional training data.
    """
    # df = pd.DataFrame(data.x.numpy())
    # print(df)
    # synth = RegularSynthesizer(modelname='fast')
    # synth.fit(data=df)
    # synth_data = synth.sample(1000)
    # print(synth_data)

    """
    Run Trainer.
    """
    run_id = 23
    trainer = TrainerTorch(run_id=run_id,)
    y_pred = trainer.run(
        train_data=train_data,
        lr=0.1,
        weight_decay=5e-4,
        n_hidden=16,
        n_epochs=5000,
        lr_decay=0.95,
        lr_patience=50,
        epoch_patience=250,
    )
    # sort predictions by correct index & save results
    y_pred = y_pred[idx_test]
    trainer.logger.info(
        f"val_mask: {train_data.val_mask.sum().item()}, train_mask: {train_data.train_mask.sum().item()}, test_mask: {train_data.test_mask.sum().item()}"
    )
    print(f"y_pred[:10] = {y_pred[:10].tolist()}")
    print(f"y_real[:10] = [1, 2, 2, 1, 1, 2, 3, 1, 1, 1]")
    np.savetxt(f'logs/submission_{run_id}.txt', y_pred, fmt='%d')


def train_neat(labels, features, adj, adj_csr, idx_train, idx_test, cora_data):
    """
    Preprocess data.
    """
    # graph aggregated spacial filter
    x = []
    for i in range(len(features)):
        neigh_feats = features[adj_csr[i].astype(bool)]
        # print(neigh_feats, neigh_feats.shape)
        agg_feats = features[i] + neigh_feats.mean(axis=0)  # weighed-mean aggregation
        x.append(agg_feats)
    x = np.array(x)

    # scale
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # dimensionality reduction
    # pca = PCA(.95)
    # x = pca.fit_transform(x)
    # print(pca.n_components_)

    # split
    x_train = np.array([x[idx] for idx in idx_train])
    x_test = np.array([x[idx] for idx in idx_test])
    print(x_train.shape, x_test.shape)

    """
    Train using NEAT.
    """
    # load configuration
    cfg = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        './src/neat.cfg'
    )
    trainer = TrainerNEAT(cfg, x_train, labels, x_test)
    trainer.run()


if __name__ == "__main__":
    """
    Load data.
    """
    _labels = np.load('./data_2024/labels.npy')  # (496,) [0-6] training labels
    _features = np.load('./data_2024/features.npy')  # (2480, 1390) [0-1] node features

    _adj = sp.load_npz('./data_2024/adj.npz')
    _adj_csr = np.load('./data_2024/adj.npz')  # (2480, 2480) CSR matrix
    _adj_csr = csr_matrix((_adj_csr['data'], _adj_csr['indices'], _adj_csr['indptr']), shape=_adj_csr['shape']).toarray()

    splits = json.load(open('./data_2024/splits.json'))  # ['idx_train', 'idx_test']
    _idx_train = splits['idx_train']  # (496,)
    _idx_test = splits['idx_test']  # (1984,)

    _cora = Planetoid(root='C:\Tmp\Cora', name='Cora')[0]

    """
    Run trainer.
    """
    train_torch(_labels, _features, _adj, _adj_csr, _idx_train, _idx_test, _cora)
    # train_neat()
