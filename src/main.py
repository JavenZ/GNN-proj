import os
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
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_scipy_sparse_matrix, mask_to_index, unbatch_edge_index, to_edge_index
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit


def train_torch():
    """
    Preprocess data.
    """
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
    #     # agg_feats = features[i] + neigh_feats.sum(axis=0)  # weighed-sum aggregation
    #     x.append(agg_feats)
    # x = np.array(x)
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    # format data
    x = torch.from_numpy(features).type(torch.float)
    # x = torch.from_numpy(x).type(torch.float)
    y = torch.zeros(x.shape[0]).type(torch.long)
    y[idx_train] = torch.from_numpy(labels).type(torch.long)
    edges = from_scipy_sparse_matrix(adj)

    """
    Construct Torch data object(s).
    """
    data = Data(
        x=x,
        y=y,
        edge_index=edges[0],
        idx_train=idx_train,
        idx_test=idx_test,
    )

    """
    Synthesize additional training data.
    """
    # d = {'0': train_data.x[idx_train][:, 0].numpy(), 'class': train_data.y[idx_train].numpy()}
    # df = pd.DataFrame(d)
    # print(df)
    # synth = RegularSynthesizer(modelname='fast')
    # synth.fit(data=df)
    # synth_data = synth.sample(1000)
    # print(synth_data)
    # exit(1)

    """
    Run Trainer.
    """
    run_id = find_run_idx()
    trainer = TrainerTorch(
        run_id=run_id,
        weight_decay=5e-4,
        lr_decay=0.90,
        lr_patience=50,
        decay_steps=50,
    )
    y_pred = trainer.run(data=data)

    # sort predictions by correct index & save results
    y_pred = y_pred[idx_test]
    trainer.logger.info(f"{x.numpy().shape}")
    print(f"y_pred[:10] = {y_pred[:10].tolist()}")
    print(f"y_real[:10] = [1, 2, 2, 1, 1, 2, 3, 1, 1, 1]")
    np.savetxt(f'logs/{run_id}/submission.txt', y_pred, fmt='%d')


def train_neat():
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


def find_run_idx():
    max_idx = -1
    for _dir in os.listdir('logs/'):
        idx = int(_dir)
        max_idx = max(max_idx, idx)
    run_idx = max_idx + 1
    if not os.path.exists(f'logs/{run_idx}'):
        os.mkdir(f'logs/{run_idx}')
    return run_idx


if __name__ == "__main__":
    """
    Load data.
    """
    labels = np.load('./data_2024/labels.npy')  # (496,) [0-6] training labels
    features = np.load('./data_2024/features.npy')  # (2480, 1390) [0-1] node features

    adj = sp.load_npz('./data_2024/adj.npz')
    adj_csr = np.load('./data_2024/adj.npz')  # (2480, 2480) CSR matrix
    adj_csr = csr_matrix((adj_csr['data'], adj_csr['indices'], adj_csr['indptr']), shape=adj_csr['shape']).toarray()

    _splits = json.load(open('./data_2024/splits.json'))  # ['idx_train', 'idx_test']
    idx_train = _splits['idx_train']  # (496,)
    idx_test = _splits['idx_test']  # (1984,)

    cora_dataset = Planetoid(root='C:\Tmp\Cora', name='Cora')

    """
    Run trainer.
    """
    train_torch()
    # train_neat()
