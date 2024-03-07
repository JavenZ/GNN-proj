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
from torch_geometric.utils import from_scipy_sparse_matrix
from reporter import ResultsReporter
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters


def train_torch():
    """
    Preprocess data.
    """
    x = features

    # scale
    scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    # dimensionality reduction
    # pca = PCA(.95)
    # print(x.shape)
    # pca.fit(x)
    # x = pca.transform(x)
    # print(x.shape)

    # format data
    x = torch.from_numpy(x).type(torch.float)
    labels.resize((x.shape[0],))
    y = torch.from_numpy(labels).type(torch.long)
    edges = from_scipy_sparse_matrix(adj)

    """
    Create data splits.
    """
    np.random.shuffle(idx_train)
    train_mask = np.zeros((x.shape[0],)).astype(bool)
    # train_mask[idx_train[:450]] = 1
    val_mask = np.zeros((x.shape[0],)).astype(bool)
    # val_mask[idx_train[450:]] = 1
    val_mask[idx_train] = 1
    test_mask = torch.zeros((x.shape[0],)).type(torch.bool)
    test_mask[idx_test] = 1

    """
    Construct Torch data objects.
    """
    data = Data(
        x=x,
        y=y,
        edge_index=edges[0],
        # edge_attr=edges[1],
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    # joined_data = Data(
    #     x=torch.concatenate((data.x, cora_data.x)),
    #     y=torch.concatenate((data.y, cora_data.y)),
    #     edge_index=torch.concatenate((data.edge_index, cora_data.edge_index), dim=1),
    #     train_mask=np.concatenate((data.train_mask, cora_data.train_mask)),
    #     val_mask=data.val_mask.copy()
    # )
    # joined_data.val_mask.resize(joined_data.x.shape[0])
    # print(joined_data)

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
    trainer = TrainerTorch()
    trainer.run(data)


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


if __name__ == "__main__":
    """
    Load data.
    """
    labels = np.load('./data_2024/labels.npy')  # (496,) [0-6] training labels
    features = np.load('./data_2024/features.npy')  # (2480, 1390) [0-1] node features

    adj = sp.load_npz('./data_2024/adj.npz')
    adj_csr = np.load('./data_2024/adj.npz')  # (2480, 2480) CSR matrix
    adj_csr = csr_matrix((adj_csr['data'], adj_csr['indices'], adj_csr['indptr']), shape=adj_csr['shape']).toarray()

    splits = json.load(open('./data_2024/splits.json'))  # ['idx_train', 'idx_test']
    idx_train = splits['idx_train']  # (496,)
    idx_test = splits['idx_test']  # (1984,)

    cora_data = Planetoid(root='C:\Tmp\Cora', name='Cora')[0]

    """
    Run trainer.
    """
    # train_torch()
    train_neat()
