import numpy as np
import pandas as pd
import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('remove_free_columns')
class RemoveFreeColumns(BaseTransform):
    """
    Removes linearly dependent feature columns.
    """
    def __init__(self, free_columns=None):
        self.free_columns = free_columns

    def forward(self, data: Data) -> Data:
        assert data.x is not None

        X = pd.DataFrame(data=data.x)
        rank = np.linalg.matrix_rank(X)  # 1385 / 1390
        print(f"Rank and Shape before dropping any features: {rank}, {X.shape}")

        # search for free columns?
        if not self.free_columns:
            self.free_columns = []
            for i in range(0, X.shape[1]):
                t = X.drop(columns=X.columns[[i]], axis=1, inplace=False)
                t_rank = np.linalg.matrix_rank(t)
                if t_rank == rank:
                    print(i)
                    self.free_columns.append(i)

        # drop found or provided free columns
        print(f"Dropping free columns: {self.free_columns}")
        X = X.drop(columns=X.columns[self.free_columns], axis=1, inplace=False)
        rank = np.linalg.matrix_rank(X)  # 1385 / 1385
        print(f"Rank and Shape after dropping free columns: {rank}, {X.shape}")

        data.x = torch.from_numpy(X.to_numpy()).type(torch.float)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.free_columns})'


def remove_free_columns(x: np.ndarray, free_columns=None):
    X = pd.DataFrame(data=x)
    rank = np.linalg.matrix_rank(X)  # 1385 / 1390
    print(f"Rank and Shape before dropping any features: {rank}, {X.shape}")
    if not free_columns:
        free_columns = []
        for i in range(0, X.shape[1]):
            t = X.drop(columns=X.columns[[i]], axis=1, inplace=False)
            t_rank = np.linalg.matrix_rank(t)
            if t_rank == rank:
                print(i)
                free_columns.append(i)
    print(f"Dropping free columns: {free_columns}")
    X = X.drop(columns=X.columns[free_columns], axis=1, inplace=False)
    rank = np.linalg.matrix_rank(X)  # 1385 / 1385
    print(f"Rank and Shape after dropping free columns: {rank}, {X.shape}")
    return X.to_numpy()