import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn

from train.trainer import predict as network_predict
from train.trainer import train
from utils.components import get_loss, get_optimizer
from utils.helpers import get_data_loader, to_tensor


class NetworkRegressor(RegressorMixin, BaseEstimator):
    """Sklearn wrapper around a PyTorch network."""

    def __init__(
        self,
        network: type[nn.Module],
        hidden_size: int = 128,
        n_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.2,
        batch_norm: bool = False,
        optimizer: str = "adamw",
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        loss: str = "mse",
        epochs: int = 100,
        batch_size: int = 64,
        random_state: int = 42,
        device: str = "cpu",
        plot_loss: bool = False,
    ) -> None:
        self.network = network
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self.plot_loss = plot_loss

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NetworkRegressor":
        """Build and train the network on a standardized target."""
        torch.manual_seed(self.random_state)

        self.network_ = self.network(
            input_size=X.shape[1],
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            activation=self.activation,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
        )

        self.network_ = self.network_.to(self.device)

        self.y_mean_ = y.mean()
        self.y_std_ = y.std()
        y_norm = (y - self.y_mean_) / self.y_std_

        optimizer = get_optimizer(
            self.optimizer, self.network_, lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = get_loss(self.loss)
        loader = get_data_loader(
            X, y_norm, batch_size=self.batch_size, seed=self.random_state
        )

        self.history_ = train(
            self.network_, loader, optimizer, loss_fn, self.epochs, plot=self.plot_loss
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predictions in the original target scale."""
        check_is_fitted(self, "network_")

        return (
            network_predict(self.network_, to_tensor(X)).numpy() * self.y_std_
            + self.y_mean_
        )
