import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from utils.helpers import plot_loss


def train(
    network: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    val_loader: DataLoader | None = None,
    scheduler: LRScheduler | None = None,
    plot: bool = False,
) -> dict[str, list[float]]:
    """Train for a fixed number of epochs and return loss history."""
    history = {"train": [], "val": []}

    for _ in range(epochs):
        history["train"].append(_train_epoch(network, train_loader, optimizer, loss_fn))

        if val_loader is not None:
            val_loss, _ = _val_epoch(network, val_loader, loss_fn)
            history["val"].append(val_loss)

        if scheduler is not None:
            scheduler.step()

    if plot:
        plot_loss(history)

    return history


def predict(network: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Predict in eval mode and return result on CPU."""
    network.eval()
    with torch.no_grad():
        return network(X.to(_device_of(network))).cpu()


def _train_epoch(
    network: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
) -> float:
    """Train one epoch and return mean loss."""
    total_loss = 0.0
    device = _device_of(network)

    network.train()
    for batch in train_loader:
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        preds = network(X_batch)
        loss = loss_fn(preds, y_batch)

        total_loss += loss.item() * len(y_batch)

        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader.dataset)


def _val_epoch(
    network: nn.Module, val_loader: DataLoader, loss_fn: nn.Module
) -> tuple[float, torch.Tensor]:
    """Evaluate one epoch and return mean loss with predictions."""
    network.eval()

    total_loss = 0.0
    preds_total = []
    device = _device_of(network)

    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            preds = network(X_batch)
            loss = loss_fn(preds, y_batch)

            preds_total.append(preds.cpu())
            total_loss += loss.item() * len(y_batch)

    return total_loss / len(val_loader.dataset), torch.cat(preds_total)


def _device_of(network: nn.Module) -> torch.device:
    """Device of the network weights."""
    return next(network.parameters()).device
