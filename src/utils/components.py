import torch
from torch import nn
from torch.optim import Optimizer

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}

LOSSES = {
    "mse": nn.MSELoss,
}


def get_optimizer(
    name: str, model: nn.Module, lr: float, weight_decay: float
) -> Optimizer:
    """Optimizer by name."""
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}. Choose from {list(OPTIMIZERS)}")

    return OPTIMIZERS[name](model.parameters(), lr=lr, weight_decay=weight_decay)


def get_loss(name: str) -> nn.Module:
    """Loss function by name."""
    if name not in LOSSES:
        raise ValueError(f"Unknown loss: {name}. Choose from {list(LOSSES)}")

    return LOSSES[name]()
