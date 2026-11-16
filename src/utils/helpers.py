import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import ArrayLike
from torch.utils.data import DataLoader, TensorDataset


def plot_loss(history: dict[str, list[float]], title: str = "Loss") -> None:
    """Plot loss per epoch."""
    plt.figure(figsize=(8, 4))
    plt.plot(history["train"], label="train")
    if history.get("val"):
        plt.plot(history["val"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def to_tensor(values: ArrayLike) -> torch.Tensor:
    """Float32 tensor from a table or array."""
    return torch.as_tensor(np.asarray(values, dtype=np.float32))


def get_data_loader(
    X: ArrayLike, y: ArrayLike, batch_size: int, seed: int
) -> DataLoader:
    """Shuffled training batches."""
    dataset = TensorDataset(to_tensor(X), to_tensor(y).reshape(-1))

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # A batch of one row breaks BatchNorm.
        drop_last=True,
        generator=generator,
    )
    return loader
