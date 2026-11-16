import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


class MLP(nn.Module):
    """Fully connected regression network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.2,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation: {activation}. Choose from {list(ACTIVATIONS)}"
            )

        layers = []
        in_size = input_size

        for _ in range(n_layers):
            layers.append(nn.Linear(in_size, hidden_size))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(ACTIVATIONS[activation]())

            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

            in_size = hidden_size

        layers.append(nn.Linear(in_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Predictions of shape (batch,)."""
        return self.layers(X).squeeze(-1)
