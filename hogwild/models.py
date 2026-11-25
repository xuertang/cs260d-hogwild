"""
Role B Task

CPU models for the four benchmark tasks.

Defines models to be used:
LogReg
Softmax
MLP
CNN
"""

import torch
from torch import nn


class LogReg(nn.Module):
    """
    Binary logits over sparse RCV1 bags of weighted feature indices.

    Batch size = 32
    LR = 1.0
    Reference Epochs = 5
    """

    def __init__(self, n_features: int = 47_236) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_features, 1))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.coalesce()
        rows, columns = x.indices()
        weights = nn.functional.embedding(
            columns, self.weight, sparse=True
        ).squeeze(1)
        return weights.new_zeros(x.size(0)).index_add(
            0, rows, weights * x.values()
        ) + self.bias


class Softmax(nn.Sequential):
    """
    MNIST linear regression. Cross-entropy applies softmax to the logits.

    Batch size = 65
    LR = 0.025
    Reference Epochs = 10
    """

    def __init__(self) -> None:
        super().__init__(nn.Flatten(), nn.Linear(28 * 28, 10))


class MLP(nn.Sequential):
    """
    Two-Layer MLP 

    Batch size = 64
    LR = 0.1
    Reference Epochs = 10
    """

    def __init__(self) -> None:
        super().__init__(
            nn.Flatten(), nn.Linear(28 * 28, 256), nn.ReLU(), nn.Linear(256, 10)
        )


class SmallCNN(nn.Sequential):
    """
    CNN model

    Batch size = 64
    LR = 0.1
    Reference Epcohs = 20
    """
    def __init__(self) -> None:
        super().__init__(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(192, 10)
        )
