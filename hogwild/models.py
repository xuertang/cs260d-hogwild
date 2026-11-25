"""The four models (README §4). Parameter counts are checked in tests/smoke.

    LogReg     T1   47,236   sparse l2-logistic regression (no bias)
    Softmax    T2    7,850   multinomial logistic regression
    MLP        T3a 203,530   784 → 256 → 10
    SmallCNN   T3b 316,554   4 conv + linear (Andrew's architecture)

All of them are small on purpose (README §9): a few hundred KB of parameters
keeps memory bandwidth from capping the speed-up we are trying to measure.
"""

from __future__ import annotations

import torch
from torch import nn


class LogReg(nn.Module):
    """w · x for sparse x. A single weight vector and no bias, so that every
    coordinate an update writes belongs to a feature the example actually has —
    a bias would be one coordinate that every update collides on."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_features))

    def forward(self, cols: torch.Tensor, vals: torch.Tensor, rows: torch.Tensor, n: int) -> torch.Tensor:
        """Batch in COO form: entry k is feature cols[k] with value vals[k] of example rows[k]."""
        return torch.zeros(n).index_add_(0, rows, self.weight[cols] * vals)


class Softmax(nn.Module):
    def __init__(self, in_features: int = 784, classes: int = 10) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


class MLP(nn.Module):
    def __init__(self, in_features: int = 784, hidden: int = 256, classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x.flatten(1))))


class SmallCNN(nn.Sequential):
    """Four 3x3 convolutions (32, 64, 128, 192 channels), global average pool, linear.

    Andrew's architecture: 316,554 parameters, the README's "0.3M". The first
    convolution has stride 2 and each of the first three is followed by a 2x2
    max-pool, so the feature maps are 8, 4, 2 and 2 pixels wide — about 3.5M
    multiply-adds per image, which is what lets a CPU run finish in minutes.
    No batch norm: its running statistics would be extra shared state with
    its own races.
    """

    def __init__(self, classes: int = 10) -> None:
        super().__init__(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(192, classes),
        )
