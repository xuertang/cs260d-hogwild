"""The four models (README §4). Parameter counts are checked in tests/smoke.

    LogReg     T1   47,236   sparse l2-logistic regression (no bias)
    Softmax    T2    7,850   multinomial logistic regression
    MLP        T3a 203,530   784 → 256 → 10
    SmallCNN   T3b 245,962   4 conv + linear

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


class SmallCNN(nn.Module):
    """(conv – relu – pool) x 4 with 32, 64, 128, 128 channels, then linear.

    Pooling after every convolution keeps the cost per image low (about 13M
    multiply-adds forward), which is what lets a CPU run finish in minutes.
    No batch norm: its running statistics would be extra shared state with
    its own races.
    """

    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.fc = nn.Linear(128 * 2 * 2, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pool = nn.functional.max_pool2d
        x = pool(torch.relu(self.conv1(x)), 2)   # 32 → 16
        x = pool(torch.relu(self.conv2(x)), 2)   # 16 → 8
        x = pool(torch.relu(self.conv3(x)), 2)   # 8 → 4
        x = pool(torch.relu(self.conv4(x)), 2)   # 4 → 2
        return self.fc(x.flatten(1))
