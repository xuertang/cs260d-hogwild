"""Synthetic stand-in Task. NOT one of the four real tasks (those are stream B's).

Why it exists: README §8 says A and C start on day 1 "against a stub Task".
This is that stub — a tiny dense linear-regression problem that satisfies the
Task contract in schema.py, needs no downloads, and trains in well under a
second. Use it to smoke-test runner.py, modes/*.py and sweep.py before
tasks.py exists.

Delete this file (or move it under tests/) once T1/T2 run end-to-end.
Nothing in the report should ever come from it.
"""

from __future__ import annotations

from typing import Iterable

import torch


class StubTask:
    name = "stub"
    target_loss = 0.05   # arbitrary; just needs to be reachable in a few epochs
    default_eta = 0.05   # arbitrary; not tuned

    def __init__(self, n: int = 2048, d: int = 32, noise: float = 0.1) -> None:
        # Fixed generator so every process that builds a StubTask sees the same data.
        g = torch.Generator().manual_seed(0)
        w_true = torch.randn(d, 1, generator=g)
        self._x = torch.randn(n, d, generator=g)
        self._y = self._x @ w_true + noise * torch.randn(n, 1, generator=g)
        self._x_test = torch.randn(n // 4, d, generator=g)
        self._y_test = self._x_test @ w_true
        self._d = d

    def model(self) -> torch.nn.Module:
        return torch.nn.Linear(self._d, 1)

    def train_loader(self, seed: int, batch_size: int) -> Iterable:
        """One epoch of (x, y) minibatches in an order fixed by `seed`."""
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(self._x.shape[0], generator=g)
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            yield self._x[idx], self._y[idx]

    def evaluate(self, params) -> tuple[float, float]:
        """`params` follows the proposal in schema.Task.evaluate: tensors in
        model().parameters() order, i.e. (weight, bias) here.

        Regression has no accuracy, so the second element is a placeholder 0.0.
        """
        weight, bias = params
        with torch.no_grad():
            pred = self._x_test @ weight.T + bias
            loss = torch.mean((pred - self._y_test) ** 2).item()
        return loss, 0.0
