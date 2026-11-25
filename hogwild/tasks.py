"""
Role B Task 

Dataset loading, objectives, and evaluation for the benchmark tasks.

Loads the datasets, shuffles training batches reproducibly, and calculates test loss and accuracy. RCV1’s feature gradients stay sparse so it can serve as the sparse baseline.

Loads dataset from data/ folder. 
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from functools import cached_property
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from hogwild.models import LogReg, MLP, SmallCNN, Softmax
from hogwild.schema import Task


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


class ClassificationTask:
    def __init__(self, data_root: str | Path = DATA_ROOT) -> None:
        self.data_root = Path(data_root)

    def train_loader(self, seed: int, batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield one independently shuffled pass over the training split."""
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(self.train_size, generator=generator)
        return self._batches(train=True, batch_size=batch_size, order=order)

    def loss(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(model(x), y)

    def _test_objective(self, model, x, y) -> tuple[torch.Tensor, torch.Tensor]:
        logits = model(x)
        return F.cross_entropy(logits, y, reduction="sum"), logits.argmax(dim=1)

    def _test_penalty(self, model: nn.Module) -> float:
        return 0.0

    @torch.no_grad()
    def evaluate(self, params: Iterable[torch.Tensor]) -> tuple[float, float]:
        """Score the test split using tensors in model.parameters() order."""
        with torch.random.fork_rng(devices=[]):
            model = self.model()
        for local, supplied in zip(model.parameters(), params, strict=True):
            if local.shape != supplied.shape:
                raise ValueError("Evaluation parameter shapes do not match the model")
            local.copy_(supplied)
        model.eval()

        total_loss = correct = count = 0
        for x, y in self._batches(train=False, batch_size=self.eval_batch_size):
            loss, predicted = self._test_objective(model, x, y)
            total_loss += loss.item()
            correct += (predicted == y).sum().item()
            count += y.numel()
        return total_loss / count + self._test_penalty(model), correct / count


class RCV1Task(ClassificationTask):
    name = "t1"
    default_eta = 1.0
    target_loss = 0.3411427186015997
    eval_batch_size = 4096
    l2 = 1e-4

    def _fetch(self, subset: str, download: bool = False):
        from sklearn.datasets import fetch_rcv1

        return fetch_rcv1(
            data_home=str(self.data_root), subset=subset, download_if_missing=download
        )

    def download(self) -> None:
        self._fetch("train", download=True)

    def _load(self, subset: str):
        data = self._fetch(subset)
        category = list(data.target_names).index("CCAT")
        labels = torch.from_numpy(data.target[:, category:category + 1].toarray().ravel()).float()
        features = data.data.astype("float32")
        features.sum_duplicates()
        features.eliminate_zeros()
        return features, labels

    @cached_property
    def _train(self):
        return self._load("train")

    @cached_property
    def _test(self):
        return self._load("test")

    @property
    def train_size(self) -> int:
        return len(self._train[1])

    @cached_property
    def _inverse_frequency(self) -> torch.Tensor:
        x, _ = self._train
        counts = torch.bincount(
            torch.from_numpy(x.indices.astype("int64")), minlength=x.shape[1]
        )
        return self.train_size / counts.clamp_min(1).float()

    def model(self) -> LogReg:
        return LogReg()

    def _batches(self, train: bool, batch_size: int, order=None):
        features, labels = self._train if train else self._test
        if order is None:
            order = torch.arange(len(labels))
        for indices in order.split(batch_size):
            batch = features[indices.numpy()].tocoo()
            coordinates = torch.stack(
                (torch.from_numpy(batch.row), torch.from_numpy(batch.col))
            )
            x = torch.sparse_coo_tensor(
                coordinates, torch.from_numpy(batch.data), batch.shape,
                is_coalesced=True, check_invariants=False,
            )
            yield x, labels[indices]

    def loss(self, model: LogReg, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        data_loss = F.binary_cross_entropy_with_logits(model(x), y)
        columns = x.indices()[1]
        weights = F.embedding(columns, model.weight, sparse=True).squeeze(1)
        # Allocate each coordinate's L2 penalty across the rows containing it.
        # The training-set mean recovers full L2 on observed coordinates.
        penalty = (weights.square() * self._inverse_frequency[columns]).sum()
        return data_loss + (0.5 * self.l2 / len(y)) * penalty

    def _test_objective(self, model, x, y) -> tuple[torch.Tensor, torch.Tensor]:
        logits = model(x)
        return F.binary_cross_entropy_with_logits(logits, y, reduction="sum"), logits >= 0

    def _test_penalty(self, model: LogReg) -> float:
        return 0.5 * self.l2 * model.weight.square().sum().item()


class VisionTask(ClassificationTask):
    eval_batch_size = 512

    def _dataset(self, train: bool, download: bool = False):
        from torchvision import datasets

        dataset_type = getattr(datasets, self.dataset_name)
        return dataset_type(root=self.data_root, train=train, download=download)

    def download(self) -> None:
        for train in (True, False):
            self._dataset(train, download=True)

    def _load(self, train: bool):
        dataset = self._dataset(train)
        images = torch.as_tensor(dataset.data)
        if self.dataset_name == "MNIST":
            images = images.unsqueeze(1)
        else:
            # Store the CIFAR image axes contiguously so indexing keeps RGB together.
            images = images.contiguous()
        return images, torch.as_tensor(dataset.targets)

    @cached_property
    def _train(self):
        return self._load(train=True)

    @cached_property
    def _test(self):
        return self._load(train=False)

    @property
    def train_size(self) -> int:
        return len(self._train[1])

    def _batches(self, train: bool, batch_size: int, order=None):
        images, labels = self._train if train else self._test
        if order is None:
            order = torch.arange(len(labels))
        mean = torch.tensor(self.mean).view(1, -1, 1, 1)
        std = torch.tensor(self.std).view(1, -1, 1, 1)
        for indices in order.split(batch_size):
            x = images[indices]
            if self.dataset_name == "CIFAR10":
                x = x.permute(0, 3, 1, 2)
            x = x.float().div_(255).sub_(mean).div_(std)
            yield x, labels[indices]


class MNISTSoftmaxTask(VisionTask):
    name = "t2"
    dataset_name = "MNIST"
    mean = (0.1307,)
    std = (0.3081,)
    default_eta = 0.025
    target_loss = 0.2796574394226074

    def model(self) -> Softmax:
        return Softmax()


class MNISTMLPTask(MNISTSoftmaxTask):
    name = "t3a"
    default_eta = 0.1
    target_loss = 0.06365422883033753

    def model(self) -> MLP:
        return MLP()


class CIFAR10Task(VisionTask):
    name = "t3b"
    dataset_name = "CIFAR10"
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    default_eta = 0.1
    target_loss = 1.3762148345947265

    def model(self) -> SmallCNN:
        return SmallCNN().to(memory_format=torch.channels_last)


_TASKS = {
    "t1_rcv1_logreg": RCV1Task,
    "t2_mnist_softmax": MNISTSoftmaxTask,
    "t3a_mnist_mlp": MNISTMLPTask,
    "t3b_cifar_cnn": CIFAR10Task,
}


def get_task(name: str, data_root: str | Path = DATA_ROOT) -> Task:
    """Resolve a short task ID or config stem without reading dataset files."""
    for stem, task_type in _TASKS.items():
        if name in (stem, task_type.name):
            return task_type(data_root)
    if name == "stub":
        from hogwild.stub_task import StubTask

        return StubTask()
    raise ValueError(f"Unknown task {name!r}. Expected t1, t2, t3a, t3b, or stub.")


def training_loss(
    task: Task, model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Use task-specific objectives while supporting the original Task protocol."""
    loss = getattr(task, "loss", None)
    if loss is not None:
        return loss(model, x, y)
    if y.is_floating_point():
        return F.mse_loss(model(x), y)
    return F.cross_entropy(model(x), y)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download"])
    parser.add_argument(
        "datasets", nargs="*", metavar="DATASET",
        help="rcv1, mnist, or cifar10 (default: all)",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args(argv)
    datasets = {"rcv1": RCV1Task, "mnist": MNISTSoftmaxTask, "cifar10": CIFAR10Task}
    for name in args.datasets or datasets:
        if name not in datasets:
            parser.error(f"unknown dataset {name!r}")
        print(f"Downloading {name} to {args.data_root}", flush=True)
        datasets[name](args.data_root).download()


if __name__ == "__main__":
    main()
