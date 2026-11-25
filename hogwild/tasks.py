"""Task registry: data, model, training gradient, evaluation (README §4).

    t1    RCV1 binary (CCAT vs. rest)   l2-logistic regression    sparse
    t2    MNIST                         softmax regression        dense, convex
    t3a   MNIST                         2-layer MLP               dense, non-convex
    t3b   CIFAR-10                      4-conv CNN                dense, non-convex
    stub  synthetic linear regression   (smoke tests only, never reported)

    python -m hogwild.tasks download     # fetch + convert everything into ./data
    python -m hogwild.tasks info         # sizes, parameter counts, reference values

Beyond the README contract, every task also provides

    grads(model, batch) → list of tensors aligned with model.parameters()
    share_memory()      → move the dataset into shared memory

`grads` exists because the contract has no training loss, and because T1 needs
a sparse gradient that autograd would not give cheaply. `share_memory` lets
the runner hand ONE copy of the dataset to all worker processes.

target_loss and default_eta are not hard-coded: they come from
configs/reference.yaml, which `python -m hogwild.tune` writes from serial runs.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Iterable, Iterator

import torch
import yaml

from hogwild import models

# torch warns once per process that sparse tensors are built without invariant
# checks. That is deliberate here (the column ids come straight from the
# dataset and the check would run on every step), so keep worker logs clean.
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
REFERENCE_PATH = REPO / "configs" / "reference.yaml"


def load_reference() -> dict[str, dict]:
    """configs/reference.yaml as a dict (empty if tuning has not been run)."""
    if not REFERENCE_PATH.exists():
        return {}
    with open(REFERENCE_PATH) as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #


class BaseTask:
    name: str = ""
    ref_epochs: int = 1          # target loss = best serial test loss within this many epochs (tune.py)
    fallback_eta: float = 0.01   # used only until configs/reference.yaml has a tuned value
    n_train: int = 0
    curve_eval_n: int | None = None   # None = loss curves use the full test set
    target_slack: float = 0.0         # target loss = (1 + slack) x best serial loss (tune.py)

    def __init__(self) -> None:
        ref = load_reference().get(self.name, {})
        self.default_eta: float = float(ref.get("default_eta", self.fallback_eta))
        # NaN = "not tuned yet": the runner then reports time_to_target as inf.
        self.target_loss: float = float(ref.get("target_loss", float("nan")))
        self._eval_model: torch.nn.Module | None = None

    # -- contract ---------------------------------------------------------- #

    def model(self) -> torch.nn.Module:
        raise NotImplementedError

    def train_loader(self, seed: int, batch_size: int) -> Iterable:
        """One pass over the training set in an order fixed by `seed`.
        The last incomplete batch is dropped, so every batch has batch_size examples."""
        raise NotImplementedError

    def evaluate(self, params) -> tuple[float, float]:
        """(test_loss, test_acc) on the full test set, for `params`: tensors in
        model().parameters() order."""
        model = self._model_with(params)
        with torch.no_grad():
            return self._evaluate(model, None)

    def evaluate_curve(self, params) -> tuple[float, float]:
        """What the evaluator calls for every point of the loss curves, and so
        what target_loss is defined on. Same as `evaluate` unless the task sets
        `curve_eval_n`, in which case only the first curve_eval_n test examples
        are used — for tasks where a full pass takes longer than the 2 s
        evaluation period."""
        model = self._model_with(params)
        with torch.no_grad():
            return self._evaluate(model, self.curve_eval_n)

    # -- extensions -------------------------------------------------------- #

    def loss(self, model: torch.nn.Module, batch) -> torch.Tensor:
        raise NotImplementedError

    def grads(self, model: torch.nn.Module, batch) -> list[torch.Tensor]:
        """Gradient of the minibatch loss, one tensor per parameter.
        Fresh tensors every call, so callers may keep them (delay queue)."""
        return list(torch.autograd.grad(self.loss(model, batch), list(model.parameters())))

    def share_memory(self) -> "BaseTask":
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value.share_memory_()
        return self

    def conflict_frac(self, pairs: int = 20000, seed: int = 0) -> float:
        """Probability that two random training examples touch a common
        coordinate. 1.0 for every dense model; overridden by the sparse task."""
        return 1.0

    # -- helpers ----------------------------------------------------------- #

    def _evaluate(self, model: torch.nn.Module, n: int | None) -> tuple[float, float]:
        raise NotImplementedError

    def _model_with(self, params) -> torch.nn.Module:
        if self._eval_model is None:
            self._eval_model = self.model()
        with torch.no_grad():
            for dst, src in zip(self._eval_model.parameters(), params):
                dst.copy_(src)
        return self._eval_model

    def __getstate__(self) -> dict:
        # Workers never evaluate; do not ship the private eval model to them.
        state = dict(self.__dict__)
        state["_eval_model"] = None
        return state


def _batches(n: int, seed: int, batch_size: int) -> Iterator[torch.Tensor]:
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    for i in range(0, n - batch_size + 1, batch_size):
        yield perm[i : i + batch_size]


# --------------------------------------------------------------------------- #
# Dense classification tasks (T2, T3a, T3b)
# --------------------------------------------------------------------------- #


class DenseClassification(BaseTask):
    """x_train / x_test are float32 and already normalised; labels are int64."""

    eval_batch = 2000

    def __init__(self, x_train, y_train, x_test, y_test) -> None:
        super().__init__()
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.n_train = x_train.shape[0]

    def train_loader(self, seed: int, batch_size: int) -> Iterable:
        for idx in _batches(self.n_train, seed, batch_size):
            yield self.x_train[idx], self.y_train[idx]

    def loss(self, model, batch) -> torch.Tensor:
        x, y = batch
        return torch.nn.functional.cross_entropy(model(x), y)

    def _evaluate(self, model, n) -> tuple[float, float]:
        n = n or self.x_test.shape[0]
        loss_sum, correct = 0.0, 0
        for i in range(0, n, self.eval_batch):
            j = min(i + self.eval_batch, n)
            logits = model(self.x_test[i:j])
            loss_sum += torch.nn.functional.cross_entropy(logits, self.y_test[i:j], reduction="sum").item()
            correct += (logits.argmax(1) == self.y_test[i:j]).sum().item()
        return loss_sum / n, correct / n


def _load_mnist():
    d = torch.load(_require("mnist.pt"))
    norm = lambda x: ((x.float() / 255.0) - 0.1307) / 0.3081
    return norm(d["x_train"]).flatten(1), d["y_train"], norm(d["x_test"]).flatten(1), d["y_test"]


class MnistSoftmax(DenseClassification):
    name, ref_epochs, fallback_eta = "t2", 10, 0.01

    def __init__(self) -> None:
        super().__init__(*_load_mnist())

    def model(self):
        return models.Softmax()


class MnistMLP(DenseClassification):
    name, ref_epochs, fallback_eta = "t3a", 10, 0.01

    def __init__(self) -> None:
        super().__init__(*_load_mnist())

    def model(self):
        return models.MLP()


class CifarCNN(DenseClassification):
    name, ref_epochs, fallback_eta = "t3b", 20, 0.01
    eval_batch = 250          # larger batches thrash the cache on one thread
    curve_eval_n = 2000       # a full 10k pass takes several seconds on one core
    # The target here is the bottom of an overfitting curve. In the pilot,
    # HOGWILD! at m = 4 bottomed out 0.06 higher than serial and so never
    # crossed the strict target, which would leave the T3b panel of the
    # primary figure empty. 10% slack keeps time-to-target a measurement.
    target_slack = 0.10

    def __init__(self) -> None:
        d = torch.load(_require("cifar10.pt"))
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        norm = lambda x: ((x.permute(0, 3, 1, 2).float() / 255.0) - mean) / std   # NHWC uint8 → NCHW
        super().__init__(norm(d["x_train"]), d["y_train"], norm(d["x_test"]), d["y_test"])

    def model(self):
        return models.SmallCNN()


# --------------------------------------------------------------------------- #
# T1: sparse logistic regression on RCV1
# --------------------------------------------------------------------------- #


class Rcv1LogReg(BaseTask):
    """CCAT vs. rest, 47,236 tf-idf features, about 76 non-zeros per document.

    Split: as in the HOGWILD! paper we train on the large part (781,265 docs,
    scikit-learn's "test" subset) and test on the small one (23,149 docs).

    l2 term: applied only to the coordinates the minibatch touches,
        loss = mean_i [ logloss(w·x_i, y_i) + (l2 / 2) * sum_{j in x_i} w_j^2 ].
    A dense (l2/2)||w||^2 would make every update touch all 47k coordinates
    and the task would stop being sparse. The reported test loss is the plain
    log-loss, without the penalty.
    """

    name, ref_epochs, fallback_eta = "t1", 5, 0.5
    l2 = 1e-5

    def __init__(self) -> None:
        super().__init__()
        d = torch.load(_require("rcv1.pt"))
        self.num_features: int = d["num_features"]
        self.ptr, self.idx, self.val, self.y = d["train_ptr"], d["train_idx"], d["train_val"], d["train_y"]
        self.n_train = self.y.shape[0]
        # Test set kept in COO form so evaluation is one index_add.
        t_ptr = d["test_ptr"]
        self.test_cols, self.test_val, self.test_y = d["test_idx"].long(), d["test_val"], d["test_y"]
        self.test_rows = torch.repeat_interleave(torch.arange(t_ptr.shape[0] - 1), t_ptr[1:] - t_ptr[:-1])

    def model(self):
        return models.LogReg(self.num_features)

    def _gather(self, rows: torch.Tensor):
        """CSR rows → COO batch (cols, vals, row-in-batch), without a Python loop over rows."""
        if rows.numel() == 1:
            r = int(rows)
            a, b = int(self.ptr[r]), int(self.ptr[r + 1])
            return self.idx[a:b].long(), self.val[a:b], torch.zeros(b - a, dtype=torch.long)
        starts = self.ptr[rows]
        counts = self.ptr[rows + 1] - starts
        local = torch.repeat_interleave(torch.arange(rows.numel()), counts)
        first = torch.cumsum(counts, 0) - counts                      # offset of each row in the batch
        pos = torch.arange(int(counts.sum())) - first[local] + starts[local]
        return self.idx[pos].long(), self.val[pos], local

    def train_loader(self, seed: int, batch_size: int) -> Iterable:
        for rows in _batches(self.n_train, seed, batch_size):
            cols, vals, local = self._gather(rows)
            yield cols, vals, local, self.y[rows]

    def loss(self, model, batch) -> torch.Tensor:
        cols, vals, local, y = batch
        z = model(cols, vals, local, y.shape[0])
        fit = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
        return fit + 0.5 * self.l2 * (model.weight[cols] ** 2).sum() / y.shape[0]

    def grads(self, model, batch) -> list[torch.Tensor]:
        """Closed-form gradient of `loss`, returned as a sparse COO vector so the
        update writes only the touched coordinates. Checked against autograd in
        the test suite. (Autograd on this model costs ~10x more per step.)"""
        cols, vals, local, y = batch
        n = y.shape[0]
        w = model.weight.detach()[cols]                 # read: only the touched coordinates
        z = torch.zeros(n).index_add_(0, local, w * vals)
        resid = (torch.sigmoid(z) - y) / n
        g = resid[local] * vals + (self.l2 / n) * w
        return [torch.sparse_coo_tensor(cols.unsqueeze(0), g, (self.num_features,), check_invariants=False)]

    def _evaluate(self, model, n) -> tuple[float, float]:
        n = self.test_y.shape[0]            # always the full test set: one sparse product
        z = model(self.test_cols, self.test_val, self.test_rows, n)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, self.test_y).item()
        acc = ((z > 0).float() == self.test_y).float().mean().item()
        return loss, acc

    def conflict_frac(self, pairs: int = 20000, seed: int = 0) -> float:
        g = torch.Generator().manual_seed(seed)
        a = torch.randint(self.n_train, (pairs,), generator=g)
        b = torch.randint(self.n_train, (pairs,), generator=g)
        hits = 0
        for i, j in zip(a.tolist(), b.tolist()):
            si = self.idx[int(self.ptr[i]) : int(self.ptr[i + 1])]
            sj = self.idx[int(self.ptr[j]) : int(self.ptr[j + 1])]
            hits += bool(torch.isin(si, sj).any())
        return hits / pairs


# --------------------------------------------------------------------------- #
# Synthetic stand-in (smoke tests)
# --------------------------------------------------------------------------- #


class StubTask(DenseClassification):
    """Small synthetic 4-class problem. Needs no downloads and trains in well
    under a second; exists so every mode can be tested without real data."""

    name, ref_epochs, fallback_eta = "stub", 2, 0.05

    def __init__(self, n: int = 4096, d: int = 32, classes: int = 4) -> None:
        g = torch.Generator().manual_seed(0)       # same data in every process
        w = torch.randn(d, classes, generator=g)
        x = torch.randn(n + n // 4, d, generator=g)
        y = (x @ w + 0.5 * torch.randn(n + n // 4, classes, generator=g)).argmax(1)
        self._d, self._classes = d, classes
        super().__init__(x[:n], y[:n], x[n:], y[n:])
        if self.target_loss != self.target_loss:   # NaN: no reference entry for the stub
            self.target_loss = 0.5

    def model(self):
        return models.Softmax(self._d, self._classes)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

TASKS = {"t1": Rcv1LogReg, "t2": MnistSoftmax, "t3a": MnistMLP, "t3b": CifarCNN, "stub": StubTask}


def get_task(name: str) -> BaseTask:
    try:
        return TASKS[name]()
    except KeyError:
        raise ValueError(f"unknown task {name!r}; expected one of {sorted(TASKS)}") from None


# --------------------------------------------------------------------------- #
# Download + conversion
# --------------------------------------------------------------------------- #


def _require(filename: str) -> Path:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing — run `python -m hogwild.tasks download` first")
    return path


def download(which: list[str] | None = None) -> None:
    """Fetch the raw datasets and convert each into one compact tensor file.

    data/mnist.pt, data/cifar10.pt   uint8 images + int64 labels (via torchvision)
    data/rcv1.pt                     CSR arrays, float32 values, int32 column ids (via scikit-learn)
    """
    which = which or ["mnist", "cifar10", "rcv1"]
    DATA_DIR.mkdir(exist_ok=True)
    raw = DATA_DIR / "raw"

    if "mnist" in which and not (DATA_DIR / "mnist.pt").exists():
        from torchvision.datasets import MNIST

        tr, te = MNIST(raw, train=True, download=True), MNIST(raw, train=False, download=True)
        torch.save({"x_train": tr.data, "y_train": tr.targets, "x_test": te.data, "y_test": te.targets},
                   DATA_DIR / "mnist.pt")
        print("wrote data/mnist.pt", tuple(tr.data.shape), tuple(te.data.shape))

    if "cifar10" in which and not (DATA_DIR / "cifar10.pt").exists():
        from torchvision.datasets import CIFAR10

        tr, te = CIFAR10(raw, train=True, download=True), CIFAR10(raw, train=False, download=True)
        torch.save({"x_train": torch.from_numpy(tr.data), "y_train": torch.tensor(tr.targets),
                    "x_test": torch.from_numpy(te.data), "y_test": torch.tensor(te.targets)},
                   DATA_DIR / "cifar10.pt")
        print("wrote data/cifar10.pt", tr.data.shape, te.data.shape)

    if "rcv1" in which and not (DATA_DIR / "rcv1.pt").exists():
        from sklearn.datasets import fetch_rcv1

        out = {}
        # Swapped on purpose (see Rcv1LogReg): scikit-learn's big "test" subset is our training set.
        for ours, theirs in (("train", "test"), ("test", "train")):
            bunch = fetch_rcv1(data_home=str(raw), subset=theirs)
            ccat = list(bunch.target_names).index("CCAT")
            x = bunch.data.tocsr()
            x.sort_indices()
            out[f"{ours}_ptr"] = torch.from_numpy(x.indptr.astype("int64"))
            out[f"{ours}_idx"] = torch.from_numpy(x.indices.astype("int32"))
            out[f"{ours}_val"] = torch.from_numpy(x.data.astype("float32"))
            out[f"{ours}_y"] = torch.from_numpy(bunch.target[:, ccat].toarray().ravel().astype("float32"))
            out["num_features"] = x.shape[1]
            print(f"rcv1 {ours}: {x.shape[0]} docs, {x.nnz} non-zeros")
        torch.save(out, DATA_DIR / "rcv1.pt")
        print("wrote data/rcv1.pt")


def info() -> None:
    for name in TASKS:
        try:
            task = get_task(name)
        except FileNotFoundError as e:
            print(f"{name:5s} not downloaded ({e})")
            continue
        n_params = sum(p.numel() for p in task.model().parameters())
        print(f"{name:5s} n_train={task.n_train:7d} params={n_params:7d} ref_epochs={task.ref_epochs:2d} "
              f"default_eta={task.default_eta:g} target_loss={task.target_loss:.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "download":
        download(sys.argv[2:] or None)
    elif cmd == "info":
        info()
    else:
        sys.exit("usage: python -m hogwild.tasks {download [mnist cifar10 rcv1] | info}")
