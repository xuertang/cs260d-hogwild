"""Shared contracts (README §6). Fixed as of Nov 22.

Change anything in this file only via a PR that all four of us review:
everything else in the repo is written against these names.

Beyond the README, this file adds only the (de)serialisation helpers, so that
sweep.py, runner.py and analysis/ all read and write rows the same way.

Decisions taken on the points the README leaves open:
  1. `params` in Task.evaluate(params) is a sequence of tensors in
     `model().parameters()` order.
  2. `Logger` is hogwild.instrument.Logger.
  3. Config.task is the short ID: "t1", "t2", "t3a", "t3b" (or "stub").
  4. Tasks also provide grads(model, batch) and share_memory(); see tasks.py.
  5. Curve points are (x, test_loss), x = epochs or seconds since start.
  6. A run that never reaches the target has time_to_target_s =
     epochs_to_target = inf (written to JSON as Infinity).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Protocol, runtime_checkable

if TYPE_CHECKING:  # type-only imports: schema.py must stay importable without torch
    import torch

    from hogwild.instrument import Logger


# --------------------------------------------------------------------------- #
# Allowed values for the string-typed Config fields
# --------------------------------------------------------------------------- #

MODES = ("serial", "sync", "backup", "hogwild")
ETA_SCALES = ("const", "1/(1+tau)")
STRAGGLERS = ("none", "mild", "heavy")


# --------------------------------------------------------------------------- #
# Config — mirrors the YAML in configs/
# --------------------------------------------------------------------------- #


# kw_only=True is required, not cosmetic: the README lists `b: int = 0` before
# `eta: float`, and a positional dataclass rejects a non-default field after a
# default one. Keyword-only keeps the README's field order legal.
@dataclass(kw_only=True)
class Config:
    task: str                 # "t1" | "t2" | "t3a" | "t3b" | "stub"
    mode: str                 # one of MODES
    m: int                    # workers whose gradients count (1 for serial)
    b: int = 0                # extra backup workers; only meaningful for mode="backup"
    eta: float                # step size actually used (sweeps multiply Task.default_eta)
    eta_scale: str = "const"  # "const" or "1/(1+tau)"
    tau_inject: int = 0       # 0 = natural staleness only; K > 0 = force tau >= K
    straggler: str = "none"   # one of STRAGGLERS
    seed: int
    batch_size: int
    max_epochs: int

    def __post_init__(self) -> None:
        # Fail at load time rather than ten minutes into a sweep.
        if self.mode not in MODES:
            raise ValueError(f"mode={self.mode!r}, expected one of {MODES}")
        if self.eta_scale not in ETA_SCALES:
            raise ValueError(f"eta_scale={self.eta_scale!r}, expected one of {ETA_SCALES}")
        if self.straggler not in STRAGGLERS:
            raise ValueError(f"straggler={self.straggler!r}, expected one of {STRAGGLERS}")
        # b outside mode="backup" and tau_inject outside mode="hogwild" are ignored.

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        """Two conveniences on top of the plain fields, so that the tuned step
        size lives in one place (configs/reference.yaml):
          - eta missing or null → the task's default_eta;
          - eta_mult (not a Config field) multiplies eta — used by the E4 sweep.
        """
        d = dict(d)
        mult = float(d.pop("eta_mult", 1.0))
        if d.get("eta") is None:
            from hogwild.tasks import TASKS, load_reference   # lazy: needs torch

            d["eta"] = load_reference().get(d["task"], {}).get("default_eta", TASKS[d["task"]].fallback_eta)
        d["eta"] = float(d["eta"]) * mult
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "Config":
        """Load a configs/*.yaml file; keyword overrides win (CLI flags, sweep grid)."""
        import yaml  # local import keeps schema.py light for worker processes

        with open(path) as f:
            d = yaml.safe_load(f) or {}
        d.update(overrides)
        return cls.from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# ResultRow — one JSON line in results/*.jsonl
# --------------------------------------------------------------------------- #


@dataclass
class ResultRow:
    run_id: str
    config: Config

    # Headline metrics. inf = the target was never reached.
    time_to_target_s: float
    epochs_to_target: float
    final_test_acc: float

    # Instrumentation (hogwild/instrument.py). Empty / NaN for modes with no staleness.
    tau_hist: dict[int, int]          # tau value → number of updates with that tau
    interference_mean: float          # mean cosine between concurrent in-flight gradients

    # Curves from the evaluator: (epochs, test_loss) and (seconds, test_loss).
    curve_epoch: list[tuple[float, float]] = field(default_factory=list)
    curve_wall: list[tuple[float, float]] = field(default_factory=list)

    param_dist_to_seed0: float = float("nan")  # reproducibility; needs --params-dir
    wall_total_s: float = 0.0

    def to_json(self) -> str:
        """One line, no trailing newline. Tuples become lists, int keys become strings."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> "ResultRow":
        """Inverse of to_json: restores Config, int tau keys and tuple curve points."""
        d = json.loads(line)
        d["config"] = Config.from_dict(d["config"])
        d["tau_hist"] = {int(k): int(v) for k, v in d["tau_hist"].items()}
        d["curve_epoch"] = [tuple(p) for p in d["curve_epoch"]]
        d["curve_wall"] = [tuple(p) for p in d["curve_wall"]]
        return cls(**d)


def append_row(path: str | Path, row: ResultRow) -> None:
    """Append one row to a results/*.jsonl file (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(row.to_json() + "\n")


def read_rows(path: str | Path) -> list[ResultRow]:
    """Read every row of a results/*.jsonl file, skipping blank lines."""
    with open(path) as f:
        return [ResultRow.from_json(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# Task and Mode — structural interfaces (implementations live elsewhere)
# --------------------------------------------------------------------------- #


@runtime_checkable
class Task(Protocol):
    """Implemented in tasks.py."""

    name: str
    target_loss: float   # fixed from the serial run, so time-to-target is comparable
    default_eta: float   # tuned on serial

    def model(self) -> "torch.nn.Module":
        """A freshly initialised model. Seeding is the caller's job (torch.manual_seed)."""
        ...

    def train_loader(self, seed: int, batch_size: int) -> Iterable:
        """One pass over the training set in an order fixed by `seed`.
        Each worker calls this with its own seed and walks the full dataset."""
        ...

    def evaluate(self, params) -> tuple[float, float]:
        """(test_loss, test_acc) for `params`: tensors in model().parameters() order."""
        ...


class Mode(Protocol):
    """Implemented in modes/*.py."""

    def run(self, task: Task, cfg: Config, log: "Logger") -> ResultRow:
        """Train `task` under `cfg`, return the finished row. Must not write files itself:
        the caller (runner CLI or sweep) owns results/*.jsonl."""
        ...
