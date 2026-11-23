"""Shared contracts (README §6). Fixed as of Nov 22.

Change anything in this file only via a PR that all four of us review: every
stream codes against these names.

Unlike the rest of the scaffolding, this file is real code rather than a stub,
because the contracts are already fully specified in the README. The only
additions beyond the README are the (de)serialisation helpers at the bottom of
each dataclass, which exist so that sweep.py, runner.py and analysis/ all read
and write rows the same way.

Open questions to settle at the next sync (marked OPEN below):
  1. What exactly is `params` in Task.evaluate(params)?
  2. What is the `Logger` type in Mode.run(task, cfg, log)?
  3. Is Config.task the short ID ("t2") or the config stem ("t2_mnist_softmax")?
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
    task: str                 # OPEN (3): registry key understood by tasks.py
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
        # TODO(all): decide whether to also reject b > 0 outside mode="backup"
        #            and tau_inject > 0 outside mode="hogwild", or just ignore them.

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
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

    # Headline metrics. TODO(all): agree on the "never reached target" sentinel.
    # Proposal: float("inf"), which Python's json writes as Infinity and pandas
    # reads back fine; analysis then filters on math.isinf.
    time_to_target_s: float
    epochs_to_target: float
    final_test_acc: float

    # Instrumentation (hogwild/instrument.py). Empty / NaN for modes with no staleness.
    tau_hist: dict[int, int]          # tau value → number of updates with that tau
    interference_mean: float          # mean cosine between concurrent in-flight gradients

    # Curves from the evaluator. OPEN: README fixes the element type as a
    # 2-tuple but not its meaning. Proposal: (x, test_loss), with x = epochs for
    # curve_epoch and x = seconds since start for curve_wall. If analysis also
    # wants test_acc per point, that is a schema change → PR.
    curve_epoch: list[tuple[float, float]] = field(default_factory=list)
    curve_wall: list[tuple[float, float]] = field(default_factory=list)

    param_dist_to_seed0: float = float("nan")  # reproducibility; filled in by analysis or sweep
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
    """Implemented in tasks.py (stream B). hogwild/stub_task.py is a stand-in."""

    name: str
    target_loss: float   # fixed from the serial run, so time-to-target is comparable
    default_eta: float   # tuned on serial

    def model(self) -> "torch.nn.Module":
        """A freshly initialised model. Seeding is the caller's job (torch.manual_seed)."""
        ...

    def train_loader(self, seed: int, batch_size: int) -> Iterable:
        """One epoch of (x, y) batches, shuffled deterministically by `seed`.

        OPEN: does each worker get its own loader over the full dataset (classic
        HOGWILD!, sampling with replacement across workers), or a disjoint shard?
        This changes what "one epoch" means for m > 1 — settle between A and B.
        """
        ...

    def evaluate(self, params) -> tuple[float, float]:
        """Return (test_loss, test_acc).

        OPEN (1): type of `params`. Proposal: a sequence of tensors in
        `self.model().parameters()` order — that is what the shared-memory
        runner naturally holds, and it avoids pickling a state_dict every 2 s.
        """
        ...


class Mode(Protocol):
    """Implemented in modes/*.py: serial (B), sync + backup (C), hogwild (A)."""

    def run(self, task: Task, cfg: Config, log: "Logger") -> ResultRow:
        """Train `task` under `cfg`, return the finished row. Must not write files itself:
        the caller (runner CLI or sweep) owns results/*.jsonl."""
        ...
