"""Instrumentation shared by every mode (README §4 "Instrumentation").

    VersionCounter      shared per-worker counters → tau per update
    TauHistogram        per-worker tau counts, merged into ResultRow.tau_hist
    DelayQueue          injected staleness (E2)
    StragglerSim        per-step delay c + Exp(lambda) (E3)
    InterferenceProbe   cosine between two workers' in-flight gradients
    Logger              the `log` argument of Mode.run; collects the curves
    first_crossing      time / epochs at which a curve first reaches the target

Anything shared between processes lives in a `share_memory_()` tensor. Plain
Python attributes are per-process copies after the workers are spawned, which
is exactly what the local accumulators below rely on.
"""

from __future__ import annotations

import math
import sys
import time
from collections import Counter, deque

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# tau per update
# --------------------------------------------------------------------------- #


class VersionCounter:
    """Global count of updates, stored as one int64 slot per worker.

    Worker protocol, per step (README §4):
        v_read  = version.read()        # before reading params
        ... compute gradient ...
        v_write = version.read()        # just before the in-place add
        ... add ...
        version.bump(worker_id)         # after the add
        tau = v_write - v_read

    Why per-worker slots: a single shared counter incremented without a lock
    loses increments under contention (so tau would read low exactly when it
    matters), and a locked counter would put a serialisation point into a run
    that is supposed to have none. Here each worker writes only its own slot,
    so nothing is lost and nothing is locked; the global version is the sum.
    The sum is not an atomic snapshot — it can be off by updates that land
    during the read, i.e. by less than the number of workers.

    The same class counts examples processed (bump by the batch size).
    """

    def __init__(self, num_workers: int) -> None:
        self.slots = torch.zeros(num_workers, dtype=torch.int64).share_memory_()
        self._view: np.ndarray | None = None

    def _np(self) -> np.ndarray:
        if self._view is None:          # numpy view: ~10x cheaper per call than tensor ops
            self._view = self.slots.numpy()
        return self._view

    def read(self) -> int:
        return int(self._np().sum())

    def bump(self, worker_id: int, by: int = 1) -> None:
        self._np()[worker_id] += by

    def __getstate__(self) -> dict:
        return {"slots": self.slots, "_view": None}


class TauHistogram:
    """tau → count, kept local to one worker and merged when the run ends."""

    def __init__(self) -> None:
        self.counts: Counter[int] = Counter()

    def record(self, tau: int) -> None:
        self.counts[tau] += 1

    @staticmethod
    def merge(parts: list[dict[int, int]]) -> dict[int, int]:
        total: Counter[int] = Counter()
        for part in parts:
            total.update(part)
        return dict(sorted(total.items()))


def tau_mean(hist: dict[int, int]) -> float:
    n = sum(hist.values())
    return sum(t * c for t, c in hist.items()) / n if n else float("nan")


# --------------------------------------------------------------------------- #
# Injected tau (E2)
# --------------------------------------------------------------------------- #


class DelayQueue:
    """Holds each gradient back for `delay` of the worker's own steps.

    The README describes injection as "wait until the version counter has
    advanced >= K before writing". Taken literally that cannot run: while one
    worker waits, at most m - 1 other updates can land, so no worker ever sees
    the counter advance by K >= m (and E2 uses K = 8, 32, 128 at m = 8).

    A delay queue gives the same effect without blocking. Each worker keeps
    computing gradients and applies the one it computed `delay` steps ago. With
    m workers interleaving, a gradient that waits d local steps is about d * m
    global versions old, so for a target of K we use

        delay = ceil(K / m)        →   tau ≈ K + natural staleness.

    The achieved tau is whatever the histogram says; reports must use the
    measured value, not K.
    """

    def __init__(self, tau_inject: int, num_workers: int) -> None:
        self.delay = math.ceil(tau_inject / num_workers) if tau_inject > 0 else 0
        self._fifo: deque = deque()

    def push(self, item):
        """Queue the newest item; return the one that is now due, or None while filling."""
        if self.delay == 0:
            return item
        self._fifo.append(item)
        return self._fifo.popleft() if len(self._fifo) > self.delay else None


# --------------------------------------------------------------------------- #
# Straggler simulation (E3)
# --------------------------------------------------------------------------- #

# c in seconds: the fixed part of the injected per-step delay. 200 us is about
# four gradient steps on T2 (50 us each on the sweep machine).
STRAGGLER_BASE_S = 200e-6

# Tasks whose gradient step is much longer than the default c get their own, or
# the injected delay disappears next to the step itself. T3b: one step takes
# about 2.5 ms at m = 8 (batch 8), so c is one step. c is NOT part of Config or
# of the run_id: changing a value here invalidates that task's straggler rows.
STRAGGLER_BASE_BY_TASK = {"t3b": 2.5e-3}


def straggler_base_s(task: str = "") -> float:
    """c for a task name (a "@val" suffix is ignored); the default for unlisted tasks."""
    return STRAGGLER_BASE_BY_TASK.get(task.partition("@")[0], STRAGGLER_BASE_S)

# s / c, where s = 1/lambda is the mean of the exponential part. Chosen so that
# p99.5 / median of (c + Exp) equals the README's tail factor r:
#     median = c + s ln 2,  p99.5 = c + s ln 200   →   s/c = (r - 1) / (ln 200 - r ln 2)
STRAGGLER_RATIO = {"none": None, "mild": 0.2556, "heavy": 4.3905}   # r = 2 and r = 6


class StragglerSim:
    """Per-step delay of c + Exp(lambda) (README §4).

    The delay is a busy-wait, not a sleep: OS sleeps are only accurate to
    ~100 µs, the same size as c, and a slow machine is busy rather than idle.
    Long delays sleep for most of the interval and spin for the last
    millisecond so the tail does not burn a core for nothing.

    Each worker has its own generator, seeded from (seed, worker_id): workers
    sharing a stream would all be slow at the same moments.

    c depends on the task (straggler_base_s); the tail factor does not, because
    the exponential part scales with c.
    """

    def __init__(self, kind: str, seed: int, worker_id: int, task: str = "") -> None:
        ratio = STRAGGLER_RATIO[kind]
        self.active = ratio is not None
        self.base = straggler_base_s(task)
        self.scale = (ratio or 0.0) * self.base
        self.rng = np.random.default_rng([seed, worker_id, 7919])

    def sleep(self) -> float:
        if not self.active:
            return 0.0
        delay = self.base + self.rng.exponential(self.scale)
        deadline = time.perf_counter() + delay
        if delay > 2e-3:
            time.sleep(delay - 1e-3)
        while time.perf_counter() < deadline:
            pass
        return delay


# --------------------------------------------------------------------------- #
# Interference probe (dense stand-in for rho)
# --------------------------------------------------------------------------- #


class InterferenceProbe:
    """Cosine between gradients that workers 0 and 1 have in flight at the same time.

    The global version is cut into windows of `every` updates. The first
    gradient a probing worker computes in each window is copied into its shared
    buffer and stamped with the window number. If the other worker's buffer
    carries the same stamp, the two gradients were computed within one window
    of each other — i.e. concurrently, up to `every` updates — and their
    cosine is recorded. The worker that dumps second does the comparison, so
    the main process is not involved and no worker ever waits.

    Caveats, both small: the other buffer may be mid-write when it is read (a
    torn gradient, same as everything else here), and if both workers dump at
    the same instant the pair is counted twice.

    For sparse T1 the cosine is 0 whenever the two examples share no feature;
    the mean is then the sparse-case overlap, which is what rho measures.
    """

    def __init__(self, num_params: int, every: int = 200) -> None:
        self.buf = torch.zeros(2, num_params).share_memory_()
        self.stamp = torch.full((2,), -1, dtype=torch.int64).share_memory_()
        self.every = every
        self._last = -1        # process-local from here down
        self._sum = 0.0
        self._n = 0

    def maybe_dump(self, worker_id: int, key: int, grads: list[torch.Tensor]) -> None:
        """`key` is the version (or round) the gradient was computed at."""
        if worker_id > 1:
            return
        window = key // self.every
        if window == self._last:
            return
        self._last = window
        flat = torch.cat([(g.to_dense() if g.is_sparse else g).reshape(-1) for g in grads])
        self.buf[worker_id].copy_(flat)
        self.stamp[worker_id] = window
        other = 1 - worker_id
        if int(self.stamp[other]) == window:
            cos = torch.nn.functional.cosine_similarity(flat, self.buf[other], dim=0).item()
            if math.isfinite(cos):
                self._sum += cos
                self._n += 1

    def local_stats(self) -> tuple[float, int]:
        return self._sum, self._n

    @staticmethod
    def mean(parts: list[tuple[float, int]]) -> float:
        n = sum(c for _, c in parts)
        return sum(s for s, _ in parts) / n if n else float("nan")


# --------------------------------------------------------------------------- #
# Logger
# --------------------------------------------------------------------------- #


class Logger:
    """The `log` argument of Mode.run(task, cfg, log).

    Lives in the main process. It collects the evaluator's points (which become
    curve_epoch / curve_wall) and carries the few per-run settings that are not
    part of Config: where to stop and how often to evaluate. Keeping those here
    leaves the Mode.run signature exactly as the README fixes it.
    """

    def __init__(self, run_id: str, epoch_limit: float, eval_period_s: float = 2.0, verbose: bool = True) -> None:
        self.run_id = run_id
        self.epoch_limit = epoch_limit        # cfg.max_epochs, or a fraction of an epoch for --smoke
        self.eval_period_s = eval_period_s
        self.verbose = verbose
        self.points: list[tuple[float, float, float, float]] = []   # (epoch, wall_s, test_loss, test_acc)
        self.final_params: list[torch.Tensor] | None = None         # set by the runner when training ends

    def eval_point(self, epoch: float, wall_s: float, test_loss: float, test_acc: float) -> None:
        self.points.append((epoch, wall_s, test_loss, test_acc))
        self.info(f"epoch {epoch:6.2f}  {wall_s:7.1f}s  loss {test_loss:.4f}  acc {test_acc:.4f}")

    def info(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.run_id}] {msg}", file=sys.stderr, flush=True)

    @property
    def curve_epoch(self) -> list[tuple[float, float]]:
        return [(e, loss) for e, _, loss, _ in self.points]

    @property
    def curve_wall(self) -> list[tuple[float, float]]:
        return [(w, loss) for _, w, loss, _ in self.points]


def first_crossing(points: list[tuple[float, float, float, float]], target: float) -> tuple[float, float]:
    """(wall_s, epochs) at which test loss first reaches `target`.

    Linear interpolation between the last point above the target and the first
    point at or below it, so the answer is not quantised to the evaluation
    period. (inf, inf) if the target is never reached or is NaN (not tuned).
    """
    prev = None
    for epoch, wall, loss, _ in points:
        if loss <= target:
            if prev is None or not math.isfinite(prev[2]) or prev[2] <= loss:
                return wall, epoch
            f = (prev[2] - target) / (prev[2] - loss)
            return prev[1] + f * (wall - prev[1]), prev[0] + f * (epoch - prev[0])
        prev = (epoch, wall, loss)
    return float("inf"), float("inf")
