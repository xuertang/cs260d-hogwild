"""Instrumentation shared by every mode (stream A, README §4 "Instrumentation").

Five pieces, each a stub below:

    VersionCounter      shared int64 → tau per update
    TauHistogram        per-worker tau counts, merged into ResultRow.tau_hist
    wait_for_staleness  injected tau (E2)
    StragglerSim        per-step sleep c + Exp(lambda) (E3)
    InterferenceProbe   cosine between two workers' in-flight gradients
    Logger              the `log` argument of Mode.run

Everything here is touched from worker processes, so anything shared must live
in a `share_memory_()` tensor or a multiprocessing primitive — plain Python
attributes are per-process copies after fork/spawn.

STUB: signatures and comments only. Nothing here is implemented yet.
"""

from __future__ import annotations

import torch

# --------------------------------------------------------------------------- #
# tau per update
# --------------------------------------------------------------------------- #


class VersionCounter:
    """Global update counter. tau = v_write - v_read (README §4).

    Worker protocol, per step:
        v_read  = version.read()        # before copying params
        ... copy params, compute gradient ...
        v_write = version.read()        # just before the in-place add
        ... add ...
        version.bump()                  # after the add

    DESIGN DECISION (pick one, document it in Appendix A):
      (a) Plain shared int64 tensor, `+= 1` with no lock. Matches the lock-free
          spirit and costs nothing, but read-modify-write races lose
          increments, so tau is biased LOW exactly when contention is highest.
      (b) `multiprocessing.Value("q")` with its lock held only for the
          increment. Exact tau, but adds a (tiny) serialisation point to a run
          that is supposed to have none — measure the throughput cost at m=16.
      (c) One counter per worker (no sharing, no race); the global version is
          the sum over workers. Exact and lock-free; the read is m loads.
    (c) looks best but confirm that the sum read is cheap next to a gradient.
    """

    def __init__(self, num_workers: int) -> None:
        # TODO: allocate the shared storage for whichever option is chosen.
        raise NotImplementedError

    def read(self) -> int:
        """Current global version. Called twice per step by every worker."""
        raise NotImplementedError

    def bump(self, worker_id: int) -> None:
        """Record that one update has been written."""
        raise NotImplementedError


class TauHistogram:
    """tau → count, accumulated locally in each worker and merged at the end.

    Keep it process-local (a dict or a fixed-size int64 array) and ship it back
    through the result queue when the worker exits: a shared histogram would be
    one more thing the workers contend on.
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def record(self, tau: int) -> None:
        raise NotImplementedError

    @staticmethod
    def merge(parts: list["TauHistogram"]) -> dict[int, int]:
        """Combine per-worker histograms into the dict stored in ResultRow.tau_hist."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Injected tau (E2: tau in {8, 32, 128} at m = 8)
# --------------------------------------------------------------------------- #


def wait_for_staleness(version: VersionCounter, v_read: int, k: int) -> int:
    """Spin until `version.read() - v_read >= k`, then return the version seen.

    README: "worker waits until the version counter has advanced >= K before
    writing".

    WARNING — as literally specified this deadlocks. If every worker is waiting
    for K other updates, nobody is writing, so the counter never advances. With
    m = 8 workers at most m - 1 = 7 updates can land while one worker waits, so
    K = 8, 32 and 128 (all of E2) can never be satisfied.

    Options to resolve before the pilot gate:
      (a) Delay queue: each worker keeps a FIFO of its own pending gradients
          and applies the one computed ~K versions ago while it keeps computing
          new ones. Never blocks; staleness is K by construction. This is the
          standard way to simulate delay and probably what we want.
      (b) Stale read instead of late write: each worker reads from a snapshot
          of the params that is K versions old (ring buffer of snapshots in the
          main process). Costs memory = K x model size; fine at 0.3M params.
      (c) Spin with a timeout and write anyway. Simple, but the achieved tau is
          then not K — it must be reported from the histogram, not the config.
    Whichever is chosen, E2 plots should use measured tau from TauHistogram.

    TODO: implement the chosen option; this signature may need to change.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Straggler simulation (E3)
# --------------------------------------------------------------------------- #


class StragglerSim:
    """Per-step sleep of c + Exp(lambda) seconds (README §4).

    Presets (README): "mild" = tail 2x median, "heavy" = tail 6x median, the
    99.5-percentile / max ratios measured in Lecture 4.

    Calibration sketch, taking "tail" = 99.5th percentile and s = 1/lambda:
        median = c + s ln 2        p99.5 = c + s ln 200
        ratio r = p99.5 / median   →   s / c = (r - 1) / (ln 200 - r ln 2)
        mild  (r = 2):  s / c ≈ 0.256
        heavy (r = 6):  s / c ≈ 4.39
    TODO: check the lecture's definition of "tail" (p99.5 vs. max) before
          trusting these numbers, and pick c. c should be small next to one
          gradient step so that straggler="none" and "mild" stay comparable —
          or should c scale with the measured per-step compute time?

    Each worker needs its own RNG stream (seed from cfg.seed and worker_id);
    forked workers that share a seed all sleep in lockstep.
    """

    def __init__(self, kind: str, seed: int, worker_id: int) -> None:
        # kind is Config.straggler: "none" | "mild" | "heavy".
        raise NotImplementedError

    def sleep(self) -> float:
        """Sleep for one sampled delay and return it (0.0 when kind == "none")."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Interference probe (dense stand-in for rho)
# --------------------------------------------------------------------------- #


class InterferenceProbe:
    """Every N steps, two workers dump their in-flight gradient; log the cosine.

    Shape of the mechanism:
      - two shared flat buffers of size num_params (one per probing worker) plus
        a shared step-stamp for each;
      - workers 0 and 1 copy their flattened gradient into their buffer every
        `every` local steps;
      - the main process (evaluator loop) reads both buffers when the stamps
        are close and appends cos(g0, g1) to a list.

    TODO: decide what "concurrent" means — same global version window? The two
          dumps are not synchronised, so record both stamps and let analysis
          filter on |v0 - v1|.
    TODO: for m = 1 (serial) and for sync modes there is no concurrent pair;
          return NaN for ResultRow.interference_mean.
    """

    def __init__(self, num_params: int, every: int = 100) -> None:
        raise NotImplementedError

    def maybe_dump(self, worker_id: int, step: int, grads: list[torch.Tensor]) -> None:
        """Called by every worker each step; a no-op unless this worker is a prober and it is time."""
        raise NotImplementedError

    def sample(self) -> float | None:
        """Called from the main process: cosine of the latest pair, or None if not fresh."""
        raise NotImplementedError

    def mean(self) -> float:
        """Value for ResultRow.interference_mean."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Logger
# --------------------------------------------------------------------------- #


class Logger:
    """The `log` argument of Mode.run(task, cfg, log).

    The README names the type but does not define it (see OPEN (2) in
    schema.py). Proposal: a thin object the runner creates in the main process
    and the Mode uses to accumulate everything that ends up in a ResultRow,
    plus human-readable progress on stderr.

    It lives in the main process only. Workers never call it directly — they
    send what they have (tau histogram, step counts) back over a queue.
    """

    def __init__(self, run_id: str, verbose: bool = True) -> None:
        raise NotImplementedError

    def eval_point(self, epoch: float, wall_s: float, test_loss: float, test_acc: float) -> None:
        """Record one evaluator snapshot → curve_epoch and curve_wall."""
        raise NotImplementedError

    def info(self, msg: str) -> None:
        """Progress line for whoever is watching the terminal / the VM log."""
        raise NotImplementedError
