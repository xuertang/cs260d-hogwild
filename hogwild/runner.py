"""Shared-memory runner (stream A).

Owns everything the modes have in common:

    SharedParams   model parameters in shared memory, visible to all workers
    pin_worker     one worker = one core, one thread
    WorkerPool     spawn / start barrier / join / collect per-worker stats
    Evaluator      main-process snapshots every 2 s wall-clock and every epoch
    run_one        Config → ResultRow (looks up Task and Mode, wires the above)
    main           CLI

CLI (README §3):
    python -m hogwild.runner --config configs/t2_mnist_softmax.yaml --mode serial --smoke

STUB: signatures and comments only. Nothing here is implemented yet.
"""

from __future__ import annotations

import argparse
from typing import Callable

import torch

from hogwild.instrument import Logger
from hogwild.schema import Config, ResultRow, Task

# --------------------------------------------------------------------------- #
# Shared parameters
# --------------------------------------------------------------------------- #


class SharedParams:
    """The single copy of the model that every worker reads and writes.

    Simplest construction: build `task.model()` once in the main process, call
    `model.share_memory()`, and hand the model to each worker (PyTorch's own
    Hogwild example does exactly this). Tensors are then backed by shared
    memory and in-place ops in any process are visible to all.

    Things to get right:
      - Start method. "fork" shares for free but is unsafe on macOS and with
        some BLAS builds; "spawn" pickles the shared tensors by handle, which
        works on both laptop (macOS) and VM (Linux). Pick ONE and use it
        everywhere so timings are comparable. Proposal: spawn.
      - Workers must never rebind a parameter (`p = p - eta * g`); only
        in-place ops (`p.add_(g, alpha=-eta)`) write to shared memory.
      - Updates happen under torch.no_grad(); autograd must not track them.
      - README §9: adds are not coordinate-atomic. That is intended. A locked
        variant on T1 is listed as a fallback data point — leave a hook for it.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        raise NotImplementedError

    def tensors(self) -> list[torch.Tensor]:
        """Shared tensors in model.parameters() order (what Task.evaluate receives)."""
        raise NotImplementedError

    def snapshot(self) -> list[torch.Tensor]:
        """Private copy for the evaluator.

        Not atomic: workers keep writing during the copy, so a snapshot may mix
        versions across (and within) tensors. Acceptable for a loss curve; say
        so in Appendix A.
        """
        raise NotImplementedError

    def load_into(self, local_model: torch.nn.Module) -> None:
        """Copy shared → a worker's private model (the 'read' of the read/compute/write cycle).

        TODO: decide whether workers compute gradients directly on the shared
              model (true HOGWILD!: the forward pass itself sees torn writes)
              or on a private copy (cleaner tau accounting, one extra memcpy
              per step). The version-counter protocol in instrument.py assumes
              an explicit copy.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Worker pinning
# --------------------------------------------------------------------------- #


def pin_worker(core_id: int) -> None:
    """First thing every worker calls: one thread, one core (README §3).

    - torch.set_num_threads(1) — and set OMP_NUM_THREADS / MKL_NUM_THREADS = 1
      in the environment BEFORE torch is imported in the child, or the thread
      pools are already sized.
    - os.sched_setaffinity(0, {core_id}) on Linux (the VM).
    - macOS has no sched_setaffinity: skip pinning with a one-time warning.
      Laptop numbers are for development only, so that is fine.

    TODO: on the 32-vCPU VM, vCPUs are hyperthreads. Map worker i to a distinct
          PHYSICAL core (check `lscpu -e`), otherwise m = 16 is 8 cores x 2.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Worker pool
# --------------------------------------------------------------------------- #


class WorkerPool:
    """Starts n worker processes on a target function and collects what they return.

    Used by every multi-worker mode (hogwild, sync, backup); serial does not
    need it.

    Lifecycle:
        pool = WorkerPool(n, target, args)   # target(worker_id, *args) runs in the child
        pool.start()                         # children block on a start barrier
        pool.release()                       # t0 for wall-clock: all workers go at once
        ... main process runs the Evaluator loop ...
        pool.stop()                          # set shared stop flag (target reached / max_epochs)
        stats = pool.join()                  # per-worker dicts: steps, tau histogram, ...

    Notes:
      - Process start-up and torch import take seconds; the start barrier keeps
        that out of time_to_target_s.
      - The stop flag is a shared value workers poll once per step.
      - A crashed worker must not hang join(): use timeouts and check exitcode,
        and surface the child's traceback.
    """

    def __init__(self, n: int, target: Callable[..., dict], args: tuple = ()) -> None:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def join(self) -> list[dict]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #


class Evaluator:
    """Main-process loop: snapshot shared params, evaluate, log (README §4).

    Triggers: every 2 s of wall-clock AND at every epoch boundary. One run thus
    yields both loss-vs-wall-clock and loss-vs-epoch curves.

    Also decides when to stop: first snapshot with test_loss <= task.target_loss
    fixes time_to_target_s / epochs_to_target; training continues to max_epochs
    for final_test_acc unless --smoke.

    TODO: "epoch" under m workers = total examples processed by all workers /
          dataset size. Needs a shared examples-processed counter (same choice
          of mechanism as VersionCounter).
    TODO: evaluation itself takes time and competes for a core. Run it on a
          core no worker is pinned to, and resolution of time_to_target_s is
          bounded by the 2 s period — interpolate between snapshots or accept
          it? Runs are 1–5 min, so 2 s is ~1–3 %.
    TODO: T1 serial may finish in seconds; consider a shorter period for it.
    """

    def __init__(self, task: Task, shared: SharedParams, log: Logger, period_s: float = 2.0) -> None:
        raise NotImplementedError

    def step(self, epochs_done: float, wall_s: float) -> bool:
        """Evaluate if a trigger fired. Returns True once the target loss has been reached."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #


def make_run_id(cfg: Config) -> str:
    """Deterministic ID from the config, so sweep.py can resume by skipping IDs already on disk.

    Proposal: readable prefix + short hash of the sorted config dict, e.g.
    "t2-hogwild-m8-s0-3fa9c1". Must NOT include a timestamp.
    """
    raise NotImplementedError


def run_one(cfg: Config, smoke: bool = False) -> ResultRow:
    """Config → ResultRow.

    Steps:
      1. torch.manual_seed(cfg.seed); torch.set_num_threads(1) in main too.
      2. Look up the Task from cfg.task (tasks.py registry, stream B; fall back
         to hogwild.stub_task.StubTask for cfg.task == "stub").
      3. Look up the Mode with hogwild.modes.get_mode(cfg.mode).
      4. Build the Logger, call mode.run(task, cfg, log).
      5. Return the row. Writing it to disk is the caller's job
         (schema.append_row).

    smoke=True: cap at a fraction of one epoch and skip the target-loss wait —
    just prove the pipeline runs end to end.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Flags from the README, plus the Config overrides the mode table implies."""
    p = argparse.ArgumentParser(prog="python -m hogwild.runner", description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True, help="configs/*.yaml")
    p.add_argument("--mode", help="override Config.mode: serial | sync | backup | hogwild")
    p.add_argument("--m", type=int, help="override Config.m")
    p.add_argument("--tau", type=int, dest="tau_inject", help="override Config.tau_inject (hogwild --tau K)")
    p.add_argument("--seed", type=int, help="override Config.seed")
    p.add_argument("--out", help="append the ResultRow to this JSONL file (default: print only)")
    p.add_argument("--smoke", action="store_true", help="tiny run that only checks the pipeline")
    # TODO: add --b, --eta, --eta-scale, --straggler if single runs need them
    #       (sweeps set these through the YAML grid instead).
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # TODO: collect the non-None overrides, cfg = Config.from_yaml(args.config, **overrides),
    #       row = run_one(cfg, smoke=args.smoke), print a one-line summary,
    #       and append_row(args.out, row) when --out is given.
    raise NotImplementedError("runner is a stub; see the comments in hogwild/runner.py")


if __name__ == "__main__":
    main()
