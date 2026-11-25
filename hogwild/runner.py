"""Shared-memory runner: everything the modes have in common.

    Shared         what each worker process receives (task, shared model, counters, ...)
    pin_worker     one worker = one core, one thread
    train          start workers, run the evaluator loop, assemble the ResultRow
    run_one        Config → ResultRow
    main           CLI

    python -m hogwild.runner --config configs/t2_mnist_softmax.yaml --mode serial --smoke
    python -m hogwild.runner --config configs/t2_mnist_softmax.yaml --mode hogwild --m 4 --out results/pilot.jsonl

Design choices (they all go into Appendix A of the report):

  - Start method "spawn" on every platform. Fork is unsafe on macOS and with
    some BLAS builds; using one method everywhere keeps laptop and VM runs
    structurally identical. Start-up cost stays out of the timings because all
    workers wait on a barrier and the clock starts when it releases.
  - The model is built once in the main process and `share_memory()`-ed.
    Workers compute gradients directly on those shared tensors and update them
    in place — the same construction as PyTorch's own Hogwild example. A
    forward pass can therefore see a write that lands mid-computation
    (inconsistent read), as in the original algorithm.
  - The dataset is shared too: one copy in shared memory, not one per worker.
  - The evaluator is the main process. It copies the shared parameters without
    stopping anyone, so a snapshot may mix versions; fine for a loss curve.
  - An "epoch" is n_train examples applied to the model, summed over workers.
  - All timestamps use time.monotonic(), which is system-wide, so worker and
    main-process times are directly comparable.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import queue as queue_mod
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Must be set before worker processes import torch: one thread per worker,
# otherwise intra-op threading hides the parallelism we are measuring (README §3).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.multiprocessing as mp

from hogwild.instrument import InterferenceProbe, Logger, TauHistogram, VersionCounter, first_crossing, tau_mean
from hogwild.schema import Config, ResultRow, Task, append_row

SMOKE_EPOCHS = 0.25          # --smoke: a quarter of an epoch, just to prove the pipeline runs
EVAL_EPOCH_STEP = 0.25       # evaluate at least this often in epochs ...
EVAL_PERIOD_S = 2.0          # ... and at least this often in seconds (README §4)


# --------------------------------------------------------------------------- #
# What a worker receives
# --------------------------------------------------------------------------- #


@dataclass
class Shared:
    """Pickled once per worker at spawn. Tensors inside are shared-memory
    handles, so every process sees the same storage."""

    task: Any                      # BaseTask with its dataset in shared memory
    model: torch.nn.Module         # parameters in shared memory
    cfg: Config
    n_workers: int
    example_limit: int             # stop once this many examples have been applied
    version: VersionCounter        # updates applied, per worker
    examples: VersionCounter       # examples applied, per worker
    stop: torch.Tensor             # int64[1]; non-zero = stop now
    probe: InterferenceProbe
    start: Any                     # Barrier(n_workers + 1): workers + main
    extra: dict = field(default_factory=dict)   # mode-specific shared state

    def should_stop(self) -> bool:
        return bool(self.stop[0]) or self.examples.read() >= self.example_limit

    def loader_seed(self, worker_id: int, epoch: int) -> int:
        """Distinct data order per (run seed, worker, pass). Every worker walks
        the full training set in its own order, as in the original HOGWILD!."""
        return self.cfg.seed * 1_000_003 + worker_id * 10_007 + epoch


# --------------------------------------------------------------------------- #
# Worker pinning
# --------------------------------------------------------------------------- #


def core_for(worker_id: int) -> int | None:
    """Core for worker i, or None where pinning is unavailable (macOS).

    HOGWILD_CORES="0,2,4,..." gives an explicit list — use it on the VM to map
    workers to distinct physical cores (vCPUs are hyperthreads; check
    `lscpu -e`). HOGWILD_CORE_OFFSET shifts the assignment so that two runs
    started side by side by sweep.py land on disjoint cores.
    """
    if not hasattr(os, "sched_setaffinity"):
        return None
    env = os.environ.get("HOGWILD_CORES")
    cores = [int(c) for c in env.split(",")] if env else sorted(os.sched_getaffinity(0))
    offset = int(os.environ.get("HOGWILD_CORE_OFFSET", "0"))
    return cores[(offset + worker_id) % len(cores)]


def pin_worker(core: int | None) -> None:
    torch.set_num_threads(1)
    if core is not None:
        os.sched_setaffinity(0, {core})


def _worker_entry(worker_fn: Callable, worker_id: int, shared: Shared, results, core: int | None) -> None:
    """Runs in the child. Reports either the worker's stats or its traceback."""
    try:
        pin_worker(core)
        torch.manual_seed(shared.cfg.seed * 7919 + worker_id)
        shared.start.wait(timeout=600)
        stats = worker_fn(worker_id, shared)
        stats["t_end"] = time.monotonic()
        stats["probe"] = shared.probe.local_stats()
        results.put(("ok", worker_id, stats))
    except BaseException:
        shared.stop[0] = 1
        results.put(("error", worker_id, traceback.format_exc()))
        shared.start.abort()        # if we died before the start, do not leave the others waiting


# --------------------------------------------------------------------------- #
# Training loop shared by all modes
# --------------------------------------------------------------------------- #


def snapshot(model: torch.nn.Module) -> list[torch.Tensor]:
    """Private copy of the shared parameters. Not atomic: workers keep writing."""
    return [p.detach().clone() for p in model.parameters()]


def train(
    task: Task,
    cfg: Config,
    log: Logger,
    worker_fn: Callable[[int, Shared], dict],
    n_workers: int,
    extra: dict | None = None,
    on_stop: Callable[[], None] | None = None,
) -> ResultRow:
    """Run `worker_fn` in `n_workers` processes on a shared model and evaluate
    from the main process until they finish.

    worker_fn(worker_id, shared) → dict with at least {"tau": {tau: count}}.
    `extra` is mode-specific shared state (locks, gradient buffers), built by
    the caller with the context from `mp_context()`.
    `on_stop` is called once if the run is aborted, so that a mode can wake
    workers that are blocked on its own primitives.
    """
    ctx = mp_context()
    torch.set_num_threads(int(os.environ.get("HOGWILD_EVAL_THREADS", "1")))
    torch.manual_seed(cfg.seed)

    model = task.model()
    model.share_memory()
    task.share_memory()
    num_params = sum(p.numel() for p in model.parameters())

    shared = Shared(
        task=task, model=model, cfg=cfg, n_workers=n_workers,
        example_limit=int(log.epoch_limit * task.n_train),
        version=VersionCounter(n_workers), examples=VersionCounter(n_workers),
        stop=torch.zeros(1, dtype=torch.int64).share_memory_(),
        probe=InterferenceProbe(num_params),
        start=ctx.Barrier(n_workers + 1),
        extra=extra or {},
    )
    results = ctx.Queue()
    procs = [ctx.Process(target=_worker_entry, args=(worker_fn, i, shared, results, core_for(i)), daemon=True)
             for i in range(n_workers)]
    for p in procs:
        p.start()

    def evaluate(epoch: float, wall: float) -> float:
        loss, acc = task.evaluate_curve(snapshot(model))
        log.eval_point(epoch, wall, loss, acc)
        return loss

    def abort(reason: str) -> None:
        log.info(f"stopping early: {reason}")
        shared.stop[0] = 1
        if on_stop:
            on_stop()

    evaluate(0.0, 0.0)                         # starting point of both curves
    try:
        shared.start.wait(timeout=600)         # everyone is up: release and start the clock
    except threading.BrokenBarrierError:
        try:
            _, _, detail = results.get(timeout=10)
        except queue_mod.Empty:
            detail = "no worker reported an error (start-up timed out?)"
        for p in procs:
            p.terminate()
        raise RuntimeError(f"a worker failed during start-up:\n{detail}") from None
    t0 = time.monotonic()

    stats: dict[int, dict] = {}
    error: str | None = None
    next_wall, next_epoch = log.eval_period_s, EVAL_EPOCH_STEP
    while len(stats) < n_workers and error is None:
        try:
            while True:
                kind, wid, payload = results.get_nowait()
                if kind == "ok":
                    stats[wid] = payload
                else:
                    error = payload
        except queue_mod.Empty:
            pass
        if error is None and any(not p.is_alive() and p.exitcode not in (0, None) for p in procs):
            error = "a worker process died without reporting (killed or out of memory?)"
        wall = time.monotonic() - t0
        epoch = shared.examples.read() / task.n_train
        if not shared.stop[0] and (wall >= next_wall or epoch >= next_epoch) and len(stats) == 0:
            loss = evaluate(epoch, wall)
            next_wall = wall + log.eval_period_s
            next_epoch = (math.floor(epoch / EVAL_EPOCH_STEP) + 1) * EVAL_EPOCH_STEP
            if not math.isfinite(loss):
                abort("test loss is not finite (diverged)")
        time.sleep(0.002)

    if error is not None:
        abort("worker failure")
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    if error is not None:
        raise RuntimeError(f"worker failed:\n{error}")

    wall_total = max(s["t_end"] for s in stats.values()) - t0
    final_epoch = shared.examples.read() / task.n_train
    log.final_params = snapshot(model)
    log.eval_point(final_epoch, wall_total, *task.evaluate_curve(log.final_params))   # last curve point
    _, final_acc = task.evaluate(log.final_params)                                    # full test set

    t_target, e_target = first_crossing(log.points, task.target_loss)
    tau_hist = TauHistogram.merge([s["tau"] for s in stats.values()])
    row = ResultRow(
        run_id=log.run_id, config=cfg,
        time_to_target_s=t_target, epochs_to_target=e_target, final_test_acc=final_acc,
        tau_hist=tau_hist,
        interference_mean=InterferenceProbe.mean([s["probe"] for s in stats.values()]),
        curve_epoch=log.curve_epoch, curve_wall=log.curve_wall,
        wall_total_s=wall_total,
    )
    updates = sum(tau_hist.values())
    extras = {k: sum(s.get(k, 0) for s in stats.values()) for k in ("dropped",) if any(k in s for s in stats.values())}
    log.info(f"done: {final_epoch:.2f} epochs in {wall_total:.1f}s, {updates} updates "
             f"({updates / max(wall_total, 1e-9):.0f}/s), mean tau {tau_mean(tau_hist):.2f}, "
             f"interference {row.interference_mean:.4f}, final acc {final_acc:.4f}, "
             f"time to target {t_target:.1f}s" + (f", {extras}" if extras else ""))
    return row


def mp_context():
    return mp.get_context("spawn")


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #


def make_run_id(cfg: Config) -> str:
    """Readable prefix + hash of the full config. A pure function of the
    Config (no timestamp), which is what lets sweep.py resume."""
    digest = hashlib.sha1(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()[:8]
    parts = [cfg.task, cfg.mode, f"m{cfg.m}"]
    if cfg.b:
        parts.append(f"b{cfg.b}")
    if cfg.tau_inject:
        parts.append(f"tau{cfg.tau_inject}")
    if cfg.straggler != "none":
        parts.append(cfg.straggler)
    parts += [f"s{cfg.seed}", digest]
    return "-".join(parts)


def flat_params(params: list[torch.Tensor]) -> np.ndarray:
    return torch.cat([p.reshape(-1) for p in params]).numpy()


def param_dist_to_seed0(cfg: Config, params: list[torch.Tensor], params_dir: Path | None) -> float:
    """L2 distance between this run's final parameters and those of the same
    config at seed 0 (README §4, reproducibility). Needs `params_dir`, where
    every run leaves its final parameters; NaN if the seed-0 twin has not run."""
    if params_dir is None:
        return float("nan")
    params_dir.mkdir(parents=True, exist_ok=True)
    mine = flat_params(params)
    np.save(params_dir / f"{make_run_id(cfg)}.npy", mine)
    if cfg.seed == 0:
        return 0.0
    twin = params_dir / f"{make_run_id(dataclasses.replace(cfg, seed=0))}.npy"
    return float(np.linalg.norm(mine - np.load(twin))) if twin.exists() else float("nan")


def run_one(cfg: Config, smoke: bool = False, params_dir: Path | None = None, verbose: bool = True) -> ResultRow:
    """Config → ResultRow. Does not write the row; the caller owns results/*.jsonl."""
    from hogwild.modes import get_mode
    from hogwild.tasks import get_task

    task = get_task(cfg.task)
    log = Logger(
        run_id=make_run_id(cfg),
        epoch_limit=min(SMOKE_EPOCHS, cfg.max_epochs) if smoke else cfg.max_epochs,
        eval_period_s=0.5 if smoke else EVAL_PERIOD_S,
        verbose=verbose,
    )
    row = get_mode(cfg.mode).run(task, cfg, log)
    row.param_dist_to_seed0 = param_dist_to_seed0(cfg, log.final_params, params_dir)
    return row


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m hogwild.runner", description="Run one configuration.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="configs/*.yaml")
    src.add_argument("--json", help="a full Config as a JSON object (used by sweep.py)")
    p.add_argument("--mode", help="serial | sync | backup | hogwild")
    p.add_argument("--m", type=int)
    p.add_argument("--b", type=int)
    p.add_argument("--tau", type=int, dest="tau_inject", help="inject staleness of about K versions (hogwild)")
    p.add_argument("--eta", type=float)
    p.add_argument("--eta-scale", dest="eta_scale", choices=["const", "1/(1+tau)"])
    p.add_argument("--straggler", choices=["none", "mild", "heavy"])
    p.add_argument("--seed", type=int)
    p.add_argument("--max-epochs", type=int, dest="max_epochs")
    p.add_argument("--out", help="append the ResultRow to this JSONL file")
    p.add_argument("--params-dir", help="save final parameters here and fill param_dist_to_seed0")
    p.add_argument("--smoke", action="store_true", help="a quarter of an epoch, to check the pipeline")
    p.add_argument("--quiet", action="store_true")
    return p


OVERRIDES = ("mode", "m", "b", "tau_inject", "eta", "eta_scale", "straggler", "seed", "max_epochs")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    overrides = {k: getattr(args, k) for k in OVERRIDES if getattr(args, k) is not None}
    if args.json:
        cfg = Config.from_dict({**json.loads(args.json), **overrides})
    else:
        cfg = Config.from_yaml(args.config, **overrides)
    if cfg.mode != "serial" and cfg.m == 1 and "m" not in overrides:
        sys.exit(f"mode {cfg.mode!r} needs --m (the task config has m = 1 for serial)")
    row = run_one(cfg, smoke=args.smoke, params_dir=Path(args.params_dir) if args.params_dir else None,
                  verbose=not args.quiet)
    if args.out:
        append_row(args.out, row)
    print(f"{row.run_id}  wall={row.wall_total_s:.1f}s  time_to_target={row.time_to_target_s:.1f}s  "
          f"epochs_to_target={row.epochs_to_target:.2f}  final_acc={row.final_test_acc:.4f}  "
          f"mean_tau={tau_mean(row.tau_hist):.2f}")


if __name__ == "__main__":
    main()
