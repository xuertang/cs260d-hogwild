"""Serial step-size sweep and reference curves → configs/reference.yaml.

    python -m hogwild.tune --task t2 --etas 0.0001,0.0003,0.001
    python -m hogwild.tune --task t3b --etas 0.003,0.01,0.02,0.05 --probe-epochs 10

For one task:
  1. Probe: run serial SGD at every candidate eta on the task's VALIDATION
     variant ("t2@val": 10% of the training set held out, seed 0; see
     tasks.py) for --probe-epochs epochs (default: the task's full reference
     length), side by side on separate cores.
  2. Pick the eta with the lowest validation score (defined below).
  3. Reference: run the winner on the full training set for the reference
     length. This run is the task's serial reference curve, and the only place
     the test set is used: its test loss gives target_loss.
  4. Write default_eta and target_loss into configs/reference.yaml.

Selecting on a validation holdout rather than on test loss is Andrew's
protocol; it keeps the test set out of every tuning decision.

Every run is appended to results/reference.jsonl and reused on the next call
(run IDs are deterministic), so widening a grid only runs the new values. The
cache is keyed on the config only: after changing a task's model, objective or
data, pass --fresh (or delete that task's rows) so stale runs are not reused.

Score (on validation loss) and target_loss (on test loss): the lowest per-epoch
MEDIAN loss within the window.

  - Median over the evaluation points of one epoch, not a single point: with a
    constant step and small batches the curve is noisy and one point would
    make the target a matter of luck.
  - Lowest epoch in the window, not the last one. The README defines the
    target as "serial loss @ N epochs", which assumes the loss is still
    falling at N. For T1, T2 and T3a it is, and the two definitions agree
    (to within 2% on T3a). The CIFAR CNN overfits: its test loss bottoms out
    around epoch 9 and then RISES, so "loss @ 20 epochs" would be a target that
    every run crosses on its way down in the first epoch. reference.yaml
    records the epoch of the minimum as target_epoch.
  - Times (1 + the task's target_slack). The slack is 0 everywhere except
    T3b (0.10, see tasks.CifarCNN). reference.yaml keeps the unslackened
    value as serial_best_loss next to target_loss and target_slack.

"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from hogwild.schema import Config, ResultRow, append_row, read_rows
from hogwild.tasks import REFERENCE_PATH, get_task, load_reference

RESULTS = Path("results/reference.jsonl")
_append_lock = threading.Lock()


def epoch_medians(row: ResultRow, epochs: int) -> list[float]:
    """Median test loss over the evaluation points of each epoch 1..epochs."""
    out = []
    for k in range(1, epochs + 1):
        pts = [loss for e, loss in row.curve_epoch if k - 1 < e <= k]
        out.append(statistics.median(pts) if pts and all(math.isfinite(l) for l in pts) else float("inf"))
    return out


def best_loss(row: ResultRow, epochs: int) -> tuple[float, int]:
    """(lowest per-epoch median test loss, the epoch it occurs in). inf if the run diverged."""
    med = epoch_medians(row, epochs)
    if not math.isfinite(row.curve_epoch[-1][1]) or len(row.curve_epoch) < 4 * epochs:
        return float("inf"), 0          # diverged, or stopped early because it did
    k = min(range(epochs), key=med.__getitem__)
    return med[k], k + 1


def serial_run(task: str, eta: float, epochs: int, batch_size: int, seed: int = 0, fresh: bool = False) -> ResultRow:
    from hogwild.runner import make_run_id

    cfg = Config(task=task, mode="serial", m=1, eta=eta, seed=seed, batch_size=batch_size, max_epochs=epochs)
    run_id = make_run_id(cfg)
    if RESULTS.exists() and not fresh:
        cached = [r for r in read_rows(RESULTS) if r.run_id == run_id]
        if cached:
            return cached[-1]
    tmp = Path(f"results/.tune.{run_id}.tmp")
    tmp.parent.mkdir(exist_ok=True)
    tmp.unlink(missing_ok=True)
    cmd = [sys.executable, "-m", "hogwild.runner", "--json", json.dumps(cfg.to_dict()), "--out", str(tmp), "--quiet"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"serial run failed for eta={eta}:\n{proc.stderr[-2000:]}")
    row = ResultRow.from_json(tmp.read_text())
    tmp.unlink()
    with _append_lock:
        append_row(RESULTS, row)
    return row


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m hogwild.tune", description=__doc__.split("\n")[0])
    p.add_argument("--task", required=True)
    p.add_argument("--etas", required=True, help="comma-separated candidates")
    p.add_argument("--probe-epochs", type=int, help="shorter probe; default = the task's reference length")
    p.add_argument("--parallel", type=int, default=6, help="probe runs side by side")
    p.add_argument("--fresh", action="store_true", help="ignore cached runs in results/reference.jsonl")
    args = p.parse_args(argv)

    task = get_task(args.task)
    with open(Path("configs") / f"{CONFIG_STEM[args.task]}.yaml") as f:
        batch_size = yaml.safe_load(f)["batch_size"]
    etas = sorted(float(e) for e in args.etas.split(","))
    probe_epochs = args.probe_epochs or task.ref_epochs

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        rows = list(pool.map(lambda eta: serial_run(f"{args.task}@val", eta, probe_epochs, batch_size,
                                                    fresh=args.fresh), etas))
    scores = [best_loss(r, probe_epochs) for r in rows]
    for eta, row, (score, at) in zip(etas, rows, scores):
        print(f"  eta={eta:<8g} best validation loss in {probe_epochs} ep = {score:.4f} (epoch {at})  "
              f"final val acc={row.final_test_acc:.4f}  wall={row.wall_total_s:.0f}s")
    best = min(range(len(etas)), key=lambda i: scores[i][0])
    if not math.isfinite(scores[best][0]):
        sys.exit("every candidate diverged; try smaller step sizes")
    if best in (0, len(etas) - 1) and len(etas) > 1:
        print(f"  WARNING: best eta {etas[best]:g} is at the edge of the grid; widen it")

    ref = serial_run(args.task, etas[best], task.ref_epochs, batch_size, fresh=args.fresh)
    serial_best, target_epoch = best_loss(ref, task.ref_epochs)
    target = serial_best * (1 + task.target_slack)

    reference = load_reference()
    reference[args.task] = {
        "default_eta": etas[best],
        "target_loss": round(target, 5),
        "ref_epochs": task.ref_epochs,
        "target_epoch": target_epoch,
        "target_slack": task.target_slack,
        "serial_best_loss": round(serial_best, 5),
        "batch_size": batch_size,
        "n_train": task.n_train,
        "serial_final_acc": round(ref.final_test_acc, 4),
        "serial_wall_s": round(ref.wall_total_s, 1),
        "conflict_frac": round(task.conflict_frac(), 4),
        "eta_grid": etas,
        "selected_on": f"10% validation holdout, seed 0, {probe_epochs} epochs",
        "val_best_loss": round(scores[best][0], 5),
    }
    with open(REFERENCE_PATH, "w") as f:
        f.write("# Written by `python -m hogwild.tune`. Serial reference values per task.\n")
        yaml.safe_dump(reference, f, sort_keys=True)
    print(f"{args.task}: default_eta={etas[best]:g}  target_loss={target:.4f} (best serial loss {serial_best:.4f} at epoch {target_epoch}, slack {task.target_slack:g})  "
          f"serial acc={ref.final_test_acc:.4f}  serial wall={ref.wall_total_s:.0f}s  → {REFERENCE_PATH}")


CONFIG_STEM = {"t1": "t1_rcv1_logreg", "t2": "t2_mnist_softmax", "t3a": "t3a_mnist_mlp", "t3b": "t3b_cifar_cnn"}

if __name__ == "__main__":
    main()
