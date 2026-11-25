"""Recompute time-to-target for a different target loss, without rerunning.

    python -m analysis.retarget results --out results_slack10 --slack 0.10
    python -m analysis.retarget results --out results_alt --target t3b=0.82 --target t2=0.28
    python -m analysis.plots --results results_slack10 --out report/figures/slack10

Every row stores its full loss curves, so time_to_target_s and
epochs_to_target are a function of the row and a target. This rewrites every
*.jsonl in a results directory into a NEW directory (the originals are never
modified) with those two fields recomputed, using the same first-crossing rule
as the runner.

--slack S     target = (1 + S) x the task's target_loss in configs/reference.yaml
--target T=X  explicit target for one task (repeatable; wins over --slack)

Why this exists: the targets in configs/reference.yaml are the best loss the
serial run reaches. A mode whose loss curve sits slightly higher never crosses
that, and its row says inf. A small slack turns "never" back into a time.
State the slack used next to any figure made this way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from hogwild.instrument import first_crossing
from hogwild.schema import append_row, read_rows


def retarget(src: Path, dst: Path, targets: dict[str, float]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.glob("*.jsonl")):
        if path.name.endswith(".failed.jsonl"):
            continue
        out = dst / path.name
        out.unlink(missing_ok=True)
        changed = total = 0
        for row in read_rows(path):
            total += 1
            target = targets.get(row.config.task)
            if target is not None:
                # curve_epoch and curve_wall hold the same evaluation points in the same order
                points = [(e, w, loss, 0.0) for (e, loss), (w, _) in zip(row.curve_epoch, row.curve_wall)]
                t, e = first_crossing(points, target)
                changed += (t, e) != (row.time_to_target_s, row.epochs_to_target)
                row.time_to_target_s, row.epochs_to_target = t, e
            append_row(out, row)
        print(f"{path} → {out}: {total} rows, {changed} changed")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m analysis.retarget", description=__doc__.split("\n")[0])
    p.add_argument("results", help="directory with *.jsonl")
    p.add_argument("--out", required=True, help="new directory for the rewritten files")
    p.add_argument("--slack", type=float, help="relative slack on every task's reference target")
    p.add_argument("--target", action="append", default=[], metavar="TASK=LOSS")
    p.add_argument("--reference", default="configs/reference.yaml")
    args = p.parse_args(argv)
    if Path(args.out).resolve() == Path(args.results).resolve():
        p.error("--out must differ from the source directory")

    targets: dict[str, float] = {}
    if args.slack is not None:
        with open(args.reference) as f:
            for task, entry in (yaml.safe_load(f) or {}).items():
                targets[task] = entry["target_loss"] * (1 + args.slack)
    for item in args.target:
        task, value = item.split("=", 1)
        targets[task] = float(value)
    if not targets:
        p.error("give --slack and/or --target")
    print("targets:", {k: round(v, 5) for k, v in sorted(targets.items())})
    retarget(Path(args.results), Path(args.out), targets)


if __name__ == "__main__":
    main()
