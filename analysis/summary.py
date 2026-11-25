"""One line per run of a results file — the quick look before any plotting.

    python -m analysis.summary results/e1.jsonl [more.jsonl ...]

speedup is serial time-to-target / this run's time-to-target, using the serial
row of the same task and seed in the same file (nan if there is none).
"""

from __future__ import annotations

import sys

from hogwild.instrument import tau_mean
from hogwild.schema import read_rows


def main(paths: list[str]) -> None:
    rows = [r for p in paths for r in read_rows(p)]
    serial = {(r.config.task, r.config.seed): r for r in rows if r.config.mode == "serial"}
    print(f"{'run':30s} {'wall_s':>7s} {'t_target':>8s} {'ep_target':>9s} {'acc':>6s} {'tau_mean':>8s} "
          f"{'tau_max':>7s} {'cosine':>7s} {'upd/s':>7s} {'speedup':>7s} {'dist_s0':>7s}")
    for r in rows:
        c = r.config
        name = f"{c.task} {c.mode} m={c.m}" + (f" b={c.b}" if c.b else "") + (f" tau={c.tau_inject}" if c.tau_inject else "")
        name += ("" if c.straggler == "none" else f" {c.straggler}") + ("" if c.eta_scale == "const" else " eta/(1+tau)")
        name += f" s{c.seed}"
        ref = serial.get((c.task, c.seed))
        speedup = ref.time_to_target_s / r.time_to_target_s if ref else float("nan")
        updates = sum(r.tau_hist.values())
        print(f"{name:30s} {r.wall_total_s:7.1f} {r.time_to_target_s:8.1f} {r.epochs_to_target:9.2f} "
              f"{r.final_test_acc:6.4f} {tau_mean(r.tau_hist):8.2f} {max(r.tau_hist, default=0):7d} "
              f"{r.interference_mean:7.4f} {updates / max(r.wall_total_s, 1e-9):7.0f} {speedup:7.2f} "
              f"{r.param_dist_to_seed0:7.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
