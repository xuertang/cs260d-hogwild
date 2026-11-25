"""Sweep driver: sweep YAML → Configs → runs → results/*.jsonl.

    python -m hogwild.sweep configs/sweeps/e1_scaling.yaml --out results/e1.jsonl --parallel 2
    python -m hogwild.sweep configs/sweeps/e1_scaling.yaml --out results/e1.jsonl --dry-run
    python -m hogwild.sweep configs/sweeps/e1_scaling.yaml --out results/e1.jsonl --only task=t1 --only seed=0

The sweep YAML format is described at the top of configs/sweeps/e1_scaling.yaml.

Resume: a run's ID is a pure function of its Config, so re-running the same
command skips every ID already in the output file. Kill it and restart it as
often as needed.

Each run is a separate `python -m hogwild.runner` process. That isolates the
sweep from a run that crashes, hangs or leaks shared memory, and lets a
timeout kill a run cleanly.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from hogwild.schema import Config

NOT_CONFIG_FIELDS = {"base"}     # grid keys that are not passed through as overrides


def expand(sweep_path: str | Path) -> list[Config]:
    """Every Config of one sweep, seed 0 of everything first.

    Seed-major order means a sweep that is interrupted (or still running)
    already has one complete, if noisy, copy of every grid point.
    """
    with open(sweep_path) as f:
        spec = yaml.safe_load(f)
    configs: list[Config] = []
    for seed in spec["seeds"]:
        for grid in spec["grids"]:
            keys = [k for k in grid if k not in NOT_CONFIG_FIELDS]
            for base in grid["base"]:
                for values in itertools.product(*(grid[k] for k in keys)):
                    configs.append(Config.from_yaml(base, seed=seed, **dict(zip(keys, values))))
    expected = spec.get("expected_runs")
    if expected is not None and len(configs) != expected:
        raise ValueError(f"{sweep_path}: expanded to {len(configs)} runs, expected_runs says {expected}")
    return configs


def done_run_ids(out_path: str | Path) -> set[str]:
    """IDs already in the output file. A truncated last line (the machine was
    killed mid-write) is skipped, so that run is simply done again."""
    ids: set[str] = set()
    path = Path(out_path)
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                ids.add(json.loads(line)["run_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def run_sweep(configs: list[Config], out_path: str | Path, parallel: int = 1, timeout_s: float = 3600.0,
              params_dir: str | Path | None = None) -> tuple[int, int]:
    """Run every Config not already in `out_path`. Returns (succeeded, failed).

    Side-by-side runs get disjoint cores: slot k is offset by k * (cores /
    parallel) through HOGWILD_CORE_OFFSET, which runner.core_for adds to every
    worker's core index. Without that, pinning would put two runs on the same
    cores and every timing would be wrong. (Memory bandwidth is still shared
    between slots; see README §9.)

    Only this process appends to `out_path`: each child writes its row to a
    private temp file, which is copied over under a lock. A failed or timed-out
    run goes to <out>.failed.jsonl with its config and the end of its stderr.
    """
    from hogwild.runner import make_run_id

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path = out_path.with_suffix(".failed.jsonl")
    done = done_run_ids(out_path)
    todo = [c for c in configs if make_run_id(c) not in done]
    print(f"{len(configs)} runs in sweep, {len(configs) - len(todo)} already done, {len(todo)} to run", flush=True)

    cores_per_slot = (os.cpu_count() or 1) // max(parallel, 1)
    slots = list(range(parallel))
    lock = threading.Lock()
    counts = {"ok": 0, "failed": 0}
    t_start = time.monotonic()

    def one(cfg: Config) -> None:
        run_id = make_run_id(cfg)
        with lock:
            slot = slots.pop()
        tmp = out_path.with_name(f".{out_path.stem}.{run_id}.tmp")
        cmd = [sys.executable, "-m", "hogwild.runner", "--json", json.dumps(cfg.to_dict()), "--out", str(tmp), "--quiet"]
        if params_dir:
            cmd += ["--params-dir", str(params_dir)]
        env = {**os.environ, "HOGWILD_CORE_OFFSET": str(slot * cores_per_slot)}
        t0 = time.monotonic()
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
            ok, detail = proc.returncode == 0 and tmp.exists(), proc.stderr[-2000:]
        except subprocess.TimeoutExpired as e:
            ok, detail = False, f"timeout after {timeout_s:.0f}s\n{(e.stderr or b'')[-2000:]}"
        with lock:
            slots.append(slot)
            if ok:
                with open(out_path, "a") as f:
                    f.write(tmp.read_text())
                counts["ok"] += 1
            else:
                with open(failed_path, "a") as f:
                    f.write(json.dumps({"run_id": run_id, "config": cfg.to_dict(), "error": detail}) + "\n")
                counts["failed"] += 1
            tmp.unlink(missing_ok=True)
            n = counts["ok"] + counts["failed"]
            print(f"[{n}/{len(todo)}] {'ok    ' if ok else 'FAILED'} {run_id}  {time.monotonic() - t0:.0f}s  "
                  f"(elapsed {(time.monotonic() - t_start) / 60:.1f} min)", flush=True)

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(one, todo))
    if counts["failed"]:
        print(f"{counts['failed']} runs failed; see {failed_path}. Re-run the same command to retry them.")
    return counts["ok"], counts["failed"]


def parse_only(items: list[str]) -> dict[str, str]:
    return dict(item.split("=", 1) for item in items)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m hogwild.sweep", description="Run a sweep with resume.")
    p.add_argument("sweep", help="configs/sweeps/*.yaml")
    p.add_argument("--out", required=True, help="results/*.jsonl to append to")
    p.add_argument("--parallel", type=int, default=1, help="runs executed side by side, on disjoint cores")
    p.add_argument("--only", action="append", default=[], metavar="FIELD=VALUE",
                   help="keep only configs with this field value, e.g. --only task=t1 (repeatable)")
    p.add_argument("--max-m", type=int, help="drop configs with m + b above this (small machines)")
    p.add_argument("--min-m", type=int, help="drop configs with m + b below this")
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds per run")
    p.add_argument("--params-dir", default="results/params",
                   help="where final parameters are kept for param_dist_to_seed0 ('' to disable)")
    p.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    return p


def main(argv: list[str] | None = None) -> None:
    from hogwild.runner import make_run_id

    args = build_parser().parse_args(argv)
    configs = expand(args.sweep)
    for key, value in parse_only(args.only).items():
        configs = [c for c in configs if str(getattr(c, key)) == value]
    if args.max_m:
        configs = [c for c in configs if c.m + c.b <= args.max_m]
    if args.min_m:
        configs = [c for c in configs if c.m + c.b >= args.min_m]
    if args.dry_run:
        done = done_run_ids(args.out)
        for c in configs:
            run_id = make_run_id(c)
            print(f"{'done' if run_id in done else 'todo'}  {run_id}")
        print(f"{len(configs)} runs, {sum(make_run_id(c) in done for c in configs)} done")
        return
    _, failed = run_sweep(configs, args.out, args.parallel, args.timeout, args.params_dir or None)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
