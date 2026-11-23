"""Sweep driver: sweep YAML → list of Configs → runs → results/*.jsonl (stream C).

CLI (README §5):
    python -m hogwild.sweep configs/sweeps/e1_scaling.yaml --out results/e1.jsonl --parallel 2

The sweep YAML format is described at the top of configs/sweeps/e1_scaling.yaml.
It is a proposal — stream C owns it and can change it freely, since nothing
outside this file reads those YAMLs.

STUB: signatures and comments only. Nothing here is implemented yet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hogwild.schema import Config


def expand(sweep_path: str | Path) -> list[Config]:
    """Turn one sweep YAML into the full list of Configs, in a stable order.

    For each grid block: load each `base` task config, take the cartesian
    product of the listed fields, cross with `seeds`, and apply as overrides
    (Config.from_yaml(base, **overrides)).

    `eta_mult` is not a Config field: multiply the base config's eta by it and
    store the product in Config.eta.

    Sanity check to keep: len(expand(path)) must equal `expected_runs` in the
    YAML (156 / 36 / 48 / 48 from README §5). A mismatch means the grid drifted
    from the experiment matrix.
    """
    raise NotImplementedError


def done_run_ids(out_path: str | Path) -> set[str]:
    """Run IDs already present in the output file — the resume mechanism.

    Relies on runner.make_run_id being a pure function of the Config. Tolerate
    a truncated last line (the VM may be killed mid-write): skip it and let
    that run be redone.
    """
    raise NotImplementedError


def run_sweep(configs: list[Config], out_path: str | Path, parallel: int = 1) -> None:
    """Run every Config not already in `out_path`, appending one row per run.

    Parallelism (README §5: two sweeps side by side on the 32-vCPU VM):
      - Each run is itself multi-process and pins workers to cores. Concurrent
        runs must get DISJOINT core sets, or pinning makes them fight over the
        same cores and every timing is garbage. Give each of the `parallel`
        slots its own core offset (slot 0 → cores 0–15, slot 1 → 16–31) and
        pass it through to pin_worker.
      - Memory bandwidth is still shared between slots. README §9 already
        lists this as a risk; a quick check is one E1 config run alone vs.
        alongside another.
      - Only this process appends to `out_path` (children return rows), so
        there is no interleaved-write problem.

    Robustness for the unattended Thanksgiving run:
      - Wrap each run in a subprocess with a timeout; a diverged or hung run
        must not stall the sweep. Record failures to <out>.failed.jsonl with
        the config and the traceback.
      - Order runs so that partial results are useful: all tasks at seed 0
        first, then seeds 1 and 2 (gives D complete-but-noisy plots on Nov 27).
      - Cut order if time runs short (README §5): E4 → m = 2 column of E1 →
        T3b from E3.
    """
    raise NotImplementedError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m hogwild.sweep", description=__doc__.split("\n")[0])
    p.add_argument("sweep", help="configs/sweeps/*.yaml")
    p.add_argument("--out", required=True, help="results/*.jsonl to append to")
    p.add_argument("--parallel", type=int, default=1, help="runs executed side by side")
    p.add_argument("--dry-run", action="store_true", help="print the expanded configs and exit")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # TODO: configs = expand(args.sweep); print count and the number already done;
    #       exit here on --dry-run; else run_sweep(configs, args.out, args.parallel).
    raise NotImplementedError("sweep is a stub; see the comments in hogwild/sweep.py")


if __name__ == "__main__":
    main()
