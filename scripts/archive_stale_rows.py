"""Move rows that the current sweep definitions no longer produce out of results/<exp>.jsonl.

    python scripts/archive_stale_rows.py --dry-run     # report only
    python scripts/archive_stale_rows.py               # e1 e2 e3 e4
    python scripts/archive_stale_rows.py e1 e3

When a sweep YAML changes, rows of the old configs stay in results/<exp>.jsonl
under their old run_ids, and analysis/plots.py would pool them with the new
rows of the same (task, mode, m). This moves every row whose run_id is not in
the current expansion of configs/sweeps/<exp>_*.yaml to
results/stale/<exp>.jsonl, which plots.py does not read (it only globs the top
level of results/). Nothing is deleted, and running it again is a no-op.

First use, Nov 25: sync and backup switched to eta_mult = m (see the header of
configs/sweeps/e1_scaling.yaml); the rows at the unscaled serial eta are kept
in results/stale/ as the "no step-size scaling" result.

Do not run this while a sweep is appending to the same file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)                       # sweep YAMLs name their base configs relative to the repo

from hogwild.runner import make_run_id  # noqa: E402
from hogwild.sweep import expand  # noqa: E402


def run_id_of(line: str) -> str | None:
    try:
        return json.loads(line)["run_id"]
    except (json.JSONDecodeError, KeyError):
        return None                  # truncated last line: leave it where it is


def archive(exp: str, results: Path, dry_run: bool) -> None:
    path = results / f"{exp}.jsonl"
    sweeps = sorted(Path("configs/sweeps").glob(f"{exp}_*.yaml"))
    if not path.exists() or len(sweeps) != 1:
        print(f"{exp}: skipped ({'no ' + str(path) if not path.exists() else f'{len(sweeps)} sweep files match'})")
        return
    current = {make_run_id(c) for c in expand(sweeps[0])}
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    stale = [ln for ln in lines if (rid := run_id_of(ln)) is not None and rid not in current]
    stale_set = set(stale)
    keep = [ln for ln in lines if ln not in stale_set]
    print(f"{exp}: {len(lines)} rows, {len(stale)} stale, {len(keep)} kept"
          f"{' (dry run)' if dry_run else ''}")
    if dry_run or not stale:
        return
    dest = results / "stale" / path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    already = {run_id_of(ln) for ln in dest.read_text().splitlines()} if dest.exists() else set()
    with open(dest, "a") as f:       # archive first, so a crash can only duplicate a row, never lose one
        f.writelines(ln + "\n" for ln in stale if run_id_of(ln) not in already)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(ln + "\n" for ln in keep))
    os.replace(tmp, path)
    print(f"{exp}: moved {len(stale)} rows to {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("exps", nargs="*", default=["e1", "e2", "e3", "e4"])
    p.add_argument("--results", default="results", help="directory holding <exp>.jsonl")
    p.add_argument("--dry-run", action="store_true", help="report what would move and exit")
    args = p.parse_args()
    results = Path(args.results).resolve()
    for exp in args.exps:
        archive(exp, results, args.dry_run)


if __name__ == "__main__":
    main()
