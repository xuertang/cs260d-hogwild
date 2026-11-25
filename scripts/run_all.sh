#!/usr/bin/env bash
# The full experiment matrix (README §5), in priority order: E1, E2, E3, E4.
#
#   bash scripts/run_all.sh              # everything
#   bash scripts/run_all.sh e1 e2        # only these
#
# Safe to re-run at any time: runs already in results/*.jsonl are skipped, so
# after a crash or a killed VM just start it again. Failed runs are listed in
# results/<exp>.failed.jsonl and are retried on the next invocation.
#
# Unattended use (the Thanksgiving run):
#   nohup bash scripts/run_all.sh > results/run_all.log 2>&1 &
#
# Core layout. Runs with at most HALF the machine's cores go two at a time on
# disjoint halves; larger runs (m = 16 on a 32-vCPU VM) go one at a time, so
# two runs never share a core. Before launching on the VM:
#   - `lscpu -e`: vCPUs are usually hyperthreads. For workers on distinct
#     PHYSICAL cores set e.g. HOGWILD_CORES=0,1,...,15 to one thread of each
#     core. A c2-standard-30 has 15 physical cores, so its m = 16 column runs
#     on hyperthreads — say so in the report or pick a bigger instance.
#   - the evaluator is not pinned; leave at least one core free for it.
set -u
cd "$(dirname "$0")/.."

CORES=$(python -c "import os; print(os.cpu_count())")
HALF=$((CORES / 2))
EXPS=("$@"); [ ${#EXPS[@]} -eq 0 ] && EXPS=(e1 e2 e3 e4)

for exp in "${EXPS[@]}"; do
  sweep=$(ls configs/sweeps/${exp}_*.yaml)
  echo "=== $exp ($sweep): runs with m + b <= $HALF, two at a time"
  python -m hogwild.sweep "$sweep" --out "results/$exp.jsonl" --parallel 2 --max-m "$HALF"
  echo "=== $exp: larger runs, one at a time"
  python -m hogwild.sweep "$sweep" --out "results/$exp.jsonl" --parallel 1 --min-m $((HALF + 1))
done

python -m analysis.plots
python -m analysis.bound_check
