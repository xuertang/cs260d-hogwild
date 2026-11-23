"""Every figure in the report, regenerated from results/*.jsonl (stream D).

Rule: no figure is edited by hand and no number is typed into the report.
`python -m analysis.plots` rebuilds everything under report/figures/ from the
rows on disk.

Until real rows land (Nov 27), develop against synthetic_rows() below
(README §8: "D builds plots.py against synthetic ResultRows").

STUB: signatures and comments only. Nothing here is implemented yet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hogwild.schema import ResultRow

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("report/figures")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load(paths: list[Path] | None = None) -> pd.DataFrame:
    """All rows from results/*.jsonl as one DataFrame, config fields flattened to columns.

    Use schema.read_rows so tau_hist keys come back as ints. Keep tau_hist and
    the two curves as object columns; everything else is scalar.

    Duplicated run_ids (a run redone after a failure) → keep the last one.
    Runs that never reached the target carry a sentinel in time_to_target_s
    (see the TODO in schema.ResultRow) — decide how each figure shows them
    rather than dropping them silently.
    """
    raise NotImplementedError


def synthetic_rows(n_seeds: int = 3) -> list[ResultRow]:
    """Fake but schema-valid rows covering the E1–E4 grids, for building plots early.

    Shapes worth faking so the plots are exercised properly: near-linear
    hogwild speed-up, sync flattening, a tau histogram centred near m, one
    diverged run (sentinel time_to_target_s).
    """
    raise NotImplementedError


def add_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """speedup = T1 / Tm (README §4: computed in analysis, not stored).

    T1 is the serial run's time_to_target_s for the same task. TODO: match
    per seed, or divide by the mean serial time across seeds? Proposal: mean
    serial time, then report mean ± std of speed-up over the m-worker seeds.
    Also plot against the pure-compute ceiling from README §9.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Figures (one function per figure; each saves a PDF into FIGURES_DIR)
# --------------------------------------------------------------------------- #


def fig_e1_time_to_target(df: pd.DataFrame) -> Path:
    """PRIMARY FIGURE. Time-to-target vs. cores, one line per mode, one panel per task.

    4 panels (T1, T2, T3a, T3b) left to right = rising density then
    non-convexity, so H1 reads straight off the figure. Log-log axes with an
    ideal 1/m reference line; error bars over 3 seeds.
    """
    raise NotImplementedError


def fig_e1_epochs_to_target(df: pd.DataFrame) -> Path:
    """Epochs-to-target vs. m: the statistical-efficiency half of H1.

    H1 predicts throughput scales but iterations-to-target grow with tau on
    dense tasks. This figure separates the two effects that the primary
    figure shows combined.
    """
    raise NotImplementedError


def fig_tau_histograms(df: pd.DataFrame) -> Path:
    """Measured tau distribution per task and m (hogwild rows of E1)."""
    raise NotImplementedError


def fig_e2_injected_tau(df: pd.DataFrame) -> Path:
    """Test accuracy vs. epoch for tau in {8, 32, 128}, eta vs. eta/(1+tau) (H2).

    Use measured tau (from tau_hist) in the legend, not the nominal
    tau_inject — see the note in configs/sweeps/e2_injected_tau.yaml.
    """
    raise NotImplementedError


def fig_e3_straggler_crossover(df: pd.DataFrame) -> Path:
    """Time-to-target vs. tail factor for sync / backup(b=1,2) / hogwild (H3).

    Only two tail settings (mild = 2x, heavy = 6x) plus "none" from E1's m = 8
    column, so the "crossover" is located between at most three points. If a
    finer answer is needed, that means more E3 runs — raise it before Nov 28.
    """
    raise NotImplementedError


def fig_e4_step_size(df: pd.DataFrame) -> Path:
    """Sensitivity to eta in {0.5x, 1x, 2x} for sync vs. hogwild at m = 8 (1x from E1)."""
    raise NotImplementedError


def fig_interference(df: pd.DataFrame) -> Path:
    """interference_mean (gradient cosine) per task — the dense stand-in for rho."""
    raise NotImplementedError


def table_reproducibility(df: pd.DataFrame) -> Path:
    """param_dist_to_seed0 and final_test_acc spread per task and mode → LaTeX table.

    Needs final params saved per run (see the TODO in modes/hogwild.py);
    confirm with stream A where those files end up.
    """
    raise NotImplementedError


def main() -> None:
    # TODO: df = add_speedup(load()); call every fig_* / table_*; print the paths written.
    raise NotImplementedError("plots is a stub; see the comments in analysis/plots.py")


if __name__ == "__main__":
    main()
