"""Theory check: measured slowdown vs. the perturbed-iterate prediction (stream D).

README §1: log the empirical tau distribution and the in-flight gradient
interference, and compare the measured slowdown against the Mania et al. bound

    tau = O( min{ n / Delta,  M^2 / (eps * m^2) } )

i.e. the staleness up to which asynchronous SGD keeps the serial rate, up to a
constant.

Before writing code, pin down from the paper (do not trust this docstring for
the definitions — read Mania et al. 2015 and Lecture 4):
  - what n, Delta, M, eps and m denote, and under which assumptions (the
    result is for strongly convex objectives; only T1 and T2 are even convex,
    and of those only T1 is sparse);
  - which of them we can measure or estimate from a run, and which we must
    bound by hand;
  - what the dense analogue of Delta / rho is. The README's answer is the
    measured gradient cosine (ResultRow.interference_mean); the report has to
    argue why that is a fair stand-in.

The honest framing for T3a/T3b: the theorem does not apply, and the check asks
whether its PREDICTION (no slowdown while tau is below the threshold) still
describes what we measure.

STUB: signatures and comments only. Nothing here is implemented yet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def measured_slowdown(df: pd.DataFrame) -> pd.DataFrame:
    """Per (task, m): epochs_to_target(hogwild) / epochs_to_target(serial).

    Epochs, not seconds: the bound is about iterations, and wall-clock mixes in
    throughput. Carry mean and std over seeds, plus a summary of tau_hist
    (mean, median, p95, max) to put on the x-axis.
    """
    raise NotImplementedError


def predicted_threshold(task: str, df: pd.DataFrame) -> float:
    """The tau threshold the bound gives for one task, from measured/estimated constants.

    TODO: one clearly documented estimate per constant, each with its source
          (measured in which run, or assumed). Constants hidden in the O(.)
          mean only the scaling can be tested, not the absolute threshold —
          fit one constant on T1 and check whether it transfers to T2/T3.
    """
    raise NotImplementedError


def fig_bound_check(df: pd.DataFrame) -> Path:
    """Measured slowdown vs. measured tau, with the predicted threshold marked per task.

    E1 only reaches tau ~ m <= 16; the E2 rows (tau up to 128 on T3a/T3b) are
    what push past the threshold, so include both.
    """
    raise NotImplementedError


def main() -> None:
    # TODO: load rows via analysis.plots.load(), build the slowdown table, save the figure
    #       and a small LaTeX table of the estimated constants for §4.4.
    raise NotImplementedError("bound_check is a stub; see the comments in analysis/bound_check.py")


if __name__ == "__main__":
    main()
