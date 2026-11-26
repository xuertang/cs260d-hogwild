"""Theory check: measured slowdown vs. the perturbed-iterate prediction (stream D).

README §1: log the empirical tau distribution and the in-flight gradient
interference, and compare the measured slowdown against the Mania et al. bound

    tau = O( min{ n / Delta,  M^2 / (eps * m^2) } )

i.e. the staleness up to which asynchronous SGD keeps the serial rate, up to a
constant.

    python -m analysis.bound_check                 # results/ -> report/figures/
    python -m analysis.bound_check --synthetic     # fake rows + fake constants -> report/figures/synthetic/
    python -m analysis.bound_check --results DIR --out DIR --reference FILE.yaml

What the symbols mean (Mania, Pan, Papailiopoulos, Recht, Ramchandran, Jordan
2015, the HOGWILD! theorem; written from memory of the paper — CHECK THE
STATEMENT AGAINST THE PAPER BEFORE QUOTING IT IN §4.4):
  n      number of component functions f_i, here the number of training examples.
  Delta  average degree of the conflict graph: two examples conflict when
         their gradients touch a common coordinate. (The paper writes an
         average-degree quantity; the README writes Delta.)
  M      bound on the stochastic gradient norm.
  m      the STRONG-CONVEXITY modulus of the objective. It is NOT the number
         of workers, which is what `m` means everywhere else in this repo.
  eps    target accuracy E||x - x*||^2 <= eps.
  tau    number of updates that overlap in time with a given update, which is
         what the version counter in hogwild/instrument.py measures.
The result is for strongly convex objectives: of our tasks only T1 (l2
logistic regression) satisfies it, T2 is convex but only strongly so through
any l2 term, and T3a/T3b are non-convex.

What this module does and does not test:
  * First term, n / Delta. Estimated as 1 / conflict_frac, where conflict_frac
    (configs/reference.yaml, stream B) is the estimated probability that two
    random training examples share a coordinate, so Delta ~= conflict_frac *
    n_train. Dense tasks have conflict_frac = 1, hence n / Delta = 1: the
    bound then allows only tau = O(1), i.e. it gives no guarantee for any real
    asynchrony. That is the "regime the theory does not cover" of the README.
  * Second term, M^2 / (eps m^2). NOT ESTIMATED. It needs a gradient-norm
    bound, a strong-convexity modulus and a target accuracy in parameter
    space, none of which a run records. It is of the order of the number of
    serial iterations needed to reach eps, which is far above any tau we run,
    so we assume the first term is the binding one. That is an assumption, not
    a measurement.
  * The constant inside the O(.) is unknown. The line drawn at n / Delta is
    therefore NOT an absolute threshold: only the scaling can be tested — does
    the tau at which slowdown sets in move with n / Delta from T1 to the dense
    tasks, and is there a tau range with no slowdown at all on dense tasks
    where the bound promises none?
  * The dense analogue of Delta / rho proposed in the README is the measured
    gradient cosine (ResultRow.interference_mean). It is reported next to the
    constants in the table but does not enter the threshold.

The honest framing for T3a/T3b: the theorem does not apply, and the check asks
whether its PREDICTION (no slowdown while tau is below the threshold) still
describes what we measure.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis import plots

REFERENCE_PATH = Path("configs/reference.yaml")

# Used instead of configs/reference.yaml under --synthetic, so that the figure
# shows a non-trivial threshold. Made-up numbers, like the rows they go with.
SYNTHETIC_REFERENCE: dict[str, dict[str, float]] = {
    "t1": {"n_train": 100_000, "conflict_frac": 0.05},
    "t2": {"n_train": 60_000, "conflict_frac": 1.0},
    "t3a": {"n_train": 60_000, "conflict_frac": 1.0},
    "t3b": {"n_train": 50_000, "conflict_frac": 1.0},
}
_use_synthetic_reference = False
_warned: set[str] = set()

# The three kinds of point in the figure. All are hogwild, so all are in the
# hogwild hue; marker (and shade) say where the staleness and step size come from.
SERIES_STYLE: dict[str, dict[str, Any]] = {
    "natural τ (E1), η": {"color": plots.MODE_STYLE["hogwild"]["color"], "marker": "o"},
    "injected τ (E2), η": {"color": plots.MODE_STYLE["hogwild"]["color"], "marker": "s"},
    "injected τ (E2), η / (1 + τ)": {"color": "#6f1414", "marker": "D"},
}


def _series(exp: str, eta_scale: str) -> str:
    if exp == "e1":
        return "natural τ (E1), η"
    return "injected τ (E2), η" if eta_scale == "const" else "injected τ (E2), η / (1 + τ)"


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"[bound_check] WARNING {message}")


def load_reference() -> dict[str, dict[str, Any]]:
    """configs/reference.yaml as {task: {...}}; {} when the file is missing or unreadable."""
    if _use_synthetic_reference:
        return SYNTHETIC_REFERENCE
    if not REFERENCE_PATH.exists():
        _warn_once("file", f"{REFERENCE_PATH} not found: every task falls back to conflict_frac = 1.0 (dense)")
        return {}
    try:
        import yaml

        with open(REFERENCE_PATH) as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001 - a broken file must not stop the figures
        _warn_once("file", f"could not read {REFERENCE_PATH} ({type(e).__name__}: {e}): using the dense fallback")
        return {}
    return {plots._task_id(k): v for k, v in data.items() if isinstance(v, dict)}


def task_constants(task: str) -> dict[str, Any]:
    """{n_train, conflict_frac, source} for one task, with the dense fallback applied."""
    entry = load_reference().get(task, {})
    n_train = entry.get("n_train")
    frac = entry.get("conflict_frac")
    try:
        frac = float(frac)
        valid = 0.0 < frac <= 1.0
    except (TypeError, ValueError):
        valid = False
    if valid:
        source = "synthetic" if _use_synthetic_reference else REFERENCE_PATH.name
    else:
        _warn_once(f"task:{task}", f"no usable conflict_frac for task {task!r} (got {entry.get('conflict_frac')!r}): "
                                   "assuming 1.0 (dense), so n/Delta = 1")
        frac, source = 1.0, "assumed dense (fallback)"
    return {"n_train": n_train, "conflict_frac": frac, "source": source}


def predicted_threshold(task: str, df: pd.DataFrame | None = None) -> float:
    """The tau scale the bound gives for one task: n / Delta = 1 / conflict_frac.

    This is ONLY the first term of min{n/Delta, M^2/(eps m^2)}, with the O(.)
    constant taken as 1:
      - n / Delta comes from configs/reference.yaml: conflict_frac is stream
        B's estimate of the probability that two random training examples share
        a coordinate, so Delta ~= conflict_frac * n_train and n / Delta =
        1 / conflict_frac. If the file or the key is missing, the task is
        treated as dense (1.0) and a warning is printed.
      - M^2 / (eps m^2) is not estimated (see the module docstring); we assume
        it is the larger term.
      - the constant hidden in O(.) is unknown, so the returned value fixes
        the SCALING across tasks, not an absolute threshold. Do not read "tau
        above this number" as "the theory predicts a slowdown here".
    `df` is unused (no constant is estimated from the runs); it is kept so the
    signature matches the original stub.
    """
    return 1.0 / task_constants(task)["conflict_frac"]


# --------------------------------------------------------------------------- #
# Measured slowdown
# --------------------------------------------------------------------------- #

SLOWDOWN_COLUMNS = ["task", "exp", "m", "tau_inject", "eta_scale", "n_seeds", "n_reached",
                    "serial_epochs", "slowdown_mean", "slowdown_std",
                    "tau_mean", "tau_median", "tau_p95", "tau_max"]


def measured_slowdown(df: pd.DataFrame) -> pd.DataFrame:
    """Per (task, m): epochs_to_target(hogwild) / epochs_to_target(serial).

    Epochs, not seconds: the bound is about iterations, and wall-clock mixes in
    throughput. One row per (task, exp, m, tau_inject, eta_scale) over the
    hogwild rows of E1 (natural staleness) and E2 (injected); E3 and E4 are
    left out because stragglers and a different step size change the slowdown
    for reasons the bound does not speak to.

    The denominator is the mean serial epochs_to_target over seeds for the
    task (same convention as the speed-up). slowdown_mean / slowdown_std are
    over the n_reached seeds that hit the target (std with ddof = 1);
    slowdown_mean is inf when none did and NaN when the task has no serial
    reference. tau_* summarise the tau histograms pooled over the group's
    seeds, to put on the x-axis.
    """
    df = plots._prep(df)
    serial = plots.serial_reference(df, "epochs_to_target")
    e1 = plots.e1_rows(df).assign(exp_key="e1")
    e2 = plots.exp_rows(df, "e2")
    e2 = e2[e2["straggler"] == "none"].assign(exp_key="e2")
    hw = pd.concat([e1, e2], ignore_index=True)
    hw = hw[hw["mode"] == "hogwild"]
    rows = []
    for (task, exp, m, tau_inject, eta_scale), g in hw.groupby(
            ["task", "exp_key", "m", "tau_inject", "eta_scale"], sort=False):
        base = serial.get(task, float("nan"))
        ratio = g["epochs_to_target"].to_numpy(dtype=float) / base
        ok = ratio[np.isfinite(ratio)]
        if len(ok):
            mean, std = float(ok.mean()), float(ok.std(ddof=1)) if len(ok) > 1 else 0.0
        else:
            mean, std = (float("inf") if np.isfinite(base) else float("nan")), float("nan")
        hist = plots.pool_hists(g["tau_hist"])
        median, p95 = plots.tau_quantiles(hist, [0.5, 0.95])
        rows.append({
            "task": task, "exp": exp, "m": int(m), "tau_inject": int(tau_inject), "eta_scale": eta_scale,
            "n_seeds": len(g), "n_reached": int(len(ok)), "serial_epochs": base,
            "slowdown_mean": mean, "slowdown_std": std,
            "tau_mean": plots.tau_mean(hist), "tau_median": median, "tau_p95": p95,
            "tau_max": float(max(hist)) if hist else float("nan"),
        })
    out = pd.DataFrame(rows, columns=SLOWDOWN_COLUMNS)
    if len(out):
        order = {t: i for i, t in enumerate(plots.tasks_present(out))}
        out = (out.assign(_o=out["task"].map(order)).sort_values(["_o", "exp", "eta_scale", "tau_mean"])
               .drop(columns="_o").reset_index(drop=True))
    return out


# --------------------------------------------------------------------------- #
# Figure and table
# --------------------------------------------------------------------------- #


@plots._styled
def fig_bound_check(df: pd.DataFrame) -> Path:
    """Measured slowdown vs. measured tau, with the predicted threshold marked per task.

    One panel per task. x = mean measured tau (log), y = epochs-to-target
    relative to serial (1 = no slowdown), error bars = std over seeds. E1 only
    reaches tau ~ m <= 16; the E2 rows (tau up to 128 on T3a/T3b) are what push
    past the threshold, so both are drawn, with E2's two step-size rules kept
    apart. The dashed vertical line is n / Delta from predicted_threshold():
    a scale, not an absolute threshold (unknown O(.) constant).
    """
    name = "bound_check"
    table = measured_slowdown(df)
    usable = table[table["tau_mean"].gt(0) & table["slowdown_mean"].notna()] if len(table) else table
    dropped = len(table) - len(usable)
    if dropped:
        print(f"[{name}] note: {dropped} group(s) left out (no tau histogram with tau > 0, "
              "or no serial reference for the task)")
    tasks = plots.tasks_present(usable)
    if not tasks:
        return plots.placeholder_figure(name, "no hogwild E1/E2 rows with a tau histogram and a serial reference")
    fig, axes = plots.panel_row(len(tasks), sharey=False, height=2.35)
    used: list[str] = []
    pending_by_ax: list[list[tuple[float, str]]] = []
    any_partial = False
    for ax, task in zip(axes, tasks):
        sub = usable[usable["task"] == task]
        ax.axhline(1.0, color="#b5b4ae", linewidth=0.6, zorder=1.5)
        pending = []
        for label, st in SERIES_STYLE.items():
            g = sub[[_series(e, r) == label for e, r in zip(sub["exp"], sub["eta_scale"])]].sort_values("tau_mean")
            if g.empty:
                continue
            used += [label] if label not in used else []
            ok = g[g["n_reached"] > 0]
            if len(ok):
                plots._errorbar(ax, ok["tau_mean"], ok["slowdown_mean"], ok["slowdown_std"].fillna(0.0),
                                color=st["color"], marker=st["marker"], log_y=False)
            for _, r in ok[ok["n_reached"] < ok["n_seeds"]].iterrows():
                any_partial = True
                ax.annotate(f"{int(r['n_reached'])}/{int(r['n_seeds'])}", (r["tau_mean"], r["slowdown_mean"]),
                            xytext=(4, 4), textcoords="offset points", fontsize=6, color=plots.MUTED, zorder=6)
            pending += [(float(x), label) for x in g.loc[g["n_reached"] == 0, "tau_mean"]]

        plots._log_scales(ax, x_base=10)  # after the error bars, see plots._log_scales
        plots._plain_log_labels(ax, "x")  # T2 spans under two decades: default labels overlap
        # x range: the data, widened to include n / Delta when that is nearby.
        thr = predicted_threshold(task, df)
        lo, hi = float(sub["tau_mean"].min()) / 1.6, float(sub["tau_mean"].max()) * 1.6
        dash = dict(color=plots.MUTED, linestyle=(0, (4, 2)), linewidth=0.9, zorder=1.5)
        thr_text = f"n/Δ = {thr:.3g}"
        if lo / 30 <= thr <= hi * 30:
            lo, hi = min(lo, thr / 1.6), max(hi, thr * 1.6)
            ax.axvline(thr, **dash)
            ax.annotate(thr_text, (thr, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -2),
                        textcoords="offset points", fontsize=6.3, color=plots.MUTED, va="top", ha="left")
        else:  # far outside the data: say so at the edge instead of squashing the points
            right = thr > hi
            ax.annotate(f"{thr_text} →" if right else f"← {thr_text}", (0.98 if right else 0.02, 0.98),
                        xycoords="axes fraction", fontsize=6.3, color=plots.MUTED, va="top",
                        ha="right" if right else "left")
        ax.set_xlim(lo, hi)
        pending_by_ax.append(pending)
        ax.set_title(plots.task_name(task))
        ax.set_xlabel("mean measured τ")
    # One y range for every panel (slowdown is the same quantity in each), set by
    # hand rather than sharey so the "not reached" band lines up across panels.
    any_pending = any(pending_by_ax)
    y_lo = min([0.9] + [ax.get_ylim()[0] for ax in axes])
    y_hi = max([1.15] + [ax.get_ylim()[1] for ax in axes])
    for ax, pending in zip(axes, pending_by_ax):
        ax.set_ylim(y_lo, y_hi)
        plots.draw_unreached(ax, pending, styles=SERIES_STYLE, always=any_pending)
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("epochs to target / serial")
    handles = [Line2D([], [], color=SERIES_STYLE[s]["color"], marker=SERIES_STYLE[s]["marker"],
                      markeredgecolor="white", markeredgewidth=0.6, label="hogwild, " + s) for s in used]
    handles.append(plots.reference_handle("n/Δ (scale only: O(·) constant unknown)"))
    plots.top_legend(fig, handles, ncol=2)
    plots.footnote(fig, plots.unreached_notes(any_pending, any_partial))
    return plots.save_figure(fig, name)


def table_constants(df: pd.DataFrame) -> Path:
    """Small LaTeX table of the estimated constants per task, for §4.4.

    n_train and conflict_frac with their source, the resulting Delta and
    n / Delta, the range of mean measured tau we actually ran (E1 and E2
    hogwild), and the E1 hogwild mean gradient cosine (the proposed dense
    stand-in for rho; reported, not used in the threshold).
    """
    table = measured_slowdown(df)
    e1 = plots.e1_rows(df)
    body = []
    for task in plots.tasks_present(table):
        c = task_constants(task)
        n = c["n_train"]
        n_ok = isinstance(n, (int, float)) and math.isfinite(n)
        taus = table.loc[table["task"] == task, "tau_mean"].dropna()
        cos = e1.loc[(e1["task"] == task) & (e1["mode"] == "hogwild"), "interference_mean"].dropna()
        body.append([
            plots._tex(plots.task_name(task)),
            f"{int(n):,}".replace(",", "{,}") if n_ok else "--",
            f"{c['conflict_frac']:.3g}",
            f"{c['conflict_frac'] * n:.3g}" if n_ok else "--",
            f"{predicted_threshold(task, df):.3g}",
            plots._tex(c["source"]),
            f"{taus.min():.1f}--{taus.max():.1f}" if len(taus) else "--",
            f"{cos.mean():.3f}" if len(cos) else "--",
        ])
    cols = ["task", r"$n$", "conflict frac.", r"$\Delta \approx$ frac.$\cdot n$", r"$n/\Delta$", "source",
            r"measured $\bar\tau$", "grad. cosine"]
    return plots.write_table(
        "table_bound_constants", cols, body, "lrrrrlrr",
        comment="Constants behind the n/Delta term of the Mania et al. bound.\n"
                "The second term M^2/(eps m^2) is NOT estimated and the O(.) constant is unknown:\n"
                "n/Delta sets a scale across tasks, not an absolute threshold.")


def main(argv: Sequence[str] | None = None) -> None:
    global REFERENCE_PATH, _use_synthetic_reference
    parser = plots.common_parser(__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, default=None,
                        help="task constants YAML (default: configs/reference.yaml)")
    args = parser.parse_args(argv)
    if args.reference is not None:
        REFERENCE_PATH = args.reference
    _use_synthetic_reference = bool(args.synthetic)
    df = plots.frame_from_args(args)
    table = measured_slowdown(df)
    if table.empty:
        print("skip  fig_bound_check, table_constants: no hogwild rows in E1 or E2")
        return
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3g}".format):
        print(table.to_string(index=False))
    print(f"wrote {fig_bound_check(df)}")
    print(f"wrote {table_constants(df)}")


if __name__ == "__main__":
    main()
