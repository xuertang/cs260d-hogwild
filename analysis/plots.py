"""Every figure in the report, regenerated from results/*.jsonl (stream D).

Rule: no figure is edited by hand and no number is typed into the report.
`python -m analysis.plots` rebuilds everything under report/figures/ from the
rows on disk.

    python -m analysis.plots                      # real rows in results/ -> report/figures/
    python -m analysis.plots --synthetic          # fake rows -> report/figures/synthetic/ (watermarked)
    python -m analysis.plots --results DIR --out DIR

Until real rows land (Nov 27), develop against synthetic_rows() below
(README §8: "D builds plots.py against synthetic ResultRows").

Conventions this module relies on (see README §4-§6 and hogwild/schema.py):
  - results/<exp>.jsonl; the file stem becomes the `exp` column ("e1" ... "e4",
    "reference"). A stem like "e1_scaling" or "e1_rerun" also counts as E1.
    reference.jsonl (the serial step-size tuning grid) is loaded but no figure
    uses it: serial baselines come from E1's own serial rows.
  - Config.task is "t1" / "t2" / "t3a" / "t3b" (a config stem such as
    "t2_mnist_softmax" is normalised to its short id on load).
  - time_to_target_s = epochs_to_target = inf for runs that never reached the
    target. They are never dropped: a group where NO seed reached the target is
    drawn as a hollow marker in a grey band at the edge of the panel, a group
    where only k of n seeds reached it is drawn at the mean of those k and
    annotated "k/n".
  - Error bars are the sample std over seeds (ddof = 1); none with one seed.
  - One colour + marker per mode in every figure (MODE_STYLE).
"""

from __future__ import annotations

import argparse
import functools
import math
import textwrap
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from hogwild.schema import Config, ResultRow, read_rows

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("report/figures")
SYNTHETIC_SUBDIR = "synthetic"

# Set by main(--synthetic): stamped across every figure and into every table so
# output built from fake rows can never be mistaken for a result.
_WATERMARK: str | None = None

# --------------------------------------------------------------------------- #
# Names and style
# --------------------------------------------------------------------------- #

TASKS = ("t1", "t2", "t3a", "t3b")
TASK_NAMES = {
    "t1": "T1 RCV1 logreg",
    "t2": "T2 MNIST softmax",
    "t3a": "T3a MNIST MLP",
    "t3b": "T3b CIFAR CNN",
}
TAIL_FACTOR = {"none": 1.0, "mild": 2.0, "heavy": 6.0}  # README §4: tail / median
ETA_RULES = ("const", "1/(1+tau)")
ETA_RULE_NAMES = {"const": "η (serial-tuned)", "1/(1+tau)": "η / (1 + τ)"}

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e9e8e4"
BAND = "#f0efeb"

# Colour set checked with a colour-vision-deficiency validator on white; the
# aqua/red pair is only moderately separated for deuteranopes, so every mode
# also gets its own marker shape. Serial is the neutral reference.
MODE_ORDER = ("serial", "sync", "backup (b=1)", "backup (b=2)", "hogwild")
MODE_STYLE: dict[str, dict[str, Any]] = {
    "serial": {"color": MUTED, "marker": "D"},
    "sync": {"color": "#2a78d6", "marker": "s"},
    "backup (b=1)": {"color": "#eda100", "marker": "^"},
    "backup (b=2)": {"color": "#1baf7a", "marker": "v"},
    "hogwild": {"color": "#e34948", "marker": "o"},
}
_FALLBACK_STYLE = {"color": "#8a8983", "marker": "P"}  # e.g. backup with b = 3

# Ordinal ramp in the hogwild hue, for "more staleness / more workers" within
# hogwild-only figures (tau histograms, E2).
_HOGWILD_RAMP = LinearSegmentedColormap.from_list("hogwild", ["#f3b3b0", "#e34948", "#6f1414"])

RC = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": "#b5b4ae",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
}
FULL_WIDTH = 6.5  # inches: \textwidth of the 11pt article with 1in margins


def _styled(fn: Callable) -> Callable:
    """Run a figure function under the shared rcParams (no global state changed)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with matplotlib.rc_context(RC):
            return fn(*args, **kwargs)

    return wrapper


def mode_label(mode: str, b: int = 0) -> str:
    return f"backup (b={int(b)})" if mode == "backup" else str(mode)


def mode_style(label: str) -> dict[str, Any]:
    return MODE_STYLE.get(label, _FALLBACK_STYLE)


def task_name(task: str) -> str:
    return TASK_NAMES.get(task, str(task))


def _task_id(task: str) -> str:
    """"t2_mnist_softmax" -> "t2" (schema.py OPEN question 3); unknown names pass through."""
    head = str(task).lower().split("_")[0]
    return head if head in TASKS else str(task)


def tasks_present(df: pd.DataFrame) -> list[str]:
    """Tasks in README order (rising density, then non-convexity); unknown ones last."""
    seen = list(pd.unique(df["task"])) if len(df) else []
    return [t for t in TASKS if t in seen] + sorted(t for t in seen if t not in TASKS)


def _ramp(values: Sequence) -> dict:
    """value -> colour, light to dark in the hogwild hue."""
    values = list(values)
    if len(values) == 1:
        return {values[0]: to_hex(_HOGWILD_RAMP(0.5))}
    return {v: to_hex(_HOGWILD_RAMP(i / (len(values) - 1))) for i, v in enumerate(values)}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

_CONFIG_COLS = list(Config.__dataclass_fields__)
_COLUMNS = (
    ["run_id", "exp"]
    + _CONFIG_COLS
    + ["time_to_target_s", "epochs_to_target", "final_test_acc", "tau_hist", "interference_mean",
       "curve_epoch", "curve_wall", "param_dist_to_seed0", "wall_total_s"]
)


def tau_mean(hist: dict[int, int] | None) -> float:
    """Mean of a tau histogram (tau -> count); NaN when it is empty."""
    if not hist:
        return float("nan")
    n = sum(hist.values())
    return sum(int(k) * v for k, v in hist.items()) / n if n else float("nan")


def tau_quantiles(hist: dict[int, int] | None, qs: Sequence[float]) -> list[float]:
    """Quantiles of a tau histogram (smallest tau whose cumulative share >= q)."""
    if not hist or not sum(hist.values()):
        return [float("nan")] * len(qs)
    taus = np.array(sorted(hist), dtype=float)
    cum = np.cumsum([hist[int(t)] for t in taus]) / sum(hist.values())
    return [float(taus[min(int(np.searchsorted(cum, q - 1e-12)), len(taus) - 1)]) for q in qs]


def pool_hists(hists: Iterable[dict[int, int] | None]) -> dict[int, int]:
    out: dict[int, int] = {}
    for h in hists:
        for k, v in (h or {}).items():
            out[int(k)] = out.get(int(k), 0) + int(v)
    return out


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns the figures use (idempotent, returns a copy if it adds any)."""
    need = [c for c in ("exp", "mode_label", "reached", "tau_mean") if c not in df.columns]
    if not need:
        return df
    df = df.copy()
    if "exp" in need:
        df["exp"] = ""
    if "mode_label" in need:
        df["mode_label"] = [mode_label(m, b) for m, b in zip(df["mode"], df["b"])]
    if "reached" in need:
        df["reached"] = np.isfinite(df["time_to_target_s"].astype(float))
    if "tau_mean" in need:
        df["tau_mean"] = [tau_mean(h) for h in df["tau_hist"]]
    return df


def frame_from_rows(groups: dict[str, list[ResultRow]]) -> pd.DataFrame:
    """{exp: rows} -> one DataFrame, config fields flattened, `exp` column filled in."""
    records = []
    for exp, rows in groups.items():
        for r in rows:
            rec = {"run_id": r.run_id, "exp": exp}
            rec.update(r.config.to_dict())
            rec.update(
                time_to_target_s=float(r.time_to_target_s),
                epochs_to_target=float(r.epochs_to_target),
                final_test_acc=float(r.final_test_acc),
                tau_hist=r.tau_hist,
                interference_mean=float(r.interference_mean),
                curve_epoch=r.curve_epoch,
                curve_wall=r.curve_wall,
                param_dist_to_seed0=float(r.param_dist_to_seed0),
                wall_total_s=float(r.wall_total_s),
            )
            records.append(rec)
    df = pd.DataFrame.from_records(records, columns=_COLUMNS)
    if len(df):
        df["task"] = df["task"].map(_task_id)
        # A run redone after a failure is appended to the same file: keep the last.
        # Deduplicated per file, not globally, so that two sweeps which happen to
        # number their runs the same way cannot delete each other's rows.
        df = df.drop_duplicates(subset=["exp", "run_id"], keep="last").reset_index(drop=True)
    return _prep(df)


def _read_file(path: Path) -> list[ResultRow]:
    """schema.read_rows, falling back to line-by-line so one bad line (e.g. the
    half-written last line of a sweep that is still running) does not lose the file."""
    try:
        return read_rows(path)
    except Exception as first_error:  # noqa: BLE001 - reported below
        rows, bad = [], 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(ResultRow.from_json(line))
                except Exception:  # noqa: BLE001
                    bad += 1
        print(f"[load] WARNING {path}: skipped {bad} unreadable line(s) "
              f"({type(first_error).__name__}: {first_error}); kept {len(rows)}")
        return rows


def load(paths: list[Path] | Path | str | None = None) -> pd.DataFrame:
    """All rows from results/*.jsonl as one DataFrame, config fields flattened to columns.

    `paths` may be a list of .jsonl files and/or directories, or a single
    file/directory; None means RESULTS_DIR. Uses schema.read_rows so tau_hist
    keys come back as ints. tau_hist and the two curves stay object columns;
    everything else is scalar. Adds `exp` (the file stem) and the derived
    columns `mode_label`, `reached` and `tau_mean`.

    Duplicated run_ids within a file (a run redone after a failure) -> keep the
    last one. Runs that never reached the target carry inf in time_to_target_s
    and epochs_to_target; they are kept (`reached` is False) and every figure
    shows them explicitly.
    """
    if paths is None:
        paths = [RESULTS_DIR]
    elif isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(p.glob("*.jsonl")))
        elif p.exists():
            files.append(p)
        else:
            print(f"[load] WARNING {p} does not exist")
    groups: dict[str, list[ResultRow]] = {}
    for f in files:
        groups.setdefault(f.stem, []).extend(_read_file(f))
    return frame_from_rows(groups)


def exp_rows(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Rows of one experiment: file stem "e1", or "e1_<anything>" / "e1-<anything>"."""
    df = _prep(df)
    stem = df["exp"].astype(str).str.lower()
    return df[(stem == key) | stem.str.startswith(key + "_") | stem.str.startswith(key + "-")]


def e1_rows(df: pd.DataFrame) -> pd.DataFrame:
    """E1 rows only, guarding against anything that is not a plain scaling run."""
    e1 = exp_rows(df, "e1")
    return e1[(e1["straggler"] == "none") & (e1["tau_inject"] == 0) & (e1["eta_scale"] == "const")]


def serial_reference(df: pd.DataFrame, col: str) -> dict[str, float]:
    """task -> mean of `col` over E1's serial seeds that have a finite value.

    Only E1's serial rows count. results/reference.jsonl also holds serial
    rows, but it is the step-size tuning grid (several etas per task, and no
    target was known yet, so time_to_target_s is inf there): averaging over it
    would not give the serial baseline. A task without a usable E1 serial row
    is simply absent from the result.
    """
    e1 = e1_rows(df)
    ser = e1[e1["mode"] == "serial"]
    out: dict[str, float] = {}
    for task in tasks_present(ser):
        v = ser.loc[ser["task"] == task, col].astype(float)
        v = v[np.isfinite(v)]
        if len(v):
            out[task] = float(v.mean())
    return out


def add_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """speedup = T1 / Tm (README §4: computed in analysis, not stored).

    T1 is the MEAN serial time_to_target_s for the task over seeds (not matched
    per seed); mean ± std of the speed-up is then taken over the m-worker seeds
    in the figures. A run that never reached the target has speedup 0. Adds the
    columns `serial_time_s` and `speedup` (NaN where the task has no serial run
    that reached the target).

    The pure-compute ceiling of README §9 is NOT available: it needs a separate
    measurement (one process doing m x the work) that has no ResultRow field, so
    fig_e1_speedup draws only the ideal S = m line.
    """
    df = _prep(df).copy()
    ref = serial_reference(df, "time_to_target_s")
    df["serial_time_s"] = df["task"].map(ref).astype(float)
    df["speedup"] = df["serial_time_s"] / df["time_to_target_s"].astype(float)
    return df


# --------------------------------------------------------------------------- #
# Synthetic rows
# --------------------------------------------------------------------------- #

_SYN = {  # T = serial seconds to target, E = serial epochs to target, k = epoch inflation per unit tau
    "t1": dict(T=40.0, E=5.0, eta=0.5, bs=16, L0=0.693, Linf=0.20, acc=0.947, chance=0.5, k=0.001, cos=0.01),
    "t2": dict(T=60.0, E=10.0, eta=0.1, bs=32, L0=2.303, Linf=0.27, acc=0.921, chance=0.1, k=0.012, cos=0.35),
    "t3a": dict(T=150.0, E=10.0, eta=0.05, bs=32, L0=2.303, Linf=0.07, acc=0.978, chance=0.1, k=0.020, cos=0.25),
    "t3b": dict(T=600.0, E=20.0, eta=0.02, bs=64, L0=2.303, Linf=0.85, acc=0.710, chance=0.1, k=0.030, cos=0.15),
}
_SYN_THROUGHPUT = {  # contention coefficient c in  throughput(m) = m / (1 + c (m - 1))
    "serial": 0.0, "sync": 0.22, "backup (b=1)": 0.16, "backup (b=2)": 0.17, "hogwild": 0.03,
}


def _syn_row(exp: str, task: str, mode: str, m: int, seed: int, rng: np.random.Generator, *,
             b: int = 0, tau_inject: int = 0, eta_scale: str = "const", straggler: str = "none",
             eta_mult: float = 1.0, epoch_factor: float = 1.0, slow: float = 1.0,
             diverged: bool = False) -> ResultRow:
    p = _SYN[task]
    label = mode_label(mode, b)
    max_epochs = int(3 * p["E"])
    c = _SYN_THROUGHPUT[label]
    throughput = m / (1 + c * (m - 1)) / slow
    sec_per_epoch = p["T"] / p["E"] / throughput * float(rng.lognormal(0, 0.04))
    need = p["E"] * epoch_factor * float(rng.lognormal(0, 0.04))  # epochs to hit the target

    # tau histogram: centred near m - 1 for hogwild, shifted by the injected wait.
    n_updates = 6000
    if mode == "hogwild":
        lam = (m - 1) * (1 + 0.3 * (TAIL_FACTOR[straggler] - 1))
        draws = rng.poisson(lam, n_updates)
        if tau_inject:
            draws = tau_inject + rng.poisson(m / 2, n_updates)
        vals, counts = np.unique(draws, return_counts=True)
        hist = {int(v): int(n) for v, n in zip(vals, counts)}
    else:
        hist = {0: n_updates}
    t_mean = tau_mean(hist)

    epochs = np.arange(0, max_epochs + 1, dtype=float)
    if diverged:
        epochs = epochs[:9]
        loss = p["L0"] * np.exp(0.35 * epochs)
        loss[-2:] = np.inf  # the evaluator saw non-finite loss before the run was stopped
        e2t = t2t = float("inf")
        acc = p["chance"] + abs(float(rng.normal(0, 0.005)))
    else:
        loss = p["Linf"] + (p["L0"] - p["Linf"]) * np.exp(-3.0 * epochs / need)
        loss = loss * (1 + rng.normal(0, 0.004, len(epochs)))
        reached = need <= max_epochs
        e2t = need if reached else float("inf")
        t2t = need * sec_per_epoch if reached else float("inf")
        dense = 0.0 if task == "t1" else 1.0
        acc = p["acc"] - 0.0003 * dense * t_mean - (0.0 if reached else 0.01) + float(rng.normal(0, 0.002))

    return ResultRow(
        run_id=f"{exp}-{task}-{mode}-m{m}-b{b}-tau{tau_inject}-{eta_scale}-{straggler}-x{eta_mult:g}-s{seed}",
        config=Config(task=task, mode=mode, m=m, b=b, eta=p["eta"] * eta_mult, eta_scale=eta_scale,
                      tau_inject=tau_inject, straggler=straggler, seed=seed, batch_size=p["bs"],
                      max_epochs=max_epochs),
        time_to_target_s=t2t,
        epochs_to_target=e2t,
        final_test_acc=float(acc),
        tau_hist=hist,
        interference_mean=(p["cos"] * (1 - 0.02 * m) + float(rng.normal(0, 0.01))) if mode == "hogwild"
        else float("nan"),
        curve_epoch=[(float(e), float(v)) for e, v in zip(epochs, loss)],
        curve_wall=[(float(e * sec_per_epoch), float(v)) for e, v in zip(epochs, loss)],
        param_dist_to_seed0=0.0 if seed == 0
        else (1.3 if mode == "hogwild" else 1.0) * (1 + 0.1 * abs(float(rng.normal()))),
        wall_total_s=float(epochs[-1] * sec_per_epoch),
    )


def synthetic_rows(n_seeds: int = 3, by_exp: bool = False) -> list[ResultRow] | dict[str, list[ResultRow]]:
    """Fake but schema-valid rows covering the E1–E4 grids, for building plots early.

    Mirrors configs/sweeps/*.yaml (156 + 36 + 48 + 48 rows at n_seeds = 3).
    by_exp=True returns {"e1": [...], "e2": [...], "e3": [...], "e4": [...]} so
    frame_from_rows() can fill the `exp` column; the default is the flat list.

    Shapes faked so the plots are exercised properly: near-linear hogwild
    speed-up with epoch inflation growing with tau on the dense tasks, sync
    flattening, a tau histogram centred near m, measured tau above the nominal
    tau_inject in E2, and runs that never reach the target (inf): one seed of
    T3b hogwild m = 16 in E1 (partial group), tau = 128 at constant eta in E2
    and T3b hogwild at 2x eta in E4 (whole groups, with non-finite loss in the
    curve).
    """
    rng = np.random.default_rng(260)
    seeds = range(n_seeds)
    last = n_seeds - 1
    out: dict[str, list[ResultRow]] = {"e1": [], "e2": [], "e3": [], "e4": []}

    def hog_factor(task: str, m: int) -> float:
        return 1 + _SYN[task]["k"] * (m - 1)

    for task in TASKS:  # E1: serial, then sync / backup(b=1) / hogwild at m in {2,4,8,16}
        for seed in seeds:
            out["e1"].append(_syn_row("e1", task, "serial", 1, seed, rng))
        for m in (2, 4, 8, 16):
            for seed in seeds:
                out["e1"].append(_syn_row("e1", task, "sync", m, seed, rng, epoch_factor=1.03))
                out["e1"].append(_syn_row("e1", task, "backup", m, seed, rng, b=1, epoch_factor=1.03))
                out["e1"].append(_syn_row("e1", task, "hogwild", m, seed, rng, epoch_factor=hog_factor(task, m),
                                          diverged=(task == "t3b" and m == 16 and seed == last and n_seeds > 1)))

    for task in ("t3a", "t3b"):  # E2: hogwild m = 8, injected tau x step-size rule
        for tau in (8, 32, 128):
            for rule in ETA_RULES:
                for seed in seeds:
                    if rule == "const":
                        factor = 1 + _SYN[task]["k"] * (tau + 4)
                        div = tau >= 128 or (task == "t3b" and tau == 32 and seed == last)
                    else:
                        factor, div = 1 + 0.004 * tau, False
                    out["e2"].append(_syn_row("e2", task, "hogwild", 8, seed, rng, tau_inject=tau,
                                              eta_scale=rule, epoch_factor=factor, diverged=div))

    for task in ("t2", "t3a"):  # E3: m = 8 under mild / heavy tails
        for tail in ("mild", "heavy"):
            f = TAIL_FACTOR[tail] - 1
            for seed in seeds:
                out["e3"].append(_syn_row("e3", task, "sync", 8, seed, rng, straggler=tail,
                                          epoch_factor=1.03, slow=1 + 0.45 * f))
                out["e3"].append(_syn_row("e3", task, "backup", 8, seed, rng, b=1, straggler=tail,
                                          epoch_factor=1.03, slow=1 + 0.12 * f))
                out["e3"].append(_syn_row("e3", task, "backup", 8, seed, rng, b=2, straggler=tail,
                                          epoch_factor=1.03, slow=1 + 0.04 * f))
                out["e3"].append(_syn_row("e3", task, "hogwild", 8, seed, rng, straggler=tail,
                                          epoch_factor=hog_factor(task, 8) * (1 + 0.12 * f), slow=1 + 0.10 * f))

    hog_2x = {"t1": 0.62, "t2": 0.8, "t3a": 1.25, "t3b": 1.0}
    for task in TASKS:  # E4: m = 8, eta at 0.5x and 2x
        for mult in (0.5, 2.0):
            for seed in seeds:
                out["e4"].append(_syn_row("e4", task, "sync", 8, seed, rng, eta_mult=mult,
                                          epoch_factor=1.03 * (1.7 if mult < 1 else 0.62)))
                out["e4"].append(_syn_row("e4", task, "hogwild", 8, seed, rng, eta_mult=mult,
                                          epoch_factor=hog_factor(task, 8) * (1.7 if mult < 1 else hog_2x[task]),
                                          diverged=(task == "t3b" and mult > 1)))

    if by_exp:
        return out
    return [r for rows in out.values() for r in rows]


def synthetic_frame(n_seeds: int = 3) -> pd.DataFrame:
    """synthetic_rows() as the DataFrame load() would return, speed-up included."""
    return add_speedup(frame_from_rows(synthetic_rows(n_seeds, by_exp=True)))


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #


def use_synthetic_output() -> Path:
    """Send all output to report/figures/synthetic/ and watermark it. Returns the directory."""
    global FIGURES_DIR, _WATERMARK
    if FIGURES_DIR.name != SYNTHETIC_SUBDIR:
        FIGURES_DIR = FIGURES_DIR / SYNTHETIC_SUBDIR
    _WATERMARK = "SYNTHETIC DATA"
    return FIGURES_DIR


def set_figures_dir(path: Path | str) -> None:
    global FIGURES_DIR
    FIGURES_DIR = Path(path)


def save_figure(fig: Figure, name: str) -> Path:
    """Write <FIGURES_DIR>/<name>.pdf (and a .png next to it); returns the PDF path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if _WATERMARK:
        fig.text(0.5, 0.5, _WATERMARK, ha="center", va="center", rotation=18, fontsize=34,
                 fontweight="bold", color="#000000", alpha=0.07, zorder=100)
    pdf = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(pdf.with_suffix(".png"), dpi=200)
    return pdf


def placeholder_figure(name: str, message: str) -> Path:
    """Nothing to draw: save a small labelled placeholder instead of raising."""
    print(f"[{name}] nothing to draw: {message} (placeholder written)")
    fig = Figure(figsize=(3.6, 1.5))
    fig.text(0.5, 0.5, f"{name}\n" + textwrap.fill(message, width=48), ha="center", va="center",
             fontsize=8, color=MUTED)
    return save_figure(fig, name)


def panel_row(n: int, *, sharey: bool = False, height: float = 2.25) -> tuple[Figure, list]:
    """A row of n panels sized for the report's text width."""
    width = {1: 3.4, 2: 4.4, 3: 5.6}.get(n, FULL_WIDTH)
    fig = Figure(figsize=(width, height + 0.35), layout="constrained")
    axes = fig.subplots(1, n, sharey=sharey, squeeze=False)[0]
    return fig, list(axes)


def _plain_ticks(ax, axis: str, values: Sequence[float], labels: Sequence[str] | None = None) -> None:
    """Fixed major ticks at the data's own values, no minor ticks (for log axes over m etc.)."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    values = [float(v) for v in values]
    if labels is None:
        labels = [f"{v:g}" for v in values]
    lookup = dict(zip(values, labels))
    target.set_major_locator(FixedLocator(values))
    target.set_major_formatter(FuncFormatter(lambda v, _pos: lookup.get(v, f"{v:g}")))
    target.set_minor_locator(NullLocator())


def _plain_log_labels(ax, axis: str) -> None:
    """Label an (already) log axis with plain numbers ("20", not "2 x 10^1"); minor
    ticks (2, 5, ...) are labelled only when the axis spans about a decade and a half
    or less. Without this matplotlib labels EVERY minor tick, as "3 x 10^-1", whenever
    fewer than two decade ticks are in view, and the labels overlap in a narrow panel."""
    target, limits = (ax.xaxis, ax.get_xlim) if axis == "x" else (ax.yaxis, ax.get_ylim)

    def minor(v: float, _pos) -> str:
        lo, hi = limits()
        if v <= 0 or lo <= 0 or hi <= 0:
            return ""
        decades = abs(math.log10(hi / lo))
        mantissa = round(v / 10 ** math.floor(math.log10(v)))
        keep = range(2, 10) if decades < 0.4 else (2, 3, 5) if decades < 0.8 else (2, 5) if decades < 1.6 else ()
        return f"{v:g}" if mantissa in keep else ""

    target.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    target.set_minor_formatter(FuncFormatter(minor))


def _plain_log_y(ax) -> None:
    """Log y axis with plain-number labels; see _plain_log_labels."""
    ax.set_yscale("log")
    _plain_log_labels(ax, "y")


def _log_scales(ax, *, x_base: float | None = None, y: bool = False) -> None:
    """Switch axes to log AFTER the data is drawn: matplotlib (3.8) computes wrong
    data limits for errorbar() on an axis that is already logarithmic."""
    if x_base:
        ax.set_xscale("log", base=x_base)
    if y:
        _plain_log_y(ax)


def _pad_log_x(ax, values: Sequence[float], pad: float = 1.25) -> None:
    ax.set_xlim(min(values) / pad, max(values) * pad)


def aggregate(sub: pd.DataFrame, xcol: str, ycol: str, *, positive: bool = False) -> pd.DataFrame:
    """Per x: mean and std (ddof = 1) of y over the seeds with a usable value.

    Usable = finite (and > 0 when `positive`, for log axes). `n` is the number
    of runs, `n_ok` the number that were usable; n_ok < n means some seeds
    never reached the target and the caller must show that.
    """
    rows = []
    for x, g in sub.groupby(xcol, sort=True):
        y = g[ycol].to_numpy(dtype=float)
        ok = np.isfinite(y) & ((y > 0) if positive else True)
        good = y[ok]
        rows.append({
            "x": float(x),
            "mean": float(good.mean()) if len(good) else float("nan"),
            "std": float(good.std(ddof=1)) if len(good) > 1 else 0.0,
            "n_ok": int(len(good)),
            "n": int(len(y)),
        })
    return pd.DataFrame(rows, columns=["x", "mean", "std", "n_ok", "n"])


def _errorbar(ax, x, mean, std, *, color, marker, log_y: bool, line: bool = True, hollow: bool = False,
              zorder: float = 3) -> None:
    mean, std = np.asarray(mean, float), np.asarray(std, float)
    lower = np.minimum(std, mean * 0.999) if log_y else std  # keep the bar on a log axis
    ax.errorbar(x, mean, yerr=[lower, std], color=color, marker=marker,
                linestyle="-" if line else "none", markerfacecolor="white" if hollow else color,
                markeredgecolor=color if hollow else "white", markeredgewidth=1.0 if hollow else 0.6,
                elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=zorder)


def plot_modes(ax, sub: pd.DataFrame, xcol: str, ycol: str, *, log_y: bool) -> dict[str, Any]:
    """One line per mode (mean ± std over seeds) on `ax`.

    Returns {"labels": modes drawn, "pending": [(x, label)] groups in which no
    seed has a usable value (to be drawn by draw_unreached), "partial": bool}.
    """
    info: dict[str, Any] = {"labels": [], "pending": [], "partial": False}
    present = list(pd.unique(sub["mode_label"]))
    for label in [m for m in MODE_ORDER if m in present] + sorted(m for m in present if m not in MODE_ORDER):
        agg = aggregate(sub[sub["mode_label"] == label], xcol, ycol, positive=log_y)
        if agg.empty:
            continue
        st = mode_style(label)
        ok = agg[agg["n_ok"] > 0]
        if len(ok):
            _errorbar(ax, ok["x"], ok["mean"], ok["std"], color=st["color"], marker=st["marker"], log_y=log_y)
        for _, r in ok[ok["n_ok"] < ok["n"]].iterrows():
            info["partial"] = True
            ax.annotate(f"{int(r['n_ok'])}/{int(r['n'])}", (r["x"], r["mean"]), xytext=(4, 4),
                        textcoords="offset points", fontsize=6, color=MUTED, zorder=6)
        info["pending"] += [(float(x), label) for x in agg.loc[agg["n_ok"] == 0, "x"]]
        info["labels"].append(label)
    return info


def draw_unreached(ax, pending: list[tuple[float, str]], *, side: str = "top",
                   styles: dict[str, dict[str, Any]] | None = None, always: bool = False) -> None:
    """Hollow markers in a grey band at the top (or bottom) of the panel for
    groups where no seed reached the target. Call after scales and data are set.
    `styles` overrides the per-mode colour/marker lookup (label -> style);
    `always` reserves the band even when this panel has nothing pending, so
    panels that are meant to share a y range stay aligned."""
    if not pending and not always:
        return
    log_y, log_x = ax.get_yscale() == "log", ax.get_xscale() == "log"
    if not any(len(line.get_xdata()) for line in ax.get_lines()):
        ax.set_ylim((1, 10) if log_y else (0, 1))
    lo, hi = ax.get_ylim()
    if log_y:
        lo, hi = math.log10(lo), math.log10(hi)
    span = hi - lo
    if side == "top":
        y, band, lim = hi + 0.11 * span, (hi + 0.03 * span, hi + 0.19 * span), (lo, hi + 0.19 * span)
    else:
        y, band, lim = lo - 0.11 * span, (lo - 0.19 * span, lo - 0.03 * span), (lo - 0.19 * span, hi)
    if log_y:
        y, band, lim = 10 ** y, tuple(10 ** v for v in band), tuple(10 ** v for v in lim)
    ax.axhspan(*band, color=BAND, linewidth=0, zorder=0)
    ax.set_ylim(lim)
    x0, x1 = ax.get_xlim()
    by_x: dict[float, list[str]] = {}
    for x, label in pending:
        by_x.setdefault(x, []).append(label)
    for x, labels in by_x.items():
        for k, label in enumerate(labels):  # dodge modes that share an x
            off = k - (len(labels) - 1) / 2
            xd = x * 1.09 ** off if log_x else x + off * 0.035 * (x1 - x0)
            st = (styles or {}).get(label) or mode_style(label)
            ax.plot([xd], [y], marker=st["marker"], linestyle="none", markerfacecolor="white",
                    markeredgecolor=st["color"], markeredgewidth=1.1, zorder=6)
    ax.set_xlim(x0, x1)


def mode_handles(labels: Iterable[str]) -> list[Line2D]:
    labels = list(labels)
    ordered = [m for m in MODE_ORDER if m in labels] + [m for m in labels if m not in MODE_ORDER]
    return [Line2D([], [], color=mode_style(m)["color"], marker=mode_style(m)["marker"],
                   markeredgecolor="white", markeredgewidth=0.6, label=m) for m in ordered]


def reference_handle(label: str) -> Line2D:
    return Line2D([], [], color=MUTED, linestyle=(0, (4, 2)), linewidth=0.9, label=label)


def top_legend(fig: Figure, handles: list[Line2D], ncol: int | None = None) -> None:
    """Figure legend above the panels, wrapped onto more rows when it would not fit the width."""
    if not handles:
        return
    widths = sorted((0.068 * len(h.get_label()) + 0.45 for h in handles), reverse=True)  # ~inches at 7pt
    room = fig.get_figwidth() - 0.1
    fits = max([k for k in range(1, len(handles) + 1) if sum(widths[:k]) <= room], default=1)
    ncol = min(ncol or len(handles), fits)
    fig.legend(handles=handles, loc="outside upper center", ncol=ncol,
               columnspacing=1.4, handletextpad=0.5, borderaxespad=0.2)


def footnote(fig: Figure, parts: Iterable[str]) -> None:
    text = "  ".join(p for p in parts if p)
    if text:
        text = textwrap.fill(text, width=int(fig.get_figwidth() * 21))
        fig.supxlabel(text, fontsize=6.3, color=MUTED, x=0.01, ha="left", linespacing=1.25)


def unreached_notes(any_pending: bool, any_partial: bool, where: str = "top") -> list[str]:
    return [
        f"Hollow marker in the grey band ({where}): no seed reached the target." if any_pending else "",
        "k/n: only k of n seeds reached the target (mean over those k)." if any_partial else "",
    ]


# --------------------------------------------------------------------------- #
# Figures (one function per figure; each saves a PDF into FIGURES_DIR)
# --------------------------------------------------------------------------- #


def _e1_figure(df: pd.DataFrame, ycol: str, ylabel: str, name: str, *, log_y: bool, ref: str,
               sharey: bool = False, side: str = "top", note: str = "", zero_base: bool = False) -> Path:
    """Shared layout of the E1 figures: y vs. workers m, one line per mode, one panel per task."""
    e1 = e1_rows(df)
    if ycol not in e1.columns:
        return placeholder_figure(name, f"column {ycol} missing (run add_speedup first)")
    e1 = e1[e1[ycol].notna()]  # NaN = does not apply (e.g. interference for sync); inf is kept
    tasks = tasks_present(e1)
    if not tasks:
        return placeholder_figure(name, "no E1 rows with a value for " + ycol)
    serial = serial_reference(df, ycol) if ref in ("ideal_time", "serial_level") else {}
    fig, axes = panel_row(len(tasks), sharey=sharey)
    labels: list[str] = []
    any_pending = any_partial = ref_drawn = False
    for ax, task in zip(axes, tasks):
        sub = e1[e1["task"] == task]
        ms = sorted(int(m) for m in pd.unique(sub["m"]))
        dash = dict(color=MUTED, linestyle=(0, (4, 2)), linewidth=0.9, zorder=1.5)
        grid = np.array([1] + [m for m in ms if m > 1], dtype=float)
        if ref == "ideal_time" and task in serial:
            ax.plot(grid, serial[task] / grid, **dash)
            ref_drawn = True
        elif ref == "serial_level" and task in serial:
            ax.axhline(serial[task], **dash)
            ref_drawn = True
        elif ref == "ideal_speedup":
            ax.plot(grid, grid, **dash)
            ref_drawn = True
        elif ref == "zero":
            ax.axhline(0.0, color="#b5b4ae", linewidth=0.6, zorder=1.5)
        info = plot_modes(ax, sub, "m", ycol, log_y=log_y)
        _log_scales(ax, x_base=2, y=log_y)
        _plain_ticks(ax, "x", ms)
        _pad_log_x(ax, ms)
        if zero_base:  # ratios between modes read correctly only from a zero baseline
            ax.set_ylim(bottom=0)
        draw_unreached(ax, info["pending"], side=side)
        labels += [m for m in info["labels"] if m not in labels]
        any_pending |= bool(info["pending"])
        any_partial |= info["partial"]
        ax.set_title(task_name(task))
        ax.set_xlabel("workers m")
    axes[0].set_ylabel(ylabel)
    handles = mode_handles(labels)
    if ref_drawn:
        handles.append(reference_handle({"ideal_time": "ideal (serial / m)", "serial_level": "serial mean",
                                         "ideal_speedup": "ideal (S = m)"}[ref]))
    top_legend(fig, handles)
    footnote(fig, [note] + unreached_notes(any_pending, any_partial, side))
    return save_figure(fig, name)


@_styled
def fig_e1_time_to_target(df: pd.DataFrame) -> Path:
    """PRIMARY FIGURE. Time-to-target vs. cores, one line per mode, one panel per task.

    One panel per task (T1, T2, T3b) left to right = rising density then
    non-convexity, so H1 reads straight off the figure. Log-log axes with an
    ideal 1/m reference line (mean serial time / m); error bars = std over
    seeds. Backup rows sit at their m (the m + b processes are not on the axis).
    """
    return _e1_figure(_prep(df), "time_to_target_s", "time to target (s)", "e1_time_to_target",
                      log_y=True, ref="ideal_time")


@_styled
def fig_e1_epochs_to_target(df: pd.DataFrame) -> Path:
    """Epochs-to-target vs. m: the statistical-efficiency half of H1.

    H1 predicts throughput scales but iterations-to-target grow with tau on
    dense tasks. This figure separates the two effects that the primary
    figure shows combined. Linear y from zero; the dashed line is the serial mean.
    """
    return _e1_figure(_prep(df), "epochs_to_target", "epochs to target", "e1_epochs_to_target",
                      log_y=False, ref="serial_level", zero_base=True)


@_styled
def fig_e1_speedup(df: pd.DataFrame) -> Path:
    """Speed-up S(m) = mean serial time / time-to-target vs. m (E1), log-log, with S = m.

    Not in the original stub list: it is the plot add_speedup() exists for. The
    pure-compute ceiling (README §9) is not drawn because it is not in the rows.
    Runs that never reached the target have S = 0 and go in the band at the bottom.
    """
    df = _prep(df)
    if "speedup" not in df.columns:
        df = add_speedup(df)
    return _e1_figure(df, "speedup", "speed-up S(m)", "e1_speedup", log_y=True, ref="ideal_speedup",
                      side="bottom")


@_styled
def fig_tau_histograms(df: pd.DataFrame) -> Path:
    """Measured tau distribution per task and m (hogwild rows of E1).

    One panel per task, one step line per m (light to dark = more workers),
    pooled over seeds and normalised to a share of updates. The triangle on
    the x axis marks each distribution's mean.
    """
    name = "tau_histograms"
    e1 = e1_rows(df)
    hw = e1[e1["mode"] == "hogwild"]
    hw = hw[[bool(h) and sum(h.values()) > 0 for h in hw["tau_hist"]]]
    tasks = tasks_present(hw)
    if not tasks:
        return placeholder_figure(name, "no E1 hogwild rows with a tau histogram")
    ms_all = sorted(int(m) for m in pd.unique(hw["m"]))
    colors = _ramp(ms_all)
    fig, axes = panel_row(len(tasks), sharey=True)
    for ax, task in zip(axes, tasks):
        sub = hw[hw["task"] == task]
        x_hi = 1.0
        for m in sorted(int(v) for v in pd.unique(sub["m"])):
            hist = pool_hists(sub.loc[sub["m"] == m, "tau_hist"])
            total = sum(hist.values())
            taus = np.arange(0, max(hist) + 1)
            if len(taus) > 200_000:  # absurd outlier: fall back to the observed keys only
                taus = np.array(sorted(hist))
            share = np.array([hist.get(int(t), 0) for t in taus], dtype=float) / total
            ax.plot(taus, share, color=colors[m], drawstyle="steps-mid", linewidth=1.3)
            ax.plot([tau_mean(hist)], [0], marker="^", color=colors[m], markeredgecolor="white",
                    markeredgewidth=0.5, markersize=5, linestyle="none", clip_on=False, zorder=6)
            x_hi = max(x_hi, tau_quantiles(hist, [0.999])[0])
        ax.set_xlim(-0.5, x_hi + 1.5)
        ax.set_ylim(bottom=0)
        ax.set_title(task_name(task))
        ax.set_xlabel("staleness τ (updates)")
    axes[0].set_ylabel("share of updates")
    top_legend(fig, [Line2D([], [], color=colors[m], linewidth=1.3, label=f"m = {m}") for m in ms_all])
    footnote(fig, ["hogwild, E1, pooled over seeds. ▲ on the axis: mean τ."])
    return save_figure(fig, name)


def mean_curve(curves: Iterable[Sequence[tuple[float, float]]]):
    """Mean ± std of several (x, y) curves over the x range every curve covers.

    Each curve is cut at its first non-finite y. Returns (x, mean, std, broke)
    where `broke` says whether any curve contained a non-finite value, or None
    if there is nothing to average.
    """
    arrs, broke = [], False
    for c in curves:
        a = np.asarray(c, dtype=float).reshape(-1, 2) if len(c) else np.empty((0, 2))
        a = a[np.isfinite(a[:, 0])]
        a = a[np.argsort(a[:, 0], kind="stable")]
        bad = ~np.isfinite(a[:, 1])
        if bad.any():
            broke = True
            a = a[: int(np.argmax(bad))]
        if len(a):
            arrs.append(a)
    if not arrs:
        return None
    lo, hi = max(a[0, 0] for a in arrs), min(a[-1, 0] for a in arrs)
    grid = np.unique(np.concatenate([a[:, 0] for a in arrs]))
    grid = grid[(grid >= lo) & (grid <= hi)]
    if not len(grid):
        return None
    ys = np.vstack([np.interp(grid, a[:, 0], a[:, 1]) for a in arrs])
    std = ys.std(axis=0, ddof=1) if len(arrs) > 1 else np.zeros(len(grid))
    return grid, ys.mean(axis=0), std, broke


def curve_diverged(curve: Sequence[tuple[float, float]]) -> bool:
    """A loss curve that contains a non-finite value or ends above where it started."""
    a = np.asarray(curve, dtype=float).reshape(-1, 2) if len(curve) else np.empty((0, 2))
    y = a[:, 1]
    if not np.isfinite(y).all():
        return True
    return len(y) > 1 and bool(y[-1] > y[0])


@_styled
def fig_e2_injected_tau(df: pd.DataFrame) -> Path:
    """Test LOSS vs. epoch for tau in {8, 32, 128}, eta vs. eta/(1+tau) (H2).

    The schema has no accuracy curve (curve_epoch is (epochs, test_loss)), so
    each task gets two loss panels (one per step-size rule, colour = injected
    tau) and a small panel of final_test_acc per (tau, rule). Thick line and
    band: mean ± std over the seeds that did not diverge. A seed that diverged
    (curve_diverged) is drawn on its own as a thin line, because averaging it
    in would hide the seeds that trained. The legend gives the MEASURED mean
    tau from tau_hist next to the nominal tau_inject — see the note in
    configs/sweeps/e2_injected_tau.yaml. Grey dashes: serial (from E1).
    """
    name = "e2_injected_tau"
    df = _prep(df)
    e2 = exp_rows(df, "e2")
    e2 = e2[(e2["mode"] == "hogwild") & (e2["tau_inject"] > 0)]
    tasks = tasks_present(e2)
    if not tasks:
        return placeholder_figure(name, "no E2 rows")
    taus = sorted(int(t) for t in pd.unique(e2["tau_inject"]))
    colors = _ramp(taus)
    rules = [r for r in ETA_RULES if r in set(e2["eta_scale"])]
    serial = e1_rows(df)
    serial = serial[serial["mode"] == "serial"]
    serial_acc = serial_reference(df, "final_test_acc")
    dash = dict(color=MUTED, linestyle=(0, (4, 2)), linewidth=0.9, zorder=1.5)

    fig = Figure(figsize=(FULL_WIDTH, 2.45 * len(tasks) + 0.3), layout="constrained")
    subfigs = fig.subfigures(len(tasks), 1, squeeze=False)[:, 0]  # one per task: own legend
    any_partial = any_broke = any_none = any_nan_acc = False
    for row, task in enumerate(tasks):
        sub = e2[e2["task"] == task]
        axs = subfigs[row].subplots(1, len(rules) + 1, squeeze=False,
                                    width_ratios=[1.5] * len(rules) + [1.0])[0]
        loss_axes, acc_ax = list(axs[:-1]), axs[-1]
        ref = mean_curve(serial.loc[serial["task"] == task, "curve_epoch"])
        lows, starts = [], []
        for ax, rule in zip(loss_axes, rules):
            if ref is not None:
                ax.plot(ref[0], ref[1], **dash)
                lows.append(float(np.min(ref[1])))
                starts.append(float(ref[1][0]))
            for tau in taus:
                g = sub[(sub["tau_inject"] == tau) & (sub["eta_scale"] == rule)]
                curves = [c for c in g["curve_epoch"] if len(c)]
                for c in (c for c in curves if curve_diverged(c)):  # one thin line per diverged seed
                    one = mean_curve([c])
                    if one is not None:
                        any_broke = True
                        ax.plot(one[0], one[1], color=colors[tau], linewidth=0.8, zorder=2.5)
                        starts.append(float(one[1][0]))
                cur = mean_curve([c for c in curves if not curve_diverged(c)])
                if cur is None:
                    continue
                x, mean, std, _ = cur
                ax.plot(x, mean, color=colors[tau], linewidth=1.4, zorder=3)
                ax.fill_between(x, np.maximum(mean - std, 1e-12), mean + std, color=colors[tau],
                                alpha=0.14, linewidth=0, zorder=2)
                lows.append(float(np.min(mean)))
                starts.append(float(mean[0]))
            ax.set_title("step size " + ETA_RULE_NAMES.get(rule, rule))
            ax.set_xlabel("epochs")
        if lows and min(lows) > 0:  # curves that blow up leave through the top of the panel
            for ax in loss_axes:
                _plain_log_y(ax)
                ax.set_ylim(min(lows) * 0.8, float(np.median(starts)) * 2.0)
        for ax in loss_axes[1:]:  # same scale and limits as the first loss panel
            ax.tick_params(which="both", labelleft=False)
        loss_axes[0].set_ylabel("test loss")

        handles = []
        for tau in taus:  # measured tau: range over the step-size rules when they disagree
            per_rule = [tau_mean(pool_hists(sub.loc[(sub["tau_inject"] == tau) & (sub["eta_scale"] == r),
                                                    "tau_hist"])) for r in rules]
            per_rule = [v for v in per_rule if np.isfinite(v)]
            if not per_rule:
                measured = "not logged"
            elif max(per_rule) - min(per_rule) > 0.05 * max(per_rule):
                measured = f"{min(per_rule):.1f}–{max(per_rule):.1f}"
            else:
                measured = f"{np.mean(per_rule):.1f}"
            if len(sub[sub["tau_inject"] == tau]):
                handles.append(Line2D([], [], color=colors[tau], linewidth=1.4,
                                      label=f"τ inject {tau} (measured mean {measured})"))
        if ref is not None:
            handles.append(reference_handle("serial"))
        subfigs[row].legend(handles=handles, loc="outside upper center", ncol=len(handles),
                            title=task_name(task), title_fontsize=8, fontsize=6.5, handlelength=1.4,
                            columnspacing=1.2, borderaxespad=0.2)

        # final accuracy: x = nominal tau, marker shape = step-size rule
        if task in serial_acc:
            acc_ax.axhline(serial_acc[task], **dash)
        for j, rule in enumerate(rules):
            off = (j - (len(rules) - 1) / 2) * 0.3
            for i, tau in enumerate(taus):
                g = sub[(sub["tau_inject"] == tau) & (sub["eta_scale"] == rule)]
                acc = g["final_test_acc"].to_numpy(dtype=float)
                acc = acc[np.isfinite(acc)]
                if not len(g):
                    continue
                n_reached = int(g["reached"].sum())
                if not len(acc):  # every seed ended with a non-finite accuracy: say so, do not drop it
                    any_nan_acc = True
                    acc_ax.annotate("n/a", (i + off, 0.03), xycoords=("data", "axes fraction"), rotation=90,
                                    ha="center", va="bottom", fontsize=6, color=colors[tau])
                if len(acc):
                    _errorbar(acc_ax, [i + off], [acc.mean()], [acc.std(ddof=1) if len(acc) > 1 else 0.0],
                              color=colors[tau], marker="os^v"[j % 4], log_y=False, line=False,
                              hollow=(n_reached == 0))
                    if 0 < n_reached < len(g):
                        any_partial = True
                        acc_ax.annotate(f"{n_reached}/{len(g)}", (i + off, acc.mean()), xytext=(3, 4),
                                        textcoords="offset points", fontsize=6, color=MUTED)
                    any_none |= n_reached == 0
        acc_ax.set_xlim(-0.6, len(taus) - 0.4)
        _plain_ticks(acc_ax, "x", range(len(taus)), [str(t) for t in taus])
        acc_ax.grid(axis="x", visible=False)
        acc_ax.set_title("final test acc")
        acc_ax.set_xlabel("τ inject")
        if row == 0:
            acc_ax.legend(handles=[Line2D([], [], color=MUTED, marker="os^v"[j % 4], linestyle="none",
                                          label=ETA_RULE_NAMES.get(r, r)) for j, r in enumerate(rules)],
                          loc="lower left", fontsize=6.3, handletextpad=0.2, borderaxespad=0.2)
    footnote(fig, [
        "Loss: mean ± std over seeds." + (" Thin lines: seeds that diverged, drawn singly and left out of the mean."
                                          if any_broke else ""),
        "Acc: hollow = no seed reached the target loss." if any_none else "",
        "k/n: k of n seeds reached it." if any_partial else "",
        "n/a: no seed has a finite final accuracy." if any_nan_acc else "",
    ])
    return save_figure(fig, name)


def _lines_vs_x(sub_by_task: dict[str, pd.DataFrame], xcol: str, ticks: dict[float, str], xlabel: str,
                name: str, note: str = "") -> Path:
    """Shared layout of E3/E4: time-to-target (log) vs. a short log-spaced x, one panel per task."""
    fig, axes = panel_row(len(sub_by_task))
    labels: list[str] = []
    any_pending = any_partial = False
    for ax, (task, sub) in zip(axes, sub_by_task.items()):
        xs = sorted(float(v) for v in pd.unique(sub[xcol]))
        info = plot_modes(ax, sub, xcol, "time_to_target_s", log_y=True)
        _log_scales(ax, x_base=2, y=True)
        _plain_ticks(ax, "x", xs, [ticks.get(x, f"{x:g}×") for x in xs])
        _pad_log_x(ax, xs, pad=1.35)
        draw_unreached(ax, info["pending"])
        labels += [m for m in info["labels"] if m not in labels]
        any_pending |= bool(info["pending"])
        any_partial |= info["partial"]
        ax.set_title(task_name(task))
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("time to target (s)")
    top_legend(fig, mode_handles(labels))
    footnote(fig, [note] + unreached_notes(any_pending, any_partial))
    return save_figure(fig, name)


def _dominant_m(rows: pd.DataFrame, default: int = 8) -> int:
    return int(rows["m"].mode().iloc[0]) if len(rows) else default


@_styled
def fig_e3_straggler_crossover(df: pd.DataFrame) -> Path:
    """Time-to-target vs. tail factor for sync / backup(b=1,2) / hogwild (H3).

    Only two tail settings (mild = 2x, heavy = 6x) plus "none" (1x) from E1's
    m = 8 column, so the "crossover" is located between at most three points
    (two for backup b=2, which E1 does not run). If a finer answer is needed,
    that means more E3 runs — raise it before Nov 28.
    """
    name = "e3_straggler_crossover"
    df = _prep(df)
    e3 = exp_rows(df, "e3")
    e3 = e3[e3["straggler"].isin(TAIL_FACTOR)]
    if e3.empty:
        return placeholder_figure(name, "no E3 rows")
    m_sel = _dominant_m(e3)
    base = e1_rows(df)
    base = base[(base["m"] == m_sel) & (base["mode"] != "serial")]
    both = pd.concat([base, e3[e3["m"] == m_sel]], ignore_index=True)
    both["tail_factor"] = both["straggler"].map(TAIL_FACTOR).astype(float)
    panels = {t: both[both["task"] == t] for t in tasks_present(e3)}
    missing = [t for t, sub in panels.items() if not (sub["straggler"] == "none").any()]
    if missing:
        print(f"[{name}] note: no E1 m={m_sel} rows for {missing}: the 'none' (1×) point is missing there")
    ticks = {1.0: "none\n1×", 2.0: "mild\n2×", 6.0: "heavy\n6×"}
    return _lines_vs_x(panels, "tail_factor", ticks, "worker-time tail / median", name,
                       note=f"m = {m_sel}; 1× is the E1 run without injected stragglers.")


def eta_multiplier(df: pd.DataFrame, rows: pd.DataFrame, task: str) -> pd.Series | None:
    """eta of `rows` (one task) as a multiple of the step the same mode uses in E1 at the same m.

    The base is per (mode, m), not per task: sync and backup run E1 at m times
    the serial-tuned eta (configs/sweeps/e1_scaling.yaml), hogwild at the
    serial-tuned eta itself, so "1x" is a different eta for each of them.

    Base eta: the (median) eta of the task's E1 rows for that mode and m.
    Without such rows, if the group carries exactly two etas a factor 4 apart
    (the 0.5x / 2x grid), their geometric mean is used and a note is printed.
    Returns None otherwise. results/reference.jsonl is deliberately not
    consulted: it is the tuning grid, so its rows carry several etas per task.
    """
    e1 = e1_rows(df)
    e1 = e1[e1["task"] == task]
    mult = pd.Series(float("nan"), index=rows.index)
    for (mode, m), g in rows.groupby(["mode_label", "m"], sort=False):
        cand = e1.loc[(e1["mode_label"] == mode) & (e1["m"] == m), "eta"]
        if len(cand):
            base = float(cand.median())
        else:
            etas = sorted(float(e) for e in pd.unique(g["eta"]))
            if len(etas) == 2 and abs(etas[1] / etas[0] - 4.0) < 0.05:
                base = math.sqrt(etas[0] * etas[1])
                print(f"[e4] note: no E1 rows for {task} {mode} m={m}; assuming its etas are 0.5× and 2× of {base:g}")
            else:
                return None
        mult[g.index] = g["eta"].astype(float) / base
    snapped = mult.copy()
    for nice in (0.25, 0.5, 1.0, 2.0, 4.0):  # absorb float noise from the sweep's multiplication
        snapped[(mult / nice - 1).abs() < 0.02] = nice
    return snapped.round(3)


@_styled
def fig_e4_step_size(df: pd.DataFrame) -> Path:
    """Sensitivity to eta in {0.5x, 1x, 2x} for sync vs. hogwild at m = 8 (1x from E1).

    The multiplier is recovered as eta / (eta of the same mode's E1 rows at
    this m); see eta_multiplier() for the fallbacks when E1 is missing.
    """
    name = "e4_step_size"
    df = _prep(df)
    e4 = exp_rows(df, "e4")
    if e4.empty:
        return placeholder_figure(name, "no E4 rows")
    m_sel = _dominant_m(e4)
    e4 = e4[e4["m"] == m_sel]
    base = e1_rows(df)
    base = base[(base["m"] == m_sel) & base["mode_label"].isin(set(e4["mode_label"]))]
    panels: dict[str, pd.DataFrame] = {}
    for task in tasks_present(e4):
        rows = pd.concat([base[base["task"] == task], e4[e4["task"] == task]], ignore_index=True)
        mult = eta_multiplier(df, rows, task)
        if mult is None:
            print(f"[{name}] skipping {task}: cannot recover the eta multiplier (no E1 rows for the task)")
            continue
        panels[task] = rows.assign(eta_mult=mult)
    if not panels:
        return placeholder_figure(name, "E4 rows present but no eta multiplier could be recovered")
    return _lines_vs_x(panels, "eta_mult", {}, r"step size  η / η$_{\mathrm{E1}}$", name,
                       note=f"m = {m_sel}; the 1× point is the E1 run "
                            f"(η$_{{\\mathrm{{E1}}}}$ = serial-tuned η for hogwild, m times that for sync).")


@_styled
def fig_interference(df: pd.DataFrame) -> Path:
    """interference_mean (gradient cosine) per task — the dense stand-in for rho.

    One panel per task on a shared y axis, vs. m, from the E1 rows where the
    probe applies (interference_mean is NaN elsewhere, and those rows are
    simply not part of this figure).
    """
    return _e1_figure(_prep(df), "interference_mean", "gradient cosine (mean)",
                      "interference", log_y=False, ref="zero", sharey=True,
                      note="E1; rows without the probe (NaN) are not shown.")


def _tex(s: Any) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_table(name: str, columns: Sequence[str], rows: Sequence[Sequence[str]], align: str,
                comment: str = "") -> Path:
    """A booktabs `tabular` (no float, no caption) for \\input from report/main.tex."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"% Generated by analysis ({name}); do not edit by hand."]
    if comment:
        lines += [f"% {c}" for c in comment.splitlines()]
    if _WATERMARK:
        lines.append(f"% {_WATERMARK} - NOT REAL RESULTS")
    lines += [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    if _WATERMARK:
        lines.append(rf"\multicolumn{{{len(columns)}}}{{c}}{{\textbf{{{_WATERMARK} --- not real results}}}} \\")
        lines.append(r"\midrule")
    lines += [" & ".join(columns) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = FIGURES_DIR / f"{name}.tex"
    path.write_text("\n".join(lines))
    return path


def table_reproducibility(df: pd.DataFrame, m: int = 8) -> Path:
    """param_dist_to_seed0 and final_test_acc spread per task and mode → LaTeX table.

    E1 rows; serial plus each parallel mode at m workers (default 8, the column
    E1, E3 and E4 share; if a (task, mode) has no rows at that m, its largest m
    is used and the m column says so). Spread = sample std and max − min over
    seeds; the distance is averaged over seeds != 0 and shown as "--" when no
    run has it (it stays NaN until final params are saved per run — see the
    TODO in modes/hogwild.py). The "reached" column counts seeds that hit the
    target, so runs that did not are visible here too.
    """
    name = "table_reproducibility"
    e1 = e1_rows(df)
    body: list[list[str]] = []
    for task in tasks_present(e1):
        for label in [x for x in MODE_ORDER if x in set(e1["mode_label"])]:
            g = e1[(e1["task"] == task) & (e1["mode_label"] == label)]
            if g.empty:
                continue
            m_use = m if (g["m"] == m).any() else int(g["m"].max())
            g = g[g["m"] == m_use]
            acc = g["final_test_acc"].to_numpy(dtype=float)
            acc = acc[np.isfinite(acc)]
            digits = 2 if len(acc) and np.nanmax(np.abs(acc)) > 1.5 else 4  # percent vs. fraction
            dist = g.loc[g["seed"] != 0, "param_dist_to_seed0"].to_numpy(dtype=float)
            dist = dist[np.isfinite(dist)]
            body.append([
                _tex(task_name(task)), _tex(label), str(m_use), str(len(g)),
                f"{int(g['reached'].sum())}/{len(g)}",
                f"{acc.mean():.{digits}f}" if len(acc) else "--",
                f"{acc.std(ddof=1):.{digits}f}" if len(acc) > 1 else "--",
                f"{acc.max() - acc.min():.{digits}f}" if len(acc) > 1 else "--",
                (f"{dist.mean():.3g}" + (rf" $\pm$ {dist.std(ddof=1):.2g}" if len(dist) > 1 else ""))
                if len(dist) else "--",
            ])
    if not body:
        print(f"[{name}] no E1 rows: writing an empty table")
    cols = ["task", "mode", "$m$", "seeds", "reached", "acc mean", "acc std", "acc range",
            r"$\lVert\theta - \theta_{\text{seed 0}}\rVert$"]
    return write_table(name, cols, body, "llrrrrrrr",
                       comment="final_test_acc spread and parameter distance to seed 0, E1 rows.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

# (function, experiments whose rows it needs; main() skips it when the first has none)
FIGURES: list[tuple[Callable[[pd.DataFrame], Path], str]] = [
    (fig_e1_time_to_target, "e1"),
    (fig_e1_epochs_to_target, "e1"),
    (fig_e1_speedup, "e1"),
    (fig_tau_histograms, "e1"),
    (fig_interference, "e1"),
    (table_reproducibility, "e1"),
    (fig_e2_injected_tau, "e2"),
    (fig_e3_straggler_crossover, "e3"),
    (fig_e4_step_size, "e4"),
]


def common_parser(description: str) -> argparse.ArgumentParser:
    """CLI flags shared by analysis.plots and analysis.bound_check."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--synthetic", action="store_true",
                   help="build everything from synthetic_rows() into <out>/synthetic/, watermarked")
    p.add_argument("--results", type=Path, default=None, help="directory of *.jsonl (default: results/)")
    p.add_argument("--out", type=Path, default=None, help="output directory (default: report/figures/)")
    p.add_argument("--seeds", type=int, default=3, help="number of seeds for --synthetic")
    return p


def frame_from_args(args: argparse.Namespace) -> pd.DataFrame:
    """Shared by plots.main and bound_check.main: set the output dir, return the rows."""
    if args.out is not None:
        set_figures_dir(args.out)
    if args.synthetic:
        out = use_synthetic_output()
        print(f"SYNTHETIC rows (not results) -> {out}/")
        return synthetic_frame(args.seeds)
    src = args.results if args.results is not None else RESULTS_DIR
    df = add_speedup(load(src))
    counts = df.groupby("exp").size().to_dict() if len(df) else {}
    print(f"loaded {len(df)} rows from {src}/: {counts or 'nothing'}")
    return df


def main(argv: Sequence[str] | None = None) -> None:
    args = common_parser(__doc__.splitlines()[0]).parse_args(argv)
    df = frame_from_args(args)
    written = 0
    for fn, exp in FIGURES:
        if exp_rows(df, exp).empty:
            print(f"skip  {fn.__name__}: no {exp.upper()} rows")
            continue
        print(f"wrote {fn(df)}")
        written += 1
    print(f"{written} of {len(FIGURES)} outputs written to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
