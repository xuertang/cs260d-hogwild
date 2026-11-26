# HOGWILD! on Dense, Non-Convex Problems

CS 260D (Large-Scale ML, UCLA, Fall 2026) final project. We measure lock-free asynchronous SGD in the regime its theory does not cover - dense, non-convex models - and check whether the perturbed-iterate analysis still predicts what happens. Four people, one shared-memory runner, ~290 short CPU runs, everything uploaded to Bruinlearn by **Dec 3**.

> Lecture open problems: dense problems, non-convexity, NUMA scaling, wall-clock proofs, reproducibility. We cover the first two, and touch reproducibility.

---

## 1. Research questions

| #                    | Question                                                                                                                | What we expect to see                                                                                                                                                               |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **H1 Density**       | As the touched-coordinate fraction goes from ~0.2% (sparse logistic regression) to 100% (MLP), what does HOGWILD! lose? | Throughput stays near-linear in cores (no barrier); iterations-to-target grow with measured staleness τ. Dense problems pay in **statistical efficiency per step**, not throughput. |
| **H2 Non-convexity** | On a CNN, does staleness change final accuracy or only time-to-reach-it?                                                | τ ≤ cores: final test acc within 0.5 pt of serial. Injected τ ≥ 32 at the serial-tuned η degrades or diverges; η/(1+τ) recovers it.                                                 |
| **H3 Practice**      | Under heavy-tailed worker times, when does sync SGD + backup workers beat HOGWILD! on time-to-accuracy?                 | HOGWILD! wins throughput; m+b backup workers close most of the straggler gap with zero staleness. We locate the crossover vs. tail factor.                                          |

**Theory check** (rides on H1/H2): log the empirical τ distribution and in-flight gradient interference (cosine between concurrent updates — the dense stand-in for ρ), and compare measured slowdown against the Mania et al. bound τ = O(min{n/Δ, M²/(εm²)}).

---

## 2. Repo layout

```
hogwild-dense/
├── README.md
├── requirements.txt
├── configs/
│   ├── t1_rcv1_logreg.yaml
│   ├── t2_mnist_softmax.yaml
│   ├── t3a_mnist_mlp.yaml
│   ├── t3b_cifar_cnn.yaml
│   ├── reference.yaml     # tuned η + target loss per task (written by hogwild/tune.py)
│   └── sweeps/            # E1–E4 grids (see §5)
├── hogwild/
│   ├── tasks.py           # Task registry: data loaders, gradients, eval, dataset download
│   ├── models.py          # LogReg (sparse), Softmax, MLP, SmallCNN
│   ├── runner.py          # worker processes, shared-memory params, evaluator snapshots, CLI
│   ├── instrument.py      # version counter → τ, delay injection, straggler sim, interference probe
│   ├── tune.py            # serial η sweep + reference curve → configs/reference.yaml
│   ├── modes/
│   │   ├── serial.py
│   │   ├── sync.py        # m workers, barrier, averaged gradient
│   │   ├── backup.py      # m + b workers, first m gradients win
│   │   └── hogwild.py     # lock-free in-place adds on shared tensors
│   ├── sweep.py           # expands a sweep YAML → runs → results/*.jsonl, with resume
│   └── schema.py          # Config + ResultRow dataclasses (the contracts in §6)
├── scripts/run_all.sh     # E1–E4 with resume, then all figures
├── tests/                 # python -m pytest tests -q  (no downloads needed)
├── analysis/
│   ├── plots.py           # every figure in the report regenerates from results/
│   ├── bound_check.py     # measured slowdown vs. perturbed-iterate prediction
│   ├── summary.py         # one line per run of a results file
│   ├── retarget.py        # recompute time-to-target for another target loss, from stored curves
│   └── notebooks/
├── results/               # JSONL rows, one per run (git-ignored, synced via Drive)
└── report/                # LaTeX report + slides
```

---

## 3. Setup

```bash
git clone <repo> && cd hogwild-dense
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch (CPU), torchvision, scikit-learn, numpy, pyyaml, pandas, matplotlib
python -m hogwild.tasks download        # RCV1 via sklearn, MNIST + CIFAR-10 via torchvision → ./data (~1.6 GB)
python -m pytest tests -q               # ~15 s; every mode on a synthetic task
python -m hogwild.runner --config configs/t2_mnist_softmax.yaml --mode serial --smoke
python -m hogwild.runner --config configs/t2_mnist_softmax.yaml --mode hogwild --m 4 --out results/pilot.jsonl
```

`configs/reference.yaml` holds each task's tuned η and target loss and is committed, so nobody needs to re-tune. To regenerate it: `python -m hogwild.tune --task t2 --etas 0.0001,0.0003,0.001`. Step sizes are selected on a 10% validation holdout of the training set (seed 0); the test set is only used by the reference run that fixes the target.

Hardware: anything with ≥ 8 physical cores for development (laptop/desktop). The full sweep runs on one **32-vCPU cloud VM** (GCP `c2-standard-30` or similar, ~$1.50/hr, < $15 total). Colab is not usable — 2 vCPUs.

Every worker runs with `torch.set_num_threads(1)` and is pinned to one core; otherwise intra-op threading hides the parallelism we are measuring.

---

## 4. Tasks, modes, metrics

### Tasks (rising density, then non-convexity)

| ID  | Dataset                     | Model                    | Params | Touched coords / example | Target loss             |
| --- | --------------------------- | ------------------------ | ------ | ------------------------ | ----------------------- |
| T1  | RCV1 binary (CCAT vs. rest) | ℓ2-logistic regression   | 47k    | ~0.2%                    | serial loss @ 5 epochs  |
| T2  | MNIST                       | softmax regression       | 7.8k   | 100%                     | serial loss @ 10 epochs |
| T3a | MNIST                       | 2-layer MLP (256 hidden) | 0.2M   | 100%                     | serial loss @ 10 epochs |
| T3b | CIFAR-10                    | 4-conv CNN               | 0.3M   | 100%                     | serial loss @ 20 epochs |

"Target loss" is fixed per task from the serial run so that time-to-target is comparable across modes. Concretely it is the lowest per-epoch median serial test loss within the reference window, stored with the tuned η in `configs/reference.yaml`. Median per epoch because single points are noisy; lowest epoch rather than last because the CIFAR CNN overfits (its test loss bottoms out near epoch 9 and then rises, so "loss @ 20 epochs" would be crossed in the first epoch of every run). For T1, T2 and T3a the minimum is at or next to the last epoch, so this matches the table. T3b's target carries 10% slack on top (0.883 = 1.10 × 0.803): the bottom of an overfitting curve is a fragile target, and in the pilot HOGWILD! at m = 4 bottomed out just above the strict value and never crossed it. `configs/reference.yaml` keeps both numbers (`serial_best_loss`, `target_slack`). Every run gets 1.5× the reference epochs as its budget (T3b: 14 epochs, see `configs/t3b_cifar_cnn.yaml`), so a mode that is less efficient per epoch can still reach the target. Batch size is 1 for T1–T3a and 8 for T3b, per worker. T1 trains on the 781k-document part of RCV1 and tests on the 23k part, as in the HOGWILD! paper; its ℓ2 term uses Andrew's inverse-frequency scheme, where each example carries a share of the penalty for its own features, so the average is full ℓ2 while updates stay sparse (the share is capped at 1000 to keep batch-1 steps stable; see `tasks.Rcv1LogReg`).

### Modes

| Mode              | Workers | Barrier | Staleness    | Notes                                                                   |
| ----------------- | ------- | ------- | ------------ | ----------------------------------------------------------------------- |
| `serial`          | 1       | —       | 0            | reference curve, η tuned here                                           |
| `sync`            | m       | yes     | 0            | gradients averaged, same total batch as `hogwild`                       |
| `backup`          | m + b   | yes     | 0            | first m gradients averaged, rest dropped (Chen et al. 2016)             |
| `hogwild`         | m       | no      | measured     | lock-free in-place add on `share_memory_()` tensors                     |
| `hogwild --tau K` | m       | no      | ≈ K injected | each worker applies the gradient it computed ⌈K/m⌉ of its own steps ago |

### Instrumentation (`hogwild/instrument.py`)

- **τ per update**: shared int64 `version` tensor with one slot per worker (a worker only ever writes its own slot, so no increment is lost and nothing is locked; the version is the sum). Worker records `v_read` before reading params, `v_write` before its add, increments after. τ = v_write − v_read. Logged as a histogram per run.
- **Injected τ**: a per-worker delay queue of ⌈K/m⌉ gradients. (Waiting for the counter to advance by K, as first planned, cannot run: while one worker waits at most m − 1 other updates land, so K ≥ m deadlocks.) The achieved τ is read from the histogram, not assumed. Sweeps τ past the core count.
- **Straggler sim**: per-step delay `c + Exp(λ)` with c = 200 µs, busy-waited for precision; `mild` = tail 2× median, `heavy` = tail 6× median, with tail = 99.5th percentile (the measured 99.5-pct / max from Lecture 4).
- **Interference probe**: once per window of N = 200 updates, workers 0 and 1 dump their in-flight gradient; cosine logged. Dense stand-in for ρ.
- **Evaluator**: main process snapshots shared params every 2 s wall-clock and every quarter epoch, evaluates test loss/acc. Gives both loss-vs-epoch and loss-vs-wall-clock curves from one run. Time-to-target is interpolated between the two snapshots around the crossing.

### Metrics (one `ResultRow` per run)

`time_to_target_s`, `epochs_to_target`, `final_test_acc`, `speedup = T1 / Tm` (computed in analysis), `tau_hist`, `interference_mean`, `curve_epoch[]`, `curve_wall[]`, `param_dist_to_seed0` (reproducibility).

---

## 5. Experiment matrix

| Exp               | Tests  | Tasks            | Modes                                          | Sweep                           | Seeds     | Runs               |
| ----------------- | ------ | ---------------- | ---------------------------------------------- | ------------------------------- | --------- | ------------------ |
| **E1** scaling    | H1, H2 | T1, T2, T3a, T3b | serial; sync, backup(b=1), hogwild             | m ∈ {2, 4, 8, 16}               | 3         | 12 + 144 = **156** |
| **E2** injected τ | H2     | T3a, T3b         | hogwild, m = 8                                 | τ ∈ {8, 32, 128} × {η, η/(1+τ)} | 3         | **36**             |
| **E3** stragglers | H3     | T2, T3a          | sync, backup(b=1), backup(b=2), hogwild, m = 8 | tail ∈ {mild, heavy}            | 3         | **48**             |
| **E4** step size  | H1     | T1–T3b           | sync, hogwild, m = 8                           | η ∈ {0.5×, 2×}                  | 3         | **48**             |
|                   |        |                  |                                                |                                 | **Total** | **≈ 290**          |

Runs take 1–5 min each on CPU. Sequential total ≈ 15 h; two sweeps in parallel on a 32-vCPU VM ≈ 8–10 h wall. **If time runs short, cut in this order:** E4 → m = 2 column of E1 → T3b from E3.

```bash
python -m hogwild.sweep configs/sweeps/e1_scaling.yaml --out results/e1.jsonl --parallel 2   # one experiment
bash scripts/run_all.sh                                                                      # E1–E4, then all figures
python -m analysis.summary results/e1.jsonl                                                  # one line per run
python -m analysis.plots && python -m analysis.bound_check                                   # → report/figures/
```

Sweeps resume: a run's ID is a hash of its config, so re-running a command skips whatever is already in the output file. Failed runs go to `results/<exp>.failed.jsonl` and are retried on the next invocation. Side-by-side runs are pinned to disjoint cores; `scripts/run_all.sh` only runs two at a time when both fit.

**Primary figure:** time-to-target vs. cores, one line per mode, one panel per task (E1). Secondary: τ histograms, injected-τ accuracy curves (E2), straggler crossover (E3), bound check.

---

## 6. Interfaces (so all four streams start on day 1)

These contracts are fixed as of Nov 22 and live in `hogwild/schema.py`. Change them only via PR with all four reviewing.

What the implementation settled where the contract was silent (all additive, nothing below changes a signature):

- `params` in `Task.evaluate(params)` is a list of tensors in `model().parameters()` order.
- Tasks also provide `grads(model, batch)` (the contract has no training loss; T1 needs a sparse gradient), `share_memory()` (one dataset copy for all workers) and `evaluate_curve(params)` (what the loss curves and the target use: the full test set, except T3b, which uses a fixed 2,000-image subset because a full pass takes several seconds on one core; `final_test_acc` always uses the full test set).
- `Logger` is `hogwild.instrument.Logger`; it also carries the epoch limit, so `Mode.run` keeps its signature.
- `Config.task` is `t1` / `t2` / `t3a` / `t3b`. In YAML, `eta: null` means "the task's tuned η" and `eta_mult` scales it (E4).
- Curve points are `(epochs, test_loss)` and `(seconds, test_loss)`. A run that never reaches the target has `time_to_target_s = epochs_to_target = inf`.
- `param_dist_to_seed0` is the L2 distance to the final parameters of the same config at seed 0; sweeps keep the parameters in `results/params/`.
- `sync` uses the tuned η unchanged with an m-times larger effective batch (one update per round); `backup` is the same round logic with b extra workers.

```python
class Task:                       # tasks.py
    name: str
    def model(self) -> torch.nn.Module
    def train_loader(self, seed: int, batch_size: int) -> Iterable
    def evaluate(self, params) -> tuple[float, float]   # (test_loss, test_acc)
    target_loss: float
    default_eta: float

class Mode:                       # modes/*.py
    def run(self, task: Task, cfg: Config, log: Logger) -> ResultRow

@dataclass
class Config:                     # schema.py — mirrors the YAML
    task: str; mode: str; m: int; b: int = 0; eta: float; eta_scale: str = "const"  # or "1/(1+tau)"
    tau_inject: int = 0; straggler: str = "none"; seed: int; batch_size: int; max_epochs: int

@dataclass
class ResultRow:                  # one JSON line in results/*.jsonl
    run_id: str; config: Config
    time_to_target_s: float; epochs_to_target: float; final_test_acc: float
    tau_hist: dict[int, int]; interference_mean: float
    curve_epoch: list[tuple[float, float]]; curve_wall: list[tuple[float, float]]
    param_dist_to_seed0: float; wall_total_s: float
```

---

## 7. Plan — Nov 22 → Dec 3

| Date           | Milestone                                                                                                           | Owner(s)        |
| -------------- | ------------------------------------------------------------------------------------------------------------------- | --------------- |
| **Sun Nov 22** | Repo up, `schema.py` + this README merged, roles confirmed                                                          | all             |
| Mon Nov 23     | `serial` mode + T1/T2 loaders running end-to-end; shared-memory runner skeleton                                     | Andrew, Richard |
| **Tue Nov 24** | **Pilot gate:** `hogwild` on T1 and T2 at m = 4 with τ logged; serial reference curves + tuned η for all four tasks | Richard, Andrew |
| Wed Nov 25     | `sync` + `backup` modes pass the same smoke test; sweep driver runs E1 for T1; VM provisioned                       | C               |
| Thu Nov 26     | E1 runs unattended on the VM (Thanksgiving — nobody babysits it)                                                    | C               |
| Fri Nov 27     | E2 + E3 launched; first E1 plots from partial results; report skeleton with figure placeholders                     | C, D            |
| Sat Nov 28     | E4 + reruns of anything that failed; all raw results in `results/` by midnight                                      | C               |
| **Sun Nov 29** | **Results freeze.** Final figures, bound check done                                                                 | D               |
| Mon Nov 30     | **Report draft complete**, every section owner has written theirs; one full read by someone else                    | all             |
| Tue Dec 1      | Slides done, dry run on Zoom, timing checked (10 min + Q&A)                                                         | all             |
| Wed Dec 2      | Record presentation; report final pass + contribution statement                                                     | all             |
| **Thu Dec 3**  | **Upload recording + report to Bruinlearn**                                                                         | Richard         |

### Status — Nov 24

| Milestone                                                                                        | State                                                                                                                     |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `serial` + T1/T2 loaders end to end; runner                                                      | Done. All four tasks, all four modes, tests in `tests/`.                                                                  |
| **Pilot gate** (hogwild on T1, T2 at m = 4 with τ logged; serial curves + tuned η for all tasks) | **Passed on a laptop** — numbers below. Tuned values in `configs/reference.yaml`.                                         |
| `sync` + `backup` pass the smoke test                                                            | Done (smoke on T1, T2, T3a; full runs on T2).                                                                             |
| Sweep driver runs E1 for T1                                                                      | Driver works with resume. For T1 it ran serial and hogwild only; T1 sync/backup were smoke-tested, not run to completion. |
| VM provisioned; E1–E4 on the VM                                                                  | **Not done.** Needs someone's cloud account. One command once it exists: `bash scripts/run_all.sh`.                       |
| Plots, bound check                                                                               | Code done; exercised on synthetic rows and on the pilot rows (`report/figures/pilot/`).                                   |
| Report                                                                                           | Skeleton with figure placeholders compiles; §3 and Appendix A have draft text. No results text.                           |
| Slides, recording, upload                                                                        | Not started.                                                                                                              |

Pilot, seed 0 only, Apple-silicon laptop with 6 performance cores (so m ≤ 4; these are development numbers, not report numbers). Rows are in `results/pilot/e1.jsonl`.

| Task | Mode         | m   | Updates/s | Time to target (s) | Epochs to target | Speed-up | Final acc | Mean τ | Gradient cosine |
| ---- | ------------ | --- | --------- | ------------------ | ---------------- | -------- | --------- | ------ | --------------- |
| T1   | serial       | 1   | 21,131    | 129.9              | 3.50             | 1.00×    | 0.9457    | 0.00   | —               |
| T1   | hogwild      | 2   | 41,240    | 75.0               | 3.98             | 1.73×    | 0.9457    | 0.60   | 0.0009          |
| T1   | hogwild      | 4   | 80,509    | 36.5               | 3.76             | 3.56×    | 0.9446    | 1.81   | 0.0006          |
| T2   | serial       | 1   | 13,922    | 32.1               | 7.25             | 1.00×    | 0.9228    | 0.00   | —               |
| T2   | hogwild      | 2   | 26,668    | 15.8               | 7.67             | 2.03×    | 0.9237    | 0.73   | 0.0202          |
| T2   | hogwild      | 4   | 53,441    | 10.1               | 8.89             | 3.16×    | 0.9218    | 2.20   | 0.0128          |
| T2   | sync         | 2   | 8,168     | 43.9               | 11.97            | 0.73×    | 0.9218    | 0.00   | 0.0107          |
| T2   | sync         | 4   | 4,989     | never              | never            | —        | 0.9202    | 0.00   | 0.0119          |
| T2   | backup (b=1) | 2   | 7,242     | 44.8               | 10.70            | 0.72×    | 0.9226    | 0.00   | 0.0098          |
| T2   | backup (b=1) | 4   | 4,355     | never              | never            | —        | 0.9201    | 0.00   | 0.0235          |
| T3a  | serial       | 1   | 5,958     | 62.9               | 6.24             | 1.00×    | 0.9838    | 0.00   | —               |
| T3a  | hogwild      | 2   | 10,287    | 38.8               | 6.62             | 1.62×    | 0.9831    | 0.74   | 0.0085          |
| T3a  | hogwild      | 4   | 19,104    | 24.1               | 7.68             | 2.61×    | 0.9821    | 2.26   | 0.0098          |
| T3b  | serial       | 1   | 312       | 98.9               | 4.93             | 1.00×    | 0.7330    | 0.00   | —               |
| T3b  | hogwild      | 2   | 597       | 49.8               | 4.74             | 1.99×    | 0.7261    | 0.95   | 0.0178          |
| T3b  | hogwild      | 4   | 1,122     | 32.5               | 5.87             | 3.04×    | 0.7217    | 2.88   | 0.0274          |

Merged from Andrew's branch (`origin/Andrew`), on top of the code above:

- **Step sizes are selected on a 10% validation holdout** instead of on test loss (`python -m hogwild.tune`; the `t2@val`-style task variants). T2 and T1 kept their step sizes; T3a moved from 0.003 to 0.002.
- **T1 uses his inverse-frequency ℓ2**, capped at 1000 because the uncapped weights are unstable at batch size 1 on the 781k split. The penalty now actually bites: serial test loss is 0.163 where it was 0.144.
- **T3b uses his 0.3M-parameter CNN.** It is about twice as fast per update as the one it replaces.
- **Not taken:** his standalone serial loop (serial runs under the shared runner so that it carries the same overheads as the parallel modes) and his hard-coded targets.
- **Still open, to decide together:** batch size (1/1/1/8 here against his 32/64/64/64) and the RCV1 split (781k training documents here against his 23k). With his settings the serial runs for T1–T3a take 1–4 s, which is cheap but leaves little to time.

What the pilot already shows, and what it changed in the plan:

- **HOGWILD! throughput scales** (3.2–3.8× the serial update rate at m = 4 on every task) and mean τ stays below m. On all three dense tasks it needs more epochs to reach the target at m = 4 than serial does, which is the H1 prediction; one seed is not enough to call it.
- **Sync at the serial-tuned η loses to serial on T2**: one update per round means m× fewer updates per epoch, and at m = 4 it does not reach the target within the budget. This is the README's design (no rescaling; E4 varies η), but expect many `inf` rows for sync/backup in E1 at large m. Decide before the sweep whether that is the comparison we want.
- **RCV1 is sparse per example but its conflict graph is not**: 89% of random document pairs share at least one feature (`conflict_frac` in `configs/reference.yaml`), so the n/Δ term of the Mania et al. bound is about 1 for T1 as well. The measured gradient cosine does separate the tasks (under 0.001 on T1 against 0.01–0.03 on the dense tasks). §4.4 has to be argued from that, not from n/Δ.
- **Target loss had to be redefined** (best serial epoch, not the last one) because the CIFAR CNN overfits; see §4. **T3b additionally gets 10% slack.** Without it HOGWILD! at m = 4 would not get there: its best epoch has test loss 0.835 against 0.799 for serial (strict target 0.803), and it ends 1.1 points lower in accuracy (72.2% vs. 73.3%). Whether that gap is a real H2 effect or one seed is exactly what E1 on T3b should answer, so report the best-epoch loss per mode next to time-to-target; the slack only keeps the time measurable. Every row stores its full curves, so `python -m analysis.retarget` recomputes time-to-target for any other target without rerunning.
- **Time-to-target is a first crossing of a noisy curve**, so it is biased early (serial reaches the T2 target at epoch 7 although the target was defined at epoch 10). All modes share the bias; the seed spread will show how much it matters.
- **Run times are longer than the 1–5 min estimate for serial runs**: about 1 min (T2), 2.5 min (T3a) and 5 min (T1 and T3b) on one laptop performance core.

---

## 8. Work allocation

Each stream has one owner who is accountable for its deliverable and its report section. Pairs review each other's PRs (Richard ↔ C, Andrew ↔ D).

| Stream                          | Owner       | Code owned                                                            | Deliverables                                                                                                                                                 | Report section                                                                        |
| ------------------------------- | ----------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| **A. Runner & instrumentation** | **Richard** | `runner.py`, `instrument.py`, `modes/hogwild.py`                      | Shared-memory worker pool; version-counter τ; injected-τ wait; straggler sim; interference probe; evaluator snapshots. Pilot numbers on T1/T2 by **Nov 24**. | §3 Method (async runner, instrumentation); §Appendix A implementation details         |
| **B. Tasks & models**           | **Andrew**  | `tasks.py`, `models.py`, `modes/serial.py`, `configs/t*.yaml`         | Four tasks with loaders, models, `target_loss`, tuned `default_eta` (η sweep on serial); evaluation harness. Serial reference curves by **Nov 24**.          | §2 Setup (tasks, models, target losses); §4.1 E1 results write-up                     |
| **C. Baselines & sweep**        | **TBD**     | `modes/sync.py`, `modes/backup.py`, `sweep.py`, `configs/sweeps/`, VM | Sync and backup-worker modes; sweep driver with resume; VM provisioning; running E1–E4 and getting every row into `results/` by **Nov 28**.                  | §4.3 E3 straggler results; §5 Threats to validity (timing, overhead ceiling)          |
| **D. Analysis & writing**       | **TBD**     | `analysis/`, `report/`                                                | All figures from `plots.py`; τ histograms and interference; `bound_check.py` vs. Mania et al.; report skeleton **Nov 27**, draft **Nov 30**; slide deck.     | §1 Intro & related work; §4.2 E2 injected-τ results; §4.4 theory check; §6 Conclusion |

Shared by everyone: own report section, slides for own section (2–3 each), contribution statement (required by the syllabus), one full read of the report by **Nov 30**.

### Dependencies between streams

```
schema.py (all, Nov 22)
   ├── tasks.py + serial (Andrew) ──┐
   └── runner + hogwild (Richard) ─┴─► pilot gate Nov 24 ─► sync/backup + sweep (C) ─► results Nov 28 ─► analysis + report (D)
```

Richard and Andrew are unblocked from day 1 by the `Task`/`Mode` contracts. C can write `sync`/`backup` against a stub `Task` and only needs the real runner for the smoke test on Nov 25. D builds `plots.py` against synthetic `ResultRow`s until real rows land Nov 27.

---

## 9. Risks and fallbacks

| Risk                                                            | Fallback                                                                                                                          |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Process/memory-bandwidth overhead caps speed-up, not asynchrony | Report S(m) against a pure-compute ceiling (one process doing m× work); keep models in the few-hundred-KB range                   |
| Lock-free CPU adds are not coordinate-atomic                    | That is how HOGWILD! runs in practice (PyTorch's own example); a locked variant on T1 bounds the error and is itself a data point |
| E1 is dull ("τ ≤ cores is fine")                                | E2 and E3 each stand alone as results; H3 does not depend on H1/H2                                                                |
| Sweep overruns                                                  | Cut order in §5; freeze stays Nov 29 regardless                                                                                   |

---

## 10. Report and presentation

**Report** (owner per section in §8): 1 Intro & related work · 2 Setup · 3 Method · 4 Results (4.1 E1, 4.2 E2, 4.3 E3, 4.4 theory check) · 5 Threats to validity · 6 Conclusion · Contribution statement · Appendix A implementation.

**Presentation** (10 min, recorded, uploaded by Dec 3): problem + open problem (D, 2 min) → runner & instrumentation (Richard, 2 min) → E1/E2 results (Andrew, 3 min) → E3 + theory check + conclusion (C, 3 min).

---

## References

- Niu, Recht, Ré, Wright. _HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent._ NeurIPS 2011.
- Mania, Pan, Papailiopoulos, Recht, Ramchandran, Jordan. _Perturbed Iterate Analysis for Asynchronous Stochastic Optimization._ 2015.
- Chen, Monga, Bengio, Jozefowicz. _Revisiting Distributed Synchronous SGD._ 2016.
- CS 260D Lecture 4 slides; HW1 Q2 (HOGWILD! trace).
