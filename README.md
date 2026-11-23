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
│   └── sweeps/            # E1–E4 grids (see §5)
├── hogwild/
│   ├── tasks.py           # Task registry: data loaders, target loss, eval
│   ├── models.py          # LogReg (sparse), Softmax, MLP, SmallCNN
│   ├── runner.py          # process pool, shared-memory params, evaluator snapshots
│   ├── instrument.py      # version counter → τ, delay injection, straggler sim, interference probe
│   ├── modes/
│   │   ├── serial.py
│   │   ├── sync.py        # m workers, barrier, averaged gradient
│   │   ├── backup.py      # m + b workers, first m gradients win
│   │   └── hogwild.py     # lock-free in-place adds on shared tensors
│   ├── sweep.py           # expands a sweep YAML → runs → results/*.jsonl
│   └── schema.py          # Config + ResultRow dataclasses (the contracts in §6)
├── analysis/
│   ├── plots.py           # every figure in the report regenerates from results/
│   ├── bound_check.py     # measured slowdown vs. perturbed-iterate prediction
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
python -m hogwild.tasks download        # RCV1 via sklearn, MNIST + CIFAR-10 via torchvision → ./data
python -m hogwild.runner --config configs/t2_mnist_softmax.yaml --mode serial --smoke
```

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

"Target loss" is fixed per task from the serial run so that time-to-target is comparable across modes.

### Modes

| Mode              | Workers | Barrier | Staleness    | Notes                                                                  |
| ----------------- | ------- | ------- | ------------ | ---------------------------------------------------------------------- |
| `serial`          | 1       | —       | 0            | reference curve, η tuned here                                          |
| `sync`            | m       | yes     | 0            | gradients averaged, same total batch as `hogwild`                      |
| `backup`          | m + b   | yes     | 0            | first m gradients averaged, rest dropped (Chen et al. 2016)            |
| `hogwild`         | m       | no      | measured     | lock-free in-place add on `share_memory_()` tensors                    |
| `hogwild --tau K` | m       | no      | ≥ K injected | worker waits until the version counter has advanced ≥ K before writing |

### Instrumentation (`hogwild/instrument.py`)

- **τ per update**: shared int64 `version` tensor. Worker records `v_read` before copying params, `v_write` before its add, increments after. τ = v_write − v_read. Logged as a histogram per run.
- **Injected τ**: spin until `version − v_read ≥ K`, then write. Sweeps τ past the core count.
- **Straggler sim**: per-step sleep `c + Exp(λ)`; `mild` = tail 2× median, `heavy` = tail 6× median (the measured 99.5-pct / max from Lecture 4).
- **Interference probe**: every N steps, two workers dump their in-flight gradient; cosine logged. Dense stand-in for ρ.
- **Evaluator**: main process snapshots shared params every 2 s wall-clock and every epoch, evaluates test loss/acc. Gives both loss-vs-epoch and loss-vs-wall-clock curves from one run.

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
python -m hogwild.sweep configs/sweeps/e1_scaling.yaml --out results/e1.jsonl --parallel 2
```

**Primary figure:** time-to-target vs. cores, one line per mode, one panel per task (E1). Secondary: τ histograms, injected-τ accuracy curves (E2), straggler crossover (E3), bound check.

---

## 6. Interfaces (so all four streams start on day 1)

These contracts are fixed as of Nov 22 and live in `hogwild/schema.py`. Change them only via PR with all four reviewing.

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

The midterm is behind us, so the original pilot window is gone; this is the compressed schedule. Thanksgiving is Nov 26.

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

Daily: 15-min sync at 9 pm PT on Discord. Blockers go in the channel immediately, not at the sync.

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
