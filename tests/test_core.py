"""Fast checks that need no downloaded data (the RCV1 gradient test is skipped
without it). Run with:  python -m pytest tests -q
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from hogwild import models
from hogwild.instrument import DelayQueue, StragglerSim, TauHistogram, VersionCounter, first_crossing
from hogwild.runner import make_run_id, run_one
from hogwild.schema import Config, ResultRow
from hogwild.sweep import expand
from hogwild.tasks import DATA_DIR, StubTask, get_task

REPO = Path(__file__).resolve().parent.parent


def stub_cfg(**kw) -> Config:
    base = dict(task="stub", mode="serial", m=1, eta=0.05, seed=0, batch_size=1, max_epochs=2)
    return Config(**{**base, **kw})


# ---- schema ---------------------------------------------------------------- #


def test_row_round_trip():
    row = ResultRow(run_id="x", config=stub_cfg(), time_to_target_s=float("inf"), epochs_to_target=1.5,
                    final_test_acc=0.9, tau_hist={0: 3, 7: 2}, interference_mean=0.1,
                    curve_epoch=[(0.0, 1.0)], curve_wall=[(2.0, 0.5)], param_dist_to_seed0=0.25)
    assert ResultRow.from_json(row.to_json()) == row


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        stub_cfg(mode="nope")
    with pytest.raises(ValueError):
        stub_cfg(straggler="sometimes")


def test_eta_mult_and_default_eta():
    cfg = Config.from_dict({**stub_cfg().to_dict(), "eta": None, "eta_mult": 2.0})
    assert cfg.eta == pytest.approx(2.0 * StubTask.fallback_eta)


def test_run_id_is_deterministic_and_seed_sensitive():
    assert make_run_id(stub_cfg()) == make_run_id(stub_cfg())
    assert make_run_id(stub_cfg()) != make_run_id(stub_cfg(seed=1))


# ---- models ---------------------------------------------------------------- #


@pytest.mark.parametrize("model, expected", [
    (models.LogReg(47236), 47236), (models.Softmax(), 7850), (models.MLP(), 203530), (models.SmallCNN(), 316554),
])
def test_param_counts(model, expected):
    assert sum(p.numel() for p in model.parameters()) == expected


def test_cnn_forward_shape():
    assert models.SmallCNN()(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


# ---- instrumentation ------------------------------------------------------- #


def test_version_counter_sums_slots():
    v = VersionCounter(3)
    v.bump(0); v.bump(2, by=5)
    assert v.read() == 6


def test_delay_queue_holds_back_by_delay_steps():
    q = DelayQueue(tau_inject=32, num_workers=8)        # delay = 4 local steps
    out = [q.push(i) for i in range(8)]
    assert out == [None, None, None, None, 0, 1, 2, 3]
    assert DelayQueue(0, 8).push("g") == "g"


def test_tau_histogram_merge():
    assert TauHistogram.merge([{0: 1, 2: 3}, {2: 1, 5: 1}]) == {0: 1, 2: 4, 5: 1}


def test_first_crossing_interpolates():
    pts = [(0.0, 0.0, 1.0, 0.0), (1.0, 10.0, 0.6, 0.0), (2.0, 20.0, 0.2, 0.0)]
    wall, epoch = first_crossing(pts, 0.4)
    assert wall == pytest.approx(15.0) and epoch == pytest.approx(1.5)
    assert first_crossing(pts, 0.1) == (math.inf, math.inf)
    assert first_crossing(pts, float("nan")) == (math.inf, math.inf)


@pytest.mark.parametrize("kind, tail", [("mild", 2.0), ("heavy", 6.0)])
def test_straggler_tail_ratio(kind, tail):
    """p99.5 / median of the sampled delays matches the README's tail factor."""
    sim = StragglerSim(kind, seed=0, worker_id=0)
    from hogwild.instrument import STRAGGLER_BASE_S
    d = STRAGGLER_BASE_S + sim.rng.exponential(sim.scale, size=400_000)
    d.sort()
    assert d[int(0.995 * len(d))] / d[len(d) // 2] == pytest.approx(tail, rel=0.03)


# ---- sweeps ---------------------------------------------------------------- #


@pytest.mark.parametrize("name, n", [("e1_scaling", 117), ("e2_injected_tau", 18), ("e3_stragglers", 24), ("e4_step_size", 36)])
def test_sweep_sizes_match_readme(name, n, monkeypatch):
    monkeypatch.chdir(REPO)
    configs = expand(f"configs/sweeps/{name}.yaml")
    assert len(configs) == n
    assert len({make_run_id(c) for c in configs}) == n          # no two runs collide
    assert all(c.seed == 0 for c in configs[: n // 3])          # seed 0 of everything first


# ---- tasks ----------------------------------------------------------------- #


def test_stub_task_trains():
    task = get_task("stub")
    model = task.model()
    before, _ = task.evaluate(list(model.parameters()))
    for batch in task.train_loader(0, 8):
        grads = task.grads(model, batch)
        with torch.no_grad():
            for p, g in zip(model.parameters(), grads):
                p.add_(g, alpha=-0.1)
    after, acc = task.evaluate(list(model.parameters()))
    assert after < before and acc > 0.8


def test_validation_variant_holds_out_training_data():
    full, val = get_task("stub"), get_task("stub@val")
    assert val.n_train == full.n_train - round(0.1 * full.n_train)
    assert val.x_test.shape[0] == round(0.1 * full.n_train)
    again = get_task("stub@val")
    assert torch.equal(val.x_test, again.x_test)                 # the split is fixed
    with pytest.raises(ValueError):
        get_task("stub@nope")


@pytest.mark.skipif(not (DATA_DIR / "rcv1.pt").exists(), reason="RCV1 not downloaded")
def test_rcv1_validation_variant_is_disjoint_from_training():
    task = get_task("t1@val")
    assert task.n_train + task.test_y.shape[0] == 781265
    held = set(range(781265)) - set(task.train_rows.tolist())
    assert len(held) == task.test_y.shape[0]
    cols, vals, local, y = next(iter(task.train_loader(0, 4)))
    assert y.shape[0] == 4 and int(local.max()) == 3


@pytest.mark.skipif(not (DATA_DIR / "rcv1.pt").exists(), reason="RCV1 not downloaded")
def test_rcv1_l2_weights_are_capped_inverse_frequency():
    task = get_task("t1")
    counts = torch.bincount(task.idx.long(), minlength=task.num_features)
    common = int(counts.argmax())
    assert task.l2_weight[common] == pytest.approx(task.n_train / int(counts[common]))
    assert float(task.l2_weight.max()) == task.L2_WEIGHT_CAP


@pytest.mark.skipif(not (DATA_DIR / "rcv1.pt").exists(), reason="RCV1 not downloaded")
@pytest.mark.parametrize("batch_size", [1, 16])
def test_rcv1_sparse_gradient_matches_autograd(batch_size):
    task = get_task("t1")
    model = task.model()
    with torch.no_grad():
        model.weight.normal_(0, 0.1, generator=torch.Generator().manual_seed(0))
    batch = next(iter(task.train_loader(0, batch_size)))
    manual = task.grads(model, batch)[0].to_dense()
    auto = torch.autograd.grad(task.loss(model, batch), model.weight)[0]
    assert torch.allclose(manual, auto, atol=1e-7)
    assert (manual != 0).sum() <= batch[0].numel()               # only touched coordinates


# ---- end to end ------------------------------------------------------------ #


@pytest.mark.parametrize("overrides", [
    dict(mode="serial"),
    dict(mode="hogwild", m=3),
    dict(mode="hogwild", m=3, tau_inject=12, eta_scale="1/(1+tau)"),
    dict(mode="sync", m=3),
    dict(mode="backup", m=3, b=1),
    dict(mode="hogwild", m=2, straggler="mild"),
])
def test_every_mode_learns_the_stub_task(overrides):
    row = run_one(stub_cfg(**overrides), verbose=False)
    assert row.final_test_acc > 0.85
    assert row.curve_epoch[-1][0] == pytest.approx(2.0, abs=0.05)       # stopped at max_epochs
    assert math.isfinite(row.time_to_target_s)
    n_updates = sum(row.tau_hist.values())
    if overrides["mode"] in ("sync", "backup"):
        assert set(row.tau_hist) == {0}                                  # zero staleness by construction
        assert n_updates == pytest.approx(2 * 4096 / overrides["m"], abs=2)
    elif overrides["mode"] == "serial":
        assert set(row.tau_hist) == {0}
    if overrides.get("tau_inject"):
        mean_tau = sum(t * c for t, c in row.tau_hist.items()) / n_updates
        assert mean_tau >= overrides["tau_inject"] * 0.8                 # injection really delays updates
