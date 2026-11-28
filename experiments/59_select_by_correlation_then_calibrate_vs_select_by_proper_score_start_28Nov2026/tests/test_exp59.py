"""Tests for Experiment 59 (run: python -m pytest experiments/59_*/tests -q). Synthetic tests are code checks of known laws, not evidence about real judges."""
from __future__ import annotations

import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
import run_exp59 as R  # noqa: E402
from judge_scores import floor_probs  # noqa: E402

NAME = "chaosnli_mnli"  # any registered name works: the synthetic data replace the bundle under it


def _synthetic(identical=False, n=500, r=30, seed=59):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    ratings = np.stack([rng.choice(3, size=r, p=pi) for pi in p])
    counts = np.stack([np.bincount(row, minlength=3) for row in ratings]).astype(float)
    P = np.zeros((n, 5, 3))
    P[:, 0, :] = floor_probs(p)  # judge 0 forecasts the true law
    for j, conc in zip(range(1, 5), (40.0, 15.0, 6.0, 2.5)):  # judges 1..4 are increasingly noisy versions of it
        P[:, j, :] = floor_probs(np.stack([rng.dirichlet(conc * pi + 1e-3) for pi in p]))
    if identical:
        P[:] = P[:, [0], :]
    return dict(index=2, K=3, counts=counts, P=P, groups=[f"g{i}" for i in range(n)], ratings=ratings, ids=[f"g{i}" for i in range(n)])


@pytest.fixture
def synthetic():
    old = R._D
    R._D = {NAME: _synthetic()}
    yield
    R._D = old


@pytest.fixture
def identical_judges():
    old = R._D
    R._D = {NAME: _synthetic(identical=True)}
    yield
    R._D = old


def test_pearson_or_neg_inf():
    assert R.pearson_or_neg_inf(np.ones(5), np.arange(5.0)) == -np.inf
    assert R.pearson_or_neg_inf(np.arange(5.0), np.ones(5)) == -np.inf
    x, y = np.array([1.0, 2.0, 4.0, 3.0]), np.array([1.0, 3.0, 2.0, 5.0])
    assert R.pearson_or_neg_inf(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])


def test_draw_is_deterministic_and_depends_on_the_draw_index(synthetic):
    a, b, c = R.one_draw(NAME, 2, 3), R.one_draw(NAME, 2, 3), R.one_draw(NAME, 2, 4)
    assert a == b
    assert a != c
    assert a["n_used"] == 100 and a["n_pool"] == 250 and a["n_test"] == 250


def test_label_budget_is_capped_at_the_pool_size(synthetic):
    rec = R.one_draw(NAME, 3, 0)  # n = 200 on 500 items: the pool has 250, so no cap here
    assert rec["n_used"] == 200
    d = R._D[NAME]
    small = {k: (v[:300] if hasattr(v, "__len__") and len(v) == 500 else v) for k, v in d.items()}
    R._D = {NAME: small}
    assert R.one_draw(NAME, 3, 0)["n_used"] == 150  # 300 items, pool of 150, budget 200 is capped at the pool


def test_pool_and_serial_runs_agree(synthetic):
    tasks = [(NAME, ni, r) for ni in (0, 1) for r in range(3)]
    serial = [R.one_draw(*t) for t in tasks]
    with Pool(2) as pool:  # fork: the workers inherit the synthetic data
        parallel = pool.map(R._task, tasks)
    assert serial == parallel


def test_known_winner_is_found_by_every_path_with_small_regret(synthetic):
    rows = [R.one_draw(NAME, 3, r) for r in range(30)]
    for k in "ABC":
        share = np.mean([x[f"pick_{k}"] == x["pick_oracle"] for x in rows])
        assert share >= 0.5, (k, share)  # thresholds chosen generously before looking at the numbers: this is a code check
        assert np.mean([x[f"regret_{k}"] for x in rows]) < 0.02
    assert np.mean([x["pick_oracle"] == 0 for x in rows]) >= 0.8  # the truthful forecaster is the best judge after calibration too


def test_regret_is_zero_when_all_judges_are_identical(identical_judges):
    for r in range(5):
        rec = R.one_draw(NAME, 1, r)
        assert rec["regret_A"] == rec["regret_B"] == rec["regret_C"] == 0.0  # claimed false: any path can have positive regret here
        assert rec["pick_A"] == 0  # ties go to the fixed judge order


def test_committed_results_are_internally_consistent():
    draws = list(csv.DictReader(open(EXP / "results" / "draws.csv")))
    assert len(draws) == 5 * 4 * 200
    assert min(float(x[f"regret_{k}"]) for x in draws for k in "ABC") >= -1e-12
    for x in draws:
        for k in "ABC":
            assert (float(x[f"regret_{k}"]) == 0.0) == (x[f"pick_{k}"] == x["pick_oracle"]) or float(x[f"regret_{k}"]) < 1e-12
    cap = {(x["dataset"], x["n_grid"]): x["n_used"] for x in draws}
    assert cap[("dices350", "200")] == "175" and cap[("dices990", "200")] == "200" and cap[("chaosnli_alphanli", "200")] == "200"
    summ = [r for r in csv.DictReader(open(EXP / "results" / "regret.csv")) if r["path"].startswith("A")]
    h = json.load(open(EXP / "results" / "hypotheses.json"))
    diff25 = {r["dataset"]: float(r["mean_A_minus_B"]) for r in summ if r["n_grid"] == "25"}
    assert h["H1"]["sets_B_lower"] == [k for k, v in diff25.items() if v > 0]
    assert h["H1"]["decision"] == ("supported" if len(h["H1"]["sets_B_lower"]) >= 3 and len(h["H1"]["sets_with_interval_excluding_zero"]) >= 2 else "falsified")
    d200 = {r["dataset"]: abs(float(r["mean_A_minus_B"])) for r in summ if r["n_grid"] == "200"}
    assert h["H2"]["decision"] == ("supported" if all(v < 0.005 for v in d200.values()) else "falsified")
    for r in csv.DictReader(open(EXP / "results" / "regret.csv")):  # the mean regrets agree with the per-draw file
        k = r["path"][0]
        vals = [float(x[f"regret_{k}"]) for x in draws if x["dataset"] == r["dataset"] and x["n_grid"] == r["n_grid"]]
        assert float(r["mean_regret"]) == pytest.approx(np.mean(vals), abs=1e-9)
