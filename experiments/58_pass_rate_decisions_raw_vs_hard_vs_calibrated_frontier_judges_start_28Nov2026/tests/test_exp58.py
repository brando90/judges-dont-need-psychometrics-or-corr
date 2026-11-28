"""Tests for Experiment 58 (run: python -m pytest experiments/58_*/tests -q). Synthetic tests are code checks of known laws, not evidence about judges."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
import run_exp58 as R  # noqa: E402
from judge_scores import apply_calibrator, boot_mean, boot_weights_grouped, crossfit_calibrate, fit_calibrator, floor_probs, make_group_folds  # noqa: E402


def _rows(name):
    with open(EXP / "results" / name) as f:
        return list(csv.DictReader(f))


def test_hard_label_share_on_a_hand_computed_example_including_a_tie():
    q = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2], [1 / 3, 1 / 3, 1 / 3], [0.45, 0.1, 0.45]])
    assert np.allclose(R.hard_label_share(q, [2]), [1.0, 0.0, 1 / 3, 0.5])  # a uniform item splits equally across the three labels
    assert np.allclose(R.hard_label_share(q, [1, 2]), [1.0, 0.0, 2 / 3, 0.5])


def test_gate_values_round_to_the_nearest_five_points():
    assert R.gate_values(32.7) == [25.0, 30.0, 35.0, 40.0, 45.0]
    assert R.gate_values(47.8) == [40.0, 45.0, 50.0, 55.0, 60.0]
    assert R.gate_values(28.2) == [20.0, 25.0, 30.0, 35.0, 40.0]


def _synthetic(n=3000, r=20, seed=58):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    counts = np.stack([np.bincount(rng.choice(3, size=r, p=pi), minlength=3) for pi in p]).astype(float)
    return p, counts


def test_known_law_biased_judge_has_the_known_rate_error_and_calibration_removes_it():
    p, counts = _synthetic()
    q = floor_probs(0.7 * p + 0.3 * np.array([1.0, 0.0, 0.0]))  # moves 30% of every forecast onto label 0: the rate of label 2 is shrunk by 30%
    human = 100 * (counts[:, 2] / counts.sum(1)).mean()
    raw = 100 * q[:, 2].mean()
    assert raw - human == pytest.approx(-0.3 * human, abs=1.0)  # about -10 points, derived from the construction
    folds = make_group_folds([str(i) for i in range(len(p))], 5, seed=1)
    cal = 100 * crossfit_calibrate(q, counts, folds)[:, 2].mean()
    assert abs(cal - human) < 1.0
    hard = 100 * R.hard_label_share(q, [2]).mean()
    assert not np.allclose(hard, raw, atol=1.0)  # claimed false: the hard-label rate equals the soft rate (here it is far smaller)


def test_calibration_matches_the_training_label_shares_by_construction():
    """The map has an unpenalised intercept per label, so the fitted forecasts reproduce the training label shares. This is why the
    calibrated soft rate has about zero error: it says the level is easy to restore, not that the judge discriminates."""
    p, counts = _synthetic(n=800, seed=3)
    q = floor_probs(0.6 * p + 0.4 * np.array([0.2, 0.7, 0.1]))
    Wm = fit_calibrator(q, counts)
    fitted = apply_calibrator(Wm, q).mean(0)
    share = counts.sum(0) / counts.sum()
    assert np.allclose(fitted, share, atol=1e-4)


def test_grouped_weights_keep_groups_together_for_the_rate_bootstrap():
    groups = ["a", "a", "b", "c", "c"]
    W = boot_weights_grouped(groups, B=40, seed=2)
    assert (W[:, 0] == W[:, 1]).all() and (W[:, 3] == W[:, 4]).all()
    v = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    assert np.all(np.isfinite(boot_mean(v, W)))


def test_analysis_is_deterministic_and_matches_committed_results():
    _, _, h1, r1, f1, p1 = R.analyse()
    _, _, h2, r2, f2, p2 = R.analyse()
    assert r1 == r2 and f1 == f2 and h1 == h2 and p1 == p2
    committed = _rows("rates.csv")
    assert len(committed) == len(r1)
    for a, b in zip(committed, r1):
        assert a["dataset"] == b["dataset"] and a["judge"] == b["judge"] and a["estimator"] == b["estimator"]
        assert float(a["error_pp"]) == pytest.approx(b["error_pp"], abs=1e-6)


def test_committed_flips_and_hypotheses_follow_from_the_committed_rates():
    rates = {(r["dataset"], r["event"], r["judge"], r["estimator"]): r for r in _rows("rates.csv")}
    n_flip = 0
    for f in _rows("flips.csv"):
        r = rates[(f["dataset"], f["event"], f["judge"], f["estimator"])]
        tau = float(f["tau_pct"])
        flip = (float(r["rate_pct"]) <= tau) != (float(r["human_rate_pct"]) <= tau)
        assert (f["flip"] == "True") == flip
        assert (f["false_pass"] == "True") == (float(r["rate_pct"]) <= tau and not float(r["human_rate_pct"]) <= tau)
        n_flip += flip
    assert n_flip > 0
    h = json.load(open(EXP / "results" / "hypotheses.json"))
    prim = [r for r in rates.values() if r["event"] == "primary"]
    raw = [r for r in prim if r["estimator"] == "raw_soft"]
    assert h["H1"]["n"] == sum(float(r["abs_error_lo"]) > 5.0 for r in raw) and len(raw) == 25
    assert h["H1"]["decision"] == ("supported" if h["H1"]["n"] >= 10 else "falsified")
    by = {(r["dataset"], r["judge"], r["estimator"]): float(r["abs_error_pp"]) for r in prim}
    cells = {(r["dataset"], r["judge"]) for r in raw}
    assert h["H2"]["n"] == sum(by[(d, j, "calibrated_soft")] < by[(d, j, "raw_soft")] for d, j in cells)
    assert h["H3"]["n"] == sum(by[(d, j, "hard_label")] > by[(d, j, "calibrated_soft")] for d, j in cells)
    for est in ("raw_soft", "hard_label", "calibrated_soft"):
        assert h["gate_flips"][f"primary:{est}"]["gates"] == 125
    for est in ("raw_soft", "hard_label", "calibrated_soft"):
        assert h["gate_flips"][f"secondary:{est}"]["gates"] == 50
