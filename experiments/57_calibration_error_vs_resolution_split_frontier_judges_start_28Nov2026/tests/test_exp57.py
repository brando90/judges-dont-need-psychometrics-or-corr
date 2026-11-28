"""Tests for Experiment 57 (run: python -m pytest experiments/57_*/tests -q).

Synthetic tests are labeled as code checks of known laws, not as evidence about real judges. Tests that read results/ check
that the committed numbers are internally consistent and agree with Experiment 55's committed raw skill."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import kendalltau, rankdata, spearmanr

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "scripts"))
import run_exp57 as R  # noqa: E402
from judge_scores import calibration_split, floor_probs, make_folds, risk  # noqa: E402


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ statistics used by the analysis
def test_tau_b_rows_matches_scipy_including_ties():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 3, size=(200, 5)).astype(float)  # many ties
    Y = rng.integers(0, 4, size=(200, 5)).astype(float)
    ours = R.tau_b_rows(X, Y)
    for i in range(len(X)):
        theirs = kendalltau(X[i], Y[i]).statistic
        if np.isnan(theirs):
            assert np.isnan(ours[i])
        else:
            assert ours[i] == pytest.approx(theirs, abs=1e-12)


def test_weighted_pearson_and_spearman_match_expanded_data():
    rng = np.random.default_rng(2)
    N, J = 40, 3
    X, y = rng.normal(size=(N, J)), rng.normal(size=N)
    W = rng.integers(0, 4, size=(6, N)).astype(float)
    pw, sw = R.weighted_pearson(X, y, W), R.spearman_draws(X, y, W)
    for r in range(len(W)):
        idx = np.repeat(np.arange(N), W[r].astype(int))
        for j in range(J):
            assert pw[r, j] == pytest.approx(np.corrcoef(X[idx, j], y[idx])[0, 1], abs=1e-10)
            assert sw[r, j] == pytest.approx(spearmanr(X[idx, j], y[idx]).statistic, abs=1e-10)


def test_holm_adjust_known_values():
    assert np.allclose(R.holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(R.holm_adjust([0.5]), [0.5])


# ------------------------------------------------------------------ the split on synthetic known laws (code checks)
def _known_law(n=2000, r=10, seed=57):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet([1.2, 1.2, 1.2], size=n)  # the true p*(y | item), known
    counts = np.stack([np.bincount(rng.choice(3, size=r, p=pi), minlength=3) for pi in p]).astype(float)
    return p, counts


def test_known_law_truthful_forecast_has_no_calibration_error_and_overconfident_one_does():
    p, counts = _known_law()
    folds = make_folds(len(p), 5, seed=3)
    truthful = calibration_split(floor_probs(p), counts, "brier", folds)
    sharp = floor_probs(p ** 2 / (p ** 2).sum(1, keepdims=True))  # same ordering of labels, overconfident
    over = calibration_split(sharp, counts, "brier", folds)
    assert abs(truthful["calibration_share"]) < 0.01  # estimation noise only (threshold fixed before looking at the numbers)
    assert over["calibration_share"] > 0.02  # the map family contains the exact fix (temperature 1/2)
    assert over["calibration_share"] > truthful["calibration_share"] + 0.02
    for s in (truthful, over):
        assert abs(s["identity_residual"]) < 1e-12


def test_forecasts_sharing_a_mean_tie_on_correlation_but_not_on_the_proper_score():
    p, counts = _known_law(n=600, seed=11)
    t = 0.9 * p[:, 1] / 2.0  # moves mass from the middle label to both ends: the forecast mean is unchanged
    spread = np.column_stack([p[:, 0] + t, p[:, 1] - 2 * t, p[:, 2] + t])
    mean = lambda q: q @ np.arange(3)  # noqa: E731
    human = (counts * np.arange(3)).sum(1) / counts.sum(1)
    assert np.allclose(mean(p), mean(spread), atol=1e-12)
    r_p = R.weighted_pearson(mean(p)[:, None], human, np.ones((1, len(p))))[0, 0]
    r_s = R.weighted_pearson(mean(spread)[:, None], human, np.ones((1, len(p))))[0, 0]
    assert r_p == pytest.approx(r_s, abs=1e-12)  # identical correlation
    assert risk(floor_probs(spread), counts, "brier") > risk(floor_probs(p), counts, "brier") + 0.01  # but a clearly worse forecast
    assert risk(floor_probs(spread), counts, "ce") > risk(floor_probs(p), counts, "ce") + 0.01


def test_a_wrong_identity_fails():
    p, counts = _known_law(n=500, seed=5)
    s = calibration_split(floor_probs(p ** 2 / (p ** 2).sum(1, keepdims=True)), counts, "brier", make_folds(500, 5, seed=1))
    wrong = s["unc"] + s["resolution"] + s["calibration_error"]  # claimed false: the sign of the resolution term is wrong
    assert abs(s["risk_raw"] - wrong) > 1e-3


def test_split_is_deterministic():
    p, counts = _known_law(n=400, seed=8)
    folds = make_folds(400, 5, seed=2)
    a = calibration_split(floor_probs(p), counts, "nrps", folds)
    b = calibration_split(floor_probs(p), counts, "nrps", folds)
    for k in ("unc", "risk_raw", "risk_cal", "calibration_error", "resolution"):
        assert a[k] == b[k]


# ------------------------------------------------------------------ committed results
def test_results_identity_and_decisions_are_internally_consistent():
    cells = _rows(EXP / "results" / "cells.csv")
    assert len(cells) == 5 * (3 * 2 + 2 * 3)
    for r in cells:
        assert abs(float(r["identity_residual"])) < 1e-12
        unc, raw, cal = (float(r[k]) for k in ("uncertainty_risk", "raw_risk", "calibrated_risk"))
        assert float(r["resolution"]) == pytest.approx(unc - cal, abs=1e-8)
        assert float(r["calibration_error"]) == pytest.approx(raw - cal, abs=1e-8)
        assert float(r["skill_raw"]) == pytest.approx(1 - raw / unc, abs=1e-8)
        assert float(r["skill_cal"]) == pytest.approx(1 - cal / unc, abs=1e-8)
        assert float(r["calibration_share"]) + float(r["skill_raw"]) == pytest.approx(float(r["resolution_share"]), abs=1e-8)
    h = json.load(open(EXP / "results" / "hypotheses.json"))
    n1 = sum(v["h1_set_supports"] for v in h["H1"]["sets"].values())
    n2 = sum(v["h2_set_supports"] for v in h["H2"]["sets"].values())
    assert n1 == h["H1"]["n_sets_supporting"] and h["H1"]["decision"] == ("supported" if n1 >= 4 else "falsified")
    assert n2 == h["H2"]["n_sets_supporting"] and h["H2"]["decision"] == ("supported" if n2 >= 4 else "falsified")
    assert h["draws"] == 4000


def test_raw_skill_agrees_with_experiment_55_within_001():
    mine = {(r["dataset"], r["rule"], r["judge"]): float(r["skill_raw"]) for r in _rows(EXP / "results" / "cells.csv")}
    theirs = {(r["dataset"], r["rule"], r["judge"]): float(r["skill"])
              for r in _rows(next((EXP.parent).glob("55_*/results/skill_all_rules.csv")))}
    common = sorted(set(mine) & set(theirs))
    assert len(common) == len(mine)  # every cell here has an Experiment 55 counterpart
    worst = max(abs(mine[k] - theirs[k]) for k in common)
    assert worst < 0.01, worst  # base-rate folds differ (group folds here, item folds there)
