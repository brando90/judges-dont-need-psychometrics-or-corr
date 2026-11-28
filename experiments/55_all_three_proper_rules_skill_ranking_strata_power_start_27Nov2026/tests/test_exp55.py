"""Deterministic checks for Experiment 55 (fixed seeds; named tolerances; claimed-false statements tested as failing cases)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("run_all_rules", HERE.parent / "scripts" / "run_all_rules.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
L = M.L
TOL = 1e-12


def test_binary_nrps_is_half_brier():
    rng = np.random.default_rng(1)
    q = rng.dirichlet([1, 1], size=200)
    assert np.allclose(L.score_matrix(q, "nrps"), L.score_matrix(q, "brier") / 2.0, atol=TOL)


def test_three_category_nrps_is_not_brier_over_anything_simple():
    """Claimed false: with three categories, nrps is a fixed multiple of Brier. A forecast pair where the ratio differs."""
    q = np.array([[0.6, 0.0, 0.4], [0.2, 0.6, 0.2]])
    y = 0
    ratio = L.score_matrix(q, "nrps")[:, y] / L.score_matrix(q, "brier")[:, y]
    assert abs(ratio[0] - ratio[1]) > 1e-3


@pytest.mark.parametrize("rule", ["brier", "ce", "nrps"])
def test_strict_propriety_on_random_laws(rule):
    rng = np.random.default_rng(2)
    for _ in range(50):
        p = rng.dirichlet([2, 2, 2])
        truth = float((p * L.score_matrix(p, rule)).sum())
        for _ in range(40):
            q = rng.dirichlet([2, 2, 2])
            assert truth < float((p * L.score_matrix(q, rule)).sum())


def test_same_mean_forecasts_tie_on_mean_but_not_on_proper_scores():
    """Panel split 50/50 on {0, 4} on a 0..5 scale. Truthful forecast versus a certain forecast at the mean 2."""
    p = np.array([0.5, 0, 0, 0, 0.5, 0])
    c = np.array([0, 0, 1.0, 0, 0, 0])
    support = np.arange(6)
    assert (p * support).sum() == (c * support).sum() == 2.0
    br_p, br_c = float((p * L.score_matrix(p, "brier")).sum()), float((p * L.score_matrix(c, "brier")).sum())
    rps_p, rps_c = float((p * L.score_matrix(p, "nrps")).sum()) * 5, float((p * L.score_matrix(c, "nrps")).sum()) * 5
    assert br_p == pytest.approx(0.5, abs=TOL) and br_c == pytest.approx(2.0, abs=TOL)
    assert rps_p == pytest.approx(1.0, abs=TOL) and rps_c == pytest.approx(2.0, abs=TOL)
    assert rps_p / 5 == pytest.approx(0.2, abs=TOL) and rps_c / 5 == pytest.approx(0.4, abs=TOL)


def test_crossfit_base_uses_only_the_other_folds():
    rng = np.random.default_rng(3)
    counts = rng.integers(0, 30, size=(60, 3)).astype(float)
    base, fold = M.crossfit_base(counts, 0)
    for f in range(M.NFOLD):
        h = counts[fold != f].sum(0)
        expect = M.floor_q(h / h.sum())
        assert np.allclose(base[fold == f], expect, atol=TOL)
        # claimed false: the base uses the held-out fold's own ratings
        own = counts[fold == f].sum(0)
        assert not np.allclose(base[fold == f][0], M.floor_q(own / own.sum()), atol=1e-6)


def test_outputs_identical_across_two_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "B", 200)
    a, b = tmp_path / "a", tmp_path / "b"
    M.main(a)
    M.main(b)
    for name in ["skill_all_rules.csv", "rank_agreement.csv", "strata_all_rules.csv", "mdd_all_rules.csv", "summary.json", "h2_cells.csv"]:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_committed_results_state_the_preregistered_outcomes():
    import json
    s = json.loads((HERE.parent / "results" / "summary.json").read_text())
    assert s["B"] == 4000 and s["folds"] == 5 and s["quintiles"] == 5
    assert set(s["h1_dices350_n_significantly_below_base_by_rule"]) == {"brier", "ce", "nrps"}
