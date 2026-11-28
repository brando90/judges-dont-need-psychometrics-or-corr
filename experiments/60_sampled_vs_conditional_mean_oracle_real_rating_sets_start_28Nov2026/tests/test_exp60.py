"""Tests for Experiment 60 (run: python -m pytest experiments/60_*/tests -q).

Synthetic tests are CODE CHECKS of known laws, not evidence about real judges or real raters. Tests that read results/ check that the
committed numbers are internally consistent with the frozen protocol's decision rules."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "scripts"))
import run_exp60 as R  # noqa: E402
import frozen as Z  # noqa: E402  (found through run_exp60's path set-up)

LAB = np.arange(3.0)


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _law():
    """A finite law with K = 3 ordered labels: conditional distributions p*(.|c) for 4 items with weights w (exact, no sampling)."""
    P = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.4, 0.4, 0.2], [0.05, 0.15, 0.8]])
    w = np.array([0.1, 0.4, 0.3, 0.2])
    mu, m2 = P @ LAB, P @ LAB**2
    ey = w @ mu
    vy = w @ m2 - ey**2
    var_mu = w @ mu**2 - ey**2
    return P, w, mu, m2, vy, var_mu


def _exact_pearson(values, weights):
    """Pearson correlation of two coordinates under a finite weighted joint law; values [M, 2], weights [M] (brute force, independent of the closed forms)."""
    w = weights / weights.sum()
    ex, ey = w @ values[:, 0], w @ values[:, 1]
    cov = w @ ((values[:, 0] - ex) * (values[:, 1] - ey))
    return cov / np.sqrt((w @ (values[:, 0] - ex) ** 2) * (w @ (values[:, 1] - ey) ** 2))


# ------------------------------------------------------------------ the theorem on a known finite law (code check)
def test_known_law_sampled_judge_is_rho_and_mean_judge_is_sqrt_rho():
    P, w, mu, m2, vy, var_mu = _law()
    rho = var_mu / vy
    assert 0 < rho < 1
    r_mean, r_samp = R.corr_stats(w[None, :], np.ones(4), mu, m2, mu, m2)  # half A and half B are both the exact law
    assert r_samp[0] == pytest.approx(rho, abs=1e-12)
    assert r_mean[0] == pytest.approx(np.sqrt(rho), abs=1e-12)
    assert r_mean[0] > r_samp[0]


def test_known_law_theory_function_recovers_rho_as_raters_grow():
    P, w, mu, m2, vy, var_mu = _law()
    rho = var_mu / vy
    n = np.full(4, 1e12)
    rho_hat, r_pred = R.theory(w[None, :], n, mu, m2, n / 2)
    assert rho_hat[0] == pytest.approx(rho, abs=1e-9)
    assert r_pred[0] == pytest.approx(np.sqrt(rho), abs=1e-9)


def test_claims_that_are_false_fail():
    P, w, mu, m2, vy, var_mu = _law()
    rho = var_mu / vy
    r_mean, r_samp = R.corr_stats(w[None, :], np.ones(4), mu, m2, mu, m2)
    assert not np.isclose(r_samp[0], np.sqrt(rho), atol=1e-3)  # false: the sampled judge does not reach sqrt(rho)
    assert not np.isclose(r_mean[0], rho, atol=1e-3)  # false: the mean judge is not at rho
    assert not (r_samp[0] > r_mean[0])  # false: sampling cannot beat reading out the mean
    # a law with no within-item spread (every p* a point mass) has rho = 1 and the two judges coincide: the strict gap needs spread
    Pd = np.eye(3)[[0, 1, 2, 2]]
    mu_d, m2_d = Pd @ LAB, Pd @ LAB**2
    rm, rs = R.corr_stats(w[None, :], np.ones(4), mu_d, m2_d, mu_d, m2_d)
    assert rm[0] == pytest.approx(rs[0], abs=1e-12)


# ------------------------------------------------------------------ closed forms with noisy halves, against brute-force enumeration
def _tiny_split():
    rng = np.random.default_rng(5)
    N = 6
    cA = rng.integers(1, 6, size=(N, 3)).astype(float)
    cB = rng.integers(0, 6, size=(N, 3)).astype(float)
    cB[:, 0] += 1  # every item has at least one half-B rating
    return cA, cB


def test_closed_form_matches_brute_force_enumeration_of_the_joint_law():
    cA, cB = _tiny_split()
    pA = cA / cA.sum(1, keepdims=True)
    nB = cB.sum(1)
    st = dict(nB=nB, S1=cB @ LAB, S2=cB @ LAB**2, mu=pA @ LAB, m2=pA @ LAB**2)
    r_mean, r_samp = R.corr_stats(np.ones((1, 6)), st["nB"], st["S1"], st["S2"], st["mu"], st["m2"])
    # sampled judge: joint over (item, s, y) with weight nB_i * pA_i(s) * qB_i(y)
    vals_s, wts_s, vals_m, wts_m = [], [], [], []
    for i in range(6):
        qB = cB[i] / nB[i]
        for s in range(3):
            for y in range(3):
                vals_s.append((s, y))
                wts_s.append(nB[i] * pA[i, s] * qB[y])
        for y in range(3):
            vals_m.append((st["mu"][i], y))
            wts_m.append(nB[i] * qB[y])
    assert r_samp[0] == pytest.approx(_exact_pearson(np.array(vals_s, float), np.array(wts_s)), abs=1e-12)
    assert r_mean[0] == pytest.approx(_exact_pearson(np.array(vals_m, float), np.array(wts_m)), abs=1e-12)


def test_item_weights_equal_duplicating_the_items():
    cA, cB = _tiny_split()
    pA = cA / cA.sum(1, keepdims=True)
    nB = cB.sum(1)
    S1, S2, mu, m2 = cB @ LAB, cB @ LAB**2, pA @ LAB, pA @ LAB**2
    mult = np.array([[2, 0, 1, 3, 1, 1]], float)  # a bootstrap-style multiplicity vector
    a = R.corr_stats(mult, nB, S1, S2, mu, m2)
    idx = np.repeat(np.arange(6), mult[0].astype(int))
    b = R.corr_stats(np.ones((1, len(idx))), nB[idx], S1[idx], S2[idx], mu[idx], m2[idx])
    assert a[0][0] == pytest.approx(b[0][0], abs=1e-12) and a[1][0] == pytest.approx(b[1][0], abs=1e-12)


# ------------------------------------------------------------------ the split and the Monte Carlo check
def _fake_dataset(N=300, n=40, seed=11):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet([0.6, 0.6, 0.6], size=N)
    ratings = np.stack([rng.choice(3, size=n, p=P[i]) for i in range(N)])
    counts = np.stack([np.bincount(r, minlength=3) for r in ratings]).astype(float)
    return dict(counts=counts, K=3, ratings=ratings)


def test_split_is_a_partition_of_every_items_ratings():
    d = _fake_dataset(N=50, n=41)  # odd n: half A gets floor(41/2) = 20
    sp = R.split_once(d, np.random.default_rng(3))
    assert np.array_equal(sp["cA"] + sp["cB"], d["counts"])
    assert np.all(sp["cA"].sum(1) == 20) and np.all(sp["cB"].sum(1) == 21)


def test_padded_ratings_are_ignored():
    d = _fake_dataset(N=20, n=30)
    padded = np.concatenate([d["ratings"], -np.ones((20, 5), dtype=int)], axis=1)
    sp = R.split_once(dict(counts=d["counts"], K=3, ratings=padded), np.random.default_rng(3))
    assert np.array_equal(sp["cA"] + sp["cB"], d["counts"])


def test_closed_form_agrees_with_monte_carlo_sampling_of_scores():
    d = _fake_dataset()
    sp = R.split_once(d, np.random.default_rng(7), keep_obs=True)
    st = R.stats_from_split(sp, 3)
    _, r_cf = R.corr_stats(np.ones((1, 300)), st["nB"], st["S1"], st["S2"], st["mu"], st["m2"])
    reps = 200
    item = np.repeat(sp["obs_item"], reps)
    y = np.repeat(sp["obs_y"], reps)
    cum = np.cumsum(st["pA"], 1)[item]
    u = np.random.default_rng(8).random(len(item))
    s = np.minimum((u[:, None] > cum).sum(1), 2)
    assert np.corrcoef(s, y)[0, 1] == pytest.approx(r_cf[0], abs=0.005)  # named tolerance: Monte Carlo error is about 0.001 at 1.2 million pairs


def test_theory_estimates_the_true_rho_from_finite_ratings():
    P, w, mu, m2, vy, var_mu = _law()
    rng = np.random.default_rng(2)
    N, n = 4000, 40
    items = rng.choice(4, size=N, p=w)
    ratings = np.stack([rng.choice(3, size=n, p=P[c]) for c in items])
    counts = np.stack([np.bincount(r, minlength=3) for r in ratings]).astype(float)
    m, q = (counts @ LAB) / n, (counts @ LAB**2) / n
    rho_hat, _ = R.theory(np.ones((1, N)), np.full(N, float(n)), m, q, np.full(N, n // 2))
    assert rho_hat[0] == pytest.approx(var_mu / vy, abs=0.02)  # named tolerance for 4,000 items of 40 ratings


# ------------------------------------------------------------------ small helpers and determinism
def test_collapsed_forecast_is_the_point_mass_at_the_nearest_label_halves_round_up():
    q = R.collapsed_forecast(np.array([0.0, 0.49, 0.5, 1.5, 2.4, 5.0]), 3)
    assert list(q.argmax(1)) == [0, 0, 1, 2, 2, 2]
    assert np.allclose(q.sum(1), 1.0)


def test_analysis_is_deterministic(monkeypatch):
    monkeypatch.setattr(R, "SPLITS", 6)
    monkeypatch.setattr(R, "B", 60)
    monkeypatch.setattr(R, "MC_SPLITS", 2)
    a = R.analyse_set("dices350")
    b = R.analyse_set("dices350")
    assert a == b


# ------------------------------------------------------------------ committed results are consistent with the frozen protocol
def test_results_match_the_protocols_decision_rules():
    rows = _rows(EXP / "results" / "sampled_vs_mean.csv")
    assert [r["dataset"] for r in rows] == R.SETS
    hyp = json.loads((EXP / "results" / "hypotheses.json").read_text())
    f = lambda r, k: float(r[k])  # noqa: E731
    assert (hyp["H1"]["decision"] == "supported") == all(f(r, "diff_lo") > 0 for r in rows)
    ok = sum(abs(f(r, "sampled_minus_rho_hat")) <= 0.02 and abs(f(r, "mean_minus_prediction")) <= 0.02 for r in rows)
    assert hyp["H2"]["n_sets"] == ok and (hyp["H2"]["decision"] == "supported") == (ok >= 2)
    assert (hyp["H3"]["decision"] == "supported") == all(f(r, "risk_diff_lo") > 0 for r in rows)
    for r in rows:
        assert f(r, "sqrt_rho_hat") == pytest.approx(np.sqrt(f(r, "rho_hat")), abs=1e-8)
        assert f(r, "r_mean_minus_r_sampled") == pytest.approx(f(r, "r_mean_judge") - f(r, "r_sampled"), abs=1e-8)
        assert f(r, "risk_collapsed_minus_truthful") == pytest.approx(f(r, "risk_collapsed") - f(r, "risk_truthful"), abs=1e-8)
        for k in ("rho_hat", "r_sampled", "r_mean_judge", "skill_truthful", "skill_collapsed", "risk_diff", "diff"):
            lo, hi = (f(r, f"{k}_lo"), f(r, f"{k}_hi")) if f"{k}_lo" in r else (None, None)
            if lo is not None:
                point = f(r, k) if k in r else None
                if point is not None:
                    assert lo - 1e-9 <= point <= hi + 1e-9
        assert f(r, "mc_max_abs_diff") < 0.02  # the Monte Carlo check of the closed form (one draw per observation) stays within 0.02


def test_figure_values_file_matches_the_results_table():
    rows = _rows(EXP / "results" / "sampled_vs_mean.csv")
    fig = _rows(EXP / "results" / "figure1_values.csv")
    assert len(rows) == len(fig) == 3
    for r, g in zip(rows, fig):
        for k in ("rho_hat", "r_sampled", "r_mean_judge", "skill_truthful", "skill_collapsed"):
            assert float(g[k]) == pytest.approx(float(r[k]), abs=1e-8)
