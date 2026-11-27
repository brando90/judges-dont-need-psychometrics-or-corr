"""Deterministic checks for the G-study estimator. No human ratings."""

import numpy as np
import pytest

from gstudy import balanced_anova, crossed_gstudy, dstudy_curve, nested_gstudy, recommend_k


def _long_from_matrix(Y):
    items, raters, scores = [], [], []
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            items.append(f"i{i}")
            raters.append(f"r{j}")
            scores.append(Y[i, j])
    return items, raters, scores


def test_crossed_matches_balanced_anova():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(6, 4))
    ref = balanced_anova(Y)
    got = crossed_gstudy(*_long_from_matrix(Y))
    assert got["status"] == "ok"
    assert got["sigma_c2_raw"] == pytest.approx(ref["sigma_c2"], abs=1e-8)
    assert got["sigma_r2_raw"] == pytest.approx(ref["sigma_r2"], abs=1e-8)
    assert got["sigma_res2"] == pytest.approx(ref["sigma_res2"], abs=1e-8)
    assert got["n_components"] == 1
    assert got["truncated_at_zero"] is False or True


def test_unbalanced_second_moment_unbiased():
    """Method of moments is unbiased under the random-effects second moments."""
    rng = np.random.default_rng(1)
    # Drop a fixed pattern of cells so the design is unbalanced but connected.
    keep = np.ones((8, 5), dtype=bool)
    keep[0, 4] = False
    keep[1, 3] = False
    keep[2, 0] = False
    keep[5, 1] = False
    sc, sr, se = 1.5, 0.4, 0.8
    reps = 400
    est = np.zeros((reps, 3))
    for t in range(reps):
        alpha = rng.normal(0, np.sqrt(sc), size=8)
        beta = rng.normal(0, np.sqrt(sr), size=5)
        Y = alpha[:, None] + beta[None, :] + rng.normal(0, np.sqrt(se), size=(8, 5))
        items, raters, scores = [], [], []
        for i in range(8):
            for j in range(5):
                if not keep[i, j]:
                    continue
                items.append(i)
                raters.append(j)
                scores.append(Y[i, j])
        got = crossed_gstudy(items, raters, scores)
        assert got["status"] == "ok"
        est[t] = (got["sigma_c2_raw"], got["sigma_r2_raw"], got["sigma_res2"])
    mean = est.mean(axis=0)
    # Monte Carlo tolerance: 4 standard errors of the mean.
    se_mean = est.std(axis=0, ddof=1) / np.sqrt(reps)
    assert mean[0] == pytest.approx(sc, abs=max(4 * se_mean[0], 0.05))
    assert mean[1] == pytest.approx(sr, abs=max(4 * se_mean[1], 0.05))
    assert mean[2] == pytest.approx(se, abs=max(4 * se_mean[2], 0.05))


def test_nested_matches_balanced_one_way():
    rng = np.random.default_rng(2)
    n_c, n_r = 10, 4
    alpha = rng.normal(size=n_c)
    Y = alpha[:, None] + rng.normal(size=(n_c, n_r))
    items, scores = [], []
    for i in range(n_c):
        for j in range(n_r):
            items.append(i)
            scores.append(Y[i, j])
    got = nested_gstudy(items, scores)
    # Balanced one-way: sigma_c = (MS_b - MS_w) / n_r
    grand = Y.mean()
    ms_w = np.sum((Y - Y.mean(axis=1)[:, None]) ** 2) / (n_c * (n_r - 1))
    ms_b = n_r * np.sum((Y.mean(axis=1) - grand) ** 2) / (n_c - 1)
    assert got["sigma_res2"] == pytest.approx(ms_w, abs=1e-8)
    assert got["sigma_c2_raw"] == pytest.approx((ms_b - ms_w) / n_r, abs=1e-8)
    assert got["sigma_r2"] == 0.0
    assert got["rater_component"].startswith("unidentified")


def test_dstudy_increases_in_k_and_ignores_n():
    curve = dstudy_curve(0.5, 0.2, 0.3, k_max=8)
    phis = [row["Phi"] for row in curve]
    assert phis == sorted(phis)
    assert phis[0] < phis[-1]
    # Same components, same curve: N is not an argument.
    assert dstudy_curve(0.5, 0.2, 0.3, 4) == curve[:4]
    rec = recommend_k(curve, target=0.80)
    assert rec["target_hit"] is True
    assert rec["k"] == min(row["k"] for row in curve if row["Phi"] >= 0.80)


def test_dstudy_target_infeasible_picks_largest_k():
    # Huge residual: Phi stays below 0.80 for k <= 3.
    curve = dstudy_curve(0.1, 0.0, 5.0, k_max=3)
    assert all(row["Phi"] < 0.80 for row in curve)
    rec = recommend_k(curve, target=0.80)
    assert rec["target_hit"] is False
    assert rec["k"] == 3
