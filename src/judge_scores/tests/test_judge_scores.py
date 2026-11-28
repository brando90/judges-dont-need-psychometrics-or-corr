"""Deterministic tests for judge_scores (fixed seeds; exact enumeration or a named tolerance; claimed-false statements as failing cases).

Run from src/judge_scores:  python -m pytest -q
"""
import importlib.util
import itertools
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from judge_scores import (RULES, agreement_vs_skill_report, base_rate_forecast, calibration_split, crossfit_base_rate,  # noqa: E402
                          floor_probs, item_scores, make_folds, make_group_folds, matched_rule, ratings_to_counts, risk,
                          scores_vs_distribution, skill)
from judge_scores.calibration import apply_calibrator, fit_calibrator  # noqa: E402

REPO = PKG.parents[1]


# --------------------------------------------------------------------------- per-rating definitions, written independently
def per_rating(q, y, rule):
    K = len(q)
    if rule == "brier":
        return sum((q[k] - (1.0 if k == y else 0.0)) ** 2 for k in range(K))
    if rule == "ce":
        return -float(np.log(q[y]))
    Q = np.cumsum(q)
    return sum((Q[m] - (1.0 if y <= m else 0.0)) ** 2 for m in range(K - 1)) / (K - 1)


# --------------------------------------------------------------------------- (1) strict propriety
@pytest.mark.parametrize("K", [2, 3, 4, 6])
def test_strict_propriety_random_laws(K):
    rng = np.random.default_rng(100 + K)
    for _ in range(300):
        p, q = rng.dirichlet(np.ones(K)), rng.dirichlet(np.ones(K))
        if np.abs(p - q).max() < 1e-3:
            continue
        for rule in RULES:
            sp = scores_vs_distribution(p[None], p[None], rule)[0]
            sq = scores_vs_distribution(q[None], p[None], rule)[0]
            assert sq > sp, (rule, p, q)


def test_claimed_false_mean_only_scores_are_proper():
    """A score that sees only the forecast mean cannot separate two forecasts with the same mean; the strictly proper scores can."""
    p = np.array([0.5, 0.0, 0.5])  # truth, mean 1
    q = np.array([0.0, 1.0, 0.0])  # same mean 1, different distribution
    v = np.arange(3.0)
    assert abs(p @ v - q @ v) < 1e-15
    mse_p = sum(p[y] * (y - p @ v) ** 2 for y in range(3))  # scalar MSE of the mean readout against the rating
    mse_q = sum(p[y] * (y - q @ v) ** 2 for y in range(3))
    assert abs(mse_p - mse_q) < 1e-15  # the tie
    for rule in RULES[::2]:  # brier, nrps (ce needs floored q)
        assert scores_vs_distribution(p[None], p[None], rule)[0] < scores_vs_distribution(q[None], p[None], rule)[0]


# --------------------------------------------------------------------------- (2) skill of the base rate is exactly 0
@pytest.mark.parametrize("rule", RULES)
def test_base_rate_skill_is_zero(rule):
    rng = np.random.default_rng(7)
    counts = rng.integers(0, 6, size=(40, 3)).astype(float) + np.array([1.0, 0, 0])
    b = np.tile(base_rate_forecast(counts), (40, 1))
    assert skill(risk(b, counts, rule), risk(b, counts, rule)) == 0.0
    with pytest.raises(ValueError):
        skill(0.3, 0.0)


# --------------------------------------------------------------------------- (3) the proper-score decomposition identity
def _bregman(rule, q, p):
    return scores_vs_distribution(q[None], p[None], rule)[0] - scores_vs_distribution(p[None], p[None], rule)[0]


@pytest.mark.parametrize("rule", RULES)
def test_population_decomposition_identity_finite_law(rule):
    """E S(qhat, Y) = E S(pbar, Y) - E d(pbar, p_qhat) + E d(qhat, p_qhat) on random finite joint laws of (forecast, outcome)."""
    rng = np.random.default_rng(11)
    for _ in range(100):
        K, J = int(rng.integers(2, 5)), int(rng.integers(2, 6))
        qs = floor_probs(rng.dirichlet(np.ones(K), size=J))  # J distinct issued forecasts
        joint = rng.dirichlet(np.ones(J * K)).reshape(J, K)  # P(qhat = j, Y = k)
        wj, pbar = joint.sum(1), joint.sum(0)
        p_cond = joint / wj[:, None]  # p_{qhat=j}
        lhs = sum(wj[j] * scores_vs_distribution(qs[j][None], p_cond[j][None], rule)[0] for j in range(J))
        unc = scores_vs_distribution(floor_probs(pbar)[None], pbar[None], rule)[0]
        # E d(pbar_floored, p_j): pbar must be a valid forecast (floored) for the CE case
        pb = floor_probs(pbar)
        res = sum(wj[j] * _bregman(rule, pb, p_cond[j]) for j in range(J))
        rel = sum(wj[j] * _bregman(rule, qs[j], p_cond[j]) for j in range(J))
        base_risk = sum(wj[j] * scores_vs_distribution(pb[None], p_cond[j][None], rule)[0] for j in range(J))
        assert abs(lhs - (base_risk - res + rel)) < 1e-10, (rule, lhs, base_risk, res, rel)
        assert unc <= base_risk + 1e-9  # flooring cannot help the base rate


def test_sample_decomposition_identity_and_determinism():
    rng = np.random.default_rng(3)
    q = floor_probs(rng.dirichlet(np.ones(3), size=60))
    counts = rng.multinomial(9, [0.5, 0.3, 0.2], size=60).astype(float)
    folds = make_folds(60, 5, seed=1)
    a = calibration_split(q, counts, "nrps", folds)
    b = calibration_split(q, counts, "nrps", folds)
    assert abs(a["identity_residual"]) < 1e-12
    assert abs(a["skill_raw"] - (a["resolution"] - a["calibration_error"]) / a["unc"]) < 1e-12
    assert abs(a["skill_cal"] - a["resolution"] / a["unc"]) < 1e-12
    for k in ("unc", "risk_raw", "risk_cal", "calibration_error", "resolution"):
        assert a[k] == b[k]  # two runs, identical output


# --------------------------------------------------------------------------- (4) hard-label reductions
def test_hard_forecast_reductions():
    """Brier/2 of a point mass is the 0-1 error; RPS (unnormalized) of a point mass is the absolute error (K-1 times nRPS)."""
    K = 5
    for a, y in itertools.product(range(K), range(K)):
        q = np.zeros((1, K))
        q[0, a] = 1.0
        h = np.zeros((1, K))
        h[0, y] = 1.0
        assert scores_vs_distribution(q, h, "brier")[0] / 2.0 == (0.0 if a == y else 1.0)
        assert abs(scores_vs_distribution(q, h, "nrps")[0] * (K - 1) - abs(a - y)) < 1e-12


# --------------------------------------------------------------------------- (5) one rating per item is unbiased (exact enumeration)
@pytest.mark.parametrize("rule", RULES)
def test_one_rating_per_item_is_unbiased_by_exact_enumeration(rule):
    ps = [np.array([0.5, 0.3, 0.2]), np.array([0.1, 0.6, 0.3]), np.array([0.25, 0.25, 0.5])]
    qs = [floor_probs(np.array(x)) for x in ([0.4, 0.4, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5])]
    exact = 0.0
    for ys in itertools.product(range(3), repeat=3):
        prob = np.prod([ps[i][ys[i]] for i in range(3)])
        exact += prob * np.mean([per_rating(qs[i], ys[i], rule) for i in range(3)])
    pkg = scores_vs_distribution(np.stack(qs), np.stack(ps), rule).mean()
    assert abs(exact - pkg) < 1e-12


def test_rational_exactness_brier():
    """Exact-fraction check of the item-level Brier formula against the per-rating definition."""
    q = [Fraction(2, 5), Fraction(1, 5), Fraction(2, 5)]
    p = [Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)]
    lhs = sum(p[y] * sum((q[k] - (1 if k == y else 0)) ** 2 for k in range(3)) for y in range(3))
    rhs = sum(x * x for x in q) - 2 * sum(a * b for a, b in zip(q, p)) + 1
    assert lhs == rhs


# --------------------------------------------------------------------------- (6) reproduce published Experiment 55 numbers
def _load_bundle():
    path = next((REPO / "experiments").glob("39_*/data/processed/study_a_bundle.npz"), None)
    if path is None or not path.exists():
        pytest.skip("Experiment 39 bundle not present (run experiments/39_*/src/prepare_data.py); skipped, not passed")
    return np.load(path, allow_pickle=True)["bundle"].item()


def test_reproduces_experiment55_skill_rows():
    import csv
    csv_path = next((REPO / "experiments").glob("55_*/results/skill_all_rules.csv"), None)
    if csv_path is None:
        pytest.skip("Experiment 55 results not present")
    bundle = _load_bundle()
    datasets = ["chaosnli_snli", "chaosnli_mnli", "chaosnli_alphanli", "dices350", "dices990"]
    judges = ["gpt-6-astra", "claude-opus-5-5", "kimi-k3", "glm-5.3", "deepseek-v4.1-flash"]
    published = {(r["dataset"], r["rule"], r["judge"]): r for r in csv.DictReader(open(csv_path))}
    checked = 0
    for ds_idx, name in enumerate(datasets):
        d = bundle[name]
        counts = np.asarray(d["counts"], float)
        N, K = counts.shape
        P = np.array(d["p_judges"], float)
        P[~np.asarray(d["judge_valid"], bool)] = 1.0 / K
        P = floor_probs(P)
        rng = np.random.default_rng(np.random.SeedSequence([20261127, 55, 0, ds_idx]))
        perm = rng.permutation(N)
        fold = np.empty(N, int)
        fold[perm] = np.arange(N) % 5
        base = crossfit_base_rate(counts, fold)
        for rule in RULES:
            for j, jn in enumerate(judges):
                row = published[(name, rule, jn)]
                r, rb = risk(P[:, j], counts, rule), risk(base, counts, rule)
                assert abs(r - float(row["risk"])) < 1e-8, (name, rule, jn)
                assert abs(skill(r, rb) - float(row["skill"])) < 1e-8, (name, rule, jn)
                checked += 1
    assert checked == 75


def test_calibrator_matches_experiment46_code():
    path = next((REPO / "experiments").glob("46_*/scripts/analyze_rank_reversal.py"), None)
    if path is None:
        pytest.skip("Experiment 46 script not present")
    spec = importlib.util.spec_from_file_location("a46", path)
    a46 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a46)
    rng = np.random.default_rng(5)
    q = floor_probs(rng.dirichlet(np.ones(3), size=80))
    cnt = rng.multinomial(10, [0.5, 0.3, 0.2], size=80).astype(float)
    W1, W2 = fit_calibrator(q, cnt), a46.fit_cal(q, cnt)
    assert np.abs(W1 - W2).max() < 1e-10
    assert np.abs(apply_calibrator(W1, q) - a46.apply_cal(W2, q)).max() < 1e-12


# --------------------------------------------------------------------------- (7) refusing the ordinal score on unordered labels
def test_refuses_ordinal_rps_on_unordered_labels():
    rng = np.random.default_rng(2)
    q = rng.dirichlet(np.ones(3), size=30)
    counts = rng.multinomial(8, [0.4, 0.4, 0.2], size=30).astype(float)
    with pytest.raises(ValueError):
        agreement_vs_skill_report(q, counts, ordered=False, rule="nrps", B=50)
    assert matched_rule(3, ordered=False) == "brier" and matched_rule(3, ordered=True) == "nrps" and matched_rule(2, True) == "brier"


# --------------------------------------------------------------------------- (8) determinism and the CLI
def test_report_is_deterministic_and_sensible():
    rng = np.random.default_rng(9)
    counts = rng.multinomial(12, [0.5, 0.3, 0.2], size=50).astype(float)
    q = floor_probs(0.7 * counts / 12 + 0.3 * np.array([0.5, 0.3, 0.2]))  # a good judge
    a = agreement_vs_skill_report(q, counts, ordered=True, B=200, seed=4, calibrate=True)
    b = agreement_vs_skill_report(q, counts, ordered=True, B=200, seed=4, calibrate=True)
    assert json.dumps(a, default=float, sort_keys=True) == json.dumps(b, default=float, sort_keys=True)
    assert a["skill"]["estimate"] > 0 and a["skill"]["ci_lo"] < a["skill"]["estimate"] < a["skill"]["ci_hi"]
    assert a["pearson"]["estimate"] > 0.5


def test_cli_smoke(tmp_path):
    rng = np.random.default_rng(1)
    K, N = 3, 20
    with open(tmp_path / "f.csv", "w") as f, open(tmp_path / "r.csv", "w") as g:
        f.write("item_id,p0,p1,p2\n")
        g.write("item_id,rating\n")
        for i in range(N):
            p = rng.dirichlet(np.ones(K))
            f.write(f"i{i}," + ",".join(f"{x:.6f}" for x in p / p.sum()) + "\n")
            for y in rng.choice(K, size=7, p=[0.5, 0.3, 0.2]):
                g.write(f"i{i},{y}\n")
    out = subprocess.run([sys.executable, "-m", "judge_scores.cli", "report", str(tmp_path / "f.csv"), str(tmp_path / "r.csv"),
                          "--labels", "3", "--ordered", "--draws", "50"], capture_output=True, text=True, cwd=str(PKG))
    assert out.returncode == 0, out.stderr
    assert "skill" in json.loads(out.stdout)
    bad = subprocess.run([sys.executable, "-m", "judge_scores.cli", "report", str(tmp_path / "f.csv"), str(tmp_path / "r.csv"),
                          "--labels", "3", "--rule", "nrps", "--draws", "50"], capture_output=True, text=True, cwd=str(PKG))
    assert bad.returncode != 0 and "ordered" in bad.stderr


def test_group_folds_keep_groups_together_and_are_deterministic():
    groups = [f"g{i // 3}" for i in range(31)]  # groups of three items, last group has one item
    a = make_group_folds(groups, 5, seed=np.random.SeedSequence([1, 2, 3]))
    b = make_group_folds(groups, 5, seed=np.random.SeedSequence([1, 2, 3]))
    assert (a == b).all()
    for g in set(groups):
        assert len({a[i] for i, x in enumerate(groups) if x == g}) == 1
    assert set(a) == {0, 1, 2, 3, 4}


def test_grouped_bootstrap_moves_groups_together():
    from judge_scores import boot_weights_grouped
    groups = ["a", "a", "b", "c", "c", "c"]
    W = boot_weights_grouped(groups, B=50, seed=3)
    assert W.shape == (50, 6)
    assert (W[:, 0] == W[:, 1]).all() and (W[:, 3] == W[:, 4]).all() and (W[:, 4] == W[:, 5]).all()
    assert (W[:, [0, 2, 3]].sum(1) == 3).all()  # three groups drawn per resample
    assert (boot_weights_grouped(groups, B=50, seed=3) == W).all()
