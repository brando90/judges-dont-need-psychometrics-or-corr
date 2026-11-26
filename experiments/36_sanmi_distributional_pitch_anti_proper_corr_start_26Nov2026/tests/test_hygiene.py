"""Unit tests for the unbiased risk, the prompt split, and the metric identities."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_io import assert_no_prompt_leakage, split_by_group
from metrics import (
    brier_one,
    ce_one,
    expected_plugin_bias,
    item_mean_losses,
    plugin_l2,
    rps_one,
    unbiased_risk,
)
from proofs_numeric import check_all


def test_unbiased_risk_matches_the_prompt_average():
    # Two prompts. Losses 1 and 3 on the first (mean 2), loss 4 on the second.
    # The estimator weights prompts equally: (2 + 4) / 2 = 3, not (1+3+4)/3.
    q = np.array([0.2, 0.8])
    # Build losses explicitly rather than through a forecast, by checking the formula.
    item_means = np.array([np.mean([1.0, 3.0]), np.mean([4.0])])
    assert unbiased_risk(item_means) == 3.0
    # And the per-prompt helper averages ratings, not prompts.
    # y=1 has CE -log(0.8); y=0 has CE -log(0.2).
    got = item_mean_losses(q, np.array([1, 0]))
    assert abs(got["ce"] - 0.5 * (ce_one(q, 1) + ce_one(q, 0))) < 1e-12
    assert abs(got["brier"] - 0.5 * (brier_one(q, 1) + brier_one(q, 0))) < 1e-12
    assert abs(got["rps"] - 0.5 * (rps_one(q, 1) + rps_one(q, 0))) < 1e-12


def test_plugin_bias_formula():
    p = np.array([0.2, 0.5, 0.3])
    n = 10
    theory = (1 - np.dot(p, p)) / n
    assert abs(expected_plugin_bias(p, n) - theory) < 1e-12
    q = np.array([0.5, 0.25, 0.25])
    # Monte Carlo: E[||freq - q||^2] = ||p-q||^2 + theory
    rng = np.random.default_rng(0)
    plugins = []
    for _ in range(3000):
        freq = np.bincount(rng.choice(3, size=n, p=p), minlength=3) / n
        plugins.append(plugin_l2(freq, q))
    assert abs(np.mean(plugins) - (plugin_l2(p, q) + theory)) < 0.01


def test_binary_brier_is_twice_scalar_l2():
    for q in (0.2, 0.5, 0.9):
        for y in (0, 1):
            dist = np.array([1 - q, q])
            assert abs(brier_one(dist, y) - 2 * (q - y) ** 2) < 1e-12


def test_prompt_split_has_no_rater_leakage():
    groups = ["a", "a", "b", "c", "c", "d", "e", "f", "g", "h"]
    rater_ids = [[f"{g}-{j}" for j in range(3)] for g in groups]
    train, val, test = split_by_group(groups, seed=0)
    assert_no_prompt_leakage(groups, rater_ids, train, val, test)
    # No group id on two sides.
    def sides(idx):
        return {groups[i] for i in idx}
    assert sides(train).isdisjoint(sides(val))
    assert sides(train).isdisjoint(sides(test))
    assert sides(val).isdisjoint(sides(test))


def test_numeric_proof_suite():
    check_all(seed=0)
