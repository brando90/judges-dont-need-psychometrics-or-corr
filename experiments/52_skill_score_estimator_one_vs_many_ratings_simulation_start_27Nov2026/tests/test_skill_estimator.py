"""Deterministic checks for Experiment 52 (PROTOCOL.md predictions S1 and S2). Run: pytest -q tests/test_skill_estimator.py"""
import sys
from fractions import Fraction
from itertools import product
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import skill_estimator_sim as sim  # noqa: E402


def brier(q, y):
    return sum((qk - (1 if k == y else 0)) ** 2 for k, qk in enumerate(q))


def test_S1_one_rating_per_item_risk_is_exactly_unbiased():
    # two item types, K = 3, exact fractions: E_y[ (1/n) sum_i S(q_i, y_i) ] = (1/n) sum_i sum_k p_ik S(q_i, k)
    P = [[Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)], [Fraction(1, 10), Fraction(3, 10), Fraction(3, 5)]]
    Q = [[Fraction(2, 5), Fraction(2, 5), Fraction(1, 5)], [Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)]]
    exact = sum(sum(p[k] * brier(q, k) for k in range(3)) for p, q in zip(P, Q)) / 2
    enum = Fraction(0)
    for y0, y1 in product(range(3), repeat=2):
        enum += P[0][y0] * P[1][y1] * (brier(Q[0], y0) + brier(Q[1], y1)) / 2
    assert enum == exact


def test_S1_claimed_false_ratio_is_not_unbiased():
    # the RATIO of two unbiased means is not unbiased: exact enumeration on a one-item law with n = 2 ratings
    p = [Fraction(1, 2), Fraction(1, 2)]; q = [Fraction(9, 10), Fraction(1, 10)]; b = [Fraction(1, 2), Fraction(1, 2)]
    true_ratio = sum(p[k] * brier(q, k) for k in range(2)) / sum(p[k] * brier(b, k) for k in range(2))
    e_ratio = sum(p[a] * p[c] * (brier(q, a) + brier(q, c)) / (brier(b, a) + brier(b, c)) for a in range(2) for c in range(2))
    assert true_ratio == e_ratio  # the base score is constant here, so equality holds
    q2 = [Fraction(9, 10), Fraction(1, 10)]; b2 = [Fraction(4, 5), Fraction(1, 5)]
    true2 = sum(p[k] * brier(q2, k) for k in range(2)) / sum(p[k] * brier(b2, k) for k in range(2))
    e2 = sum(p[a] * p[c] * (brier(q2, a) + brier(q2, c)) / (brier(b2, a) + brier(b2, c)) for a in range(2) for c in range(2))
    assert e2 != true2  # a non-constant base score makes the ratio biased


def test_S2_ratio_bias_shrinks_like_one_over_n():
    sim.NS = [25, 1000]
    out = sim.run(None, 2000, seed=(20261127, 5299))
    r = {(x["n"], x["estimator"]): x for x in out["rows"] if x["score"] == "nrps" and x["judge"] == "J1_noisy_calibrated"}
    b25, b1000 = r[25, "E2_one_rating_insample_base"], r[1000, "E2_one_rating_insample_base"]
    assert abs(b25["bias"]) > 3 * b25["mcse_bias"]
    assert abs(b1000["bias"]) < abs(b25["bias"]) / 10 + 3 * b1000["mcse_bias"]


def test_identical_output_twice():
    sim.NS = [25]
    a = sim.run(None, 50, seed=(1, 2)); b = sim.run(None, 50, seed=(1, 2))
    assert a == b
