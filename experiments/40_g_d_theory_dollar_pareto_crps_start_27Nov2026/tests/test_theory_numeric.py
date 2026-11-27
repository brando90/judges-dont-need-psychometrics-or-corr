"""Deterministic numerical twin of the Lean witness. Exact fractions where the quantity is rational."""

from fractions import Fraction
import math

import numpy as np
import pytest


P = [Fraction(1, 4)] * 4
Q = [Fraction(1, 5), Fraction(2, 5), Fraction(1, 10), Fraction(3, 10)]


def _mean(v):
    return sum(Fraction(i) * v[i] for i in range(4))


def _second(v):
    return sum(Fraction(i * i) * v[i] for i in range(4))


def _cdf(v, t):
    return sum(v[: t + 1])


def test_witness_moments_and_scores():
    assert _mean(P) == _mean(Q) == Fraction(3, 2)
    assert _second(P) - _mean(P) ** 2 == Fraction(5, 4)
    assert _second(Q) - _mean(Q) ** 2 == Fraction(5, 4)
    brier_gap = sum((Q[j] - P[j]) ** 2 for j in range(4))
    assert brier_gap == Fraction(1, 20)
    rps_gap = sum((_cdf(Q, t) - _cdf(P, t)) ** 2 for t in range(3))
    assert rps_gap == Fraction(3, 200)
    assert math.prod(Q) < math.prod(P)
    ce_p = -sum(float(pi) * math.log(float(pi)) for pi in P)
    ce_q = -sum(float(P[i]) * math.log(float(Q[i])) for i in range(4))
    assert ce_q > ce_p
    assert ce_q - ce_p == pytest.approx(0.121776, abs=1e-5)


def test_same_mean_does_not_imply_same_brier():
    """The false claim 'equal means imply equal Brier' fails on the witness."""
    assert _mean(P) == _mean(Q)
    assert sum((Q[j] - P[j]) ** 2 for j in range(4)) > 0


def test_brier_and_rps_decomposition_on_random_laws():
    rng = np.random.default_rng(0)
    for _ in range(300):
        k = int(rng.integers(2, 8))
        p = rng.random(k)
        p = p / p.sum()
        q = rng.random(k)
        q = q / q.sum()
        # One-hot Brier: E[||q - e_Y||^2] = ||q - p||^2 + (1 - ||p||^2)
        e_brier = float(np.sum(p * (np.sum(q * q) + 1.0 - 2.0 * q)))
        bayes = 1.0 - float(np.dot(p, p))
        div = float(np.sum((q - p) ** 2))
        assert e_brier == pytest.approx(div + bayes, abs=1e-9)
        # RPS via cumulative binary scores.
        fq = np.cumsum(q)[:-1]
        fp = np.cumsum(p)[:-1]
        div_r = float(np.sum((fq - fp) ** 2))
        bayes_r = float(np.sum(fp * (1.0 - fp)))
        # Expectation by enumerating classes.
        exp_r = 0.0
        for y in range(k):
            ind = (np.arange(k - 1) >= y).astype(float)
            exp_r += p[y] * float(np.sum((fq - ind) ** 2))
        assert exp_r == pytest.approx(div_r + bayes_r, abs=1e-9)


def test_refinement_adds_a_nonnegative_term():
    rng = np.random.default_rng(1)
    for _ in range(100):
        k = 6
        q = rng.random(k)
        q = q / q.sum()
        y = int(rng.integers(0, k))
        fq = np.cumsum(q)[:-1]
        ind = (np.arange(k - 1) >= y).astype(float)
        full = float(np.sum((fq - ind) ** 2))
        # Drop the middle threshold.
        keep = [0, 1, 3, 4]
        coarse = float(np.sum((fq[keep] - ind[keep]) ** 2))
        assert coarse <= full + 1e-12


def test_g_coefficients_are_the_lean_rationals():
    cell = {
        (0, 0): Fraction(3, 2),
        (0, 1): Fraction(1, 1),
        (1, 0): Fraction(3, 4),
        (1, 1): Fraction(1, 4),
    }
    item = [ (cell[i, 0] + cell[i, 1]) / 2 for i in (0, 1) ]
    rater = [ (cell[0, j] + cell[1, j]) / 2 for j in (0, 1) ]
    grand = (item[0] + item[1]) / 2
    sigma_c = sum((m - grand) ** 2 for m in item) / 2
    sigma_r = sum((m - grand) ** 2 for m in rater) / 2
    sigma_e = Fraction(5, 4)
    assert sigma_c == Fraction(9, 64)
    assert sigma_r == Fraction(1, 16)
    e_rho = sigma_c / (sigma_c + sigma_e)
    phi = sigma_c / (sigma_c + sigma_r + sigma_e)
    assert e_rho == Fraction(9, 89)
    assert phi == Fraction(3, 31)
    phi4 = sigma_c / (sigma_c + (sigma_r + sigma_e) / 4)
    assert phi4 == Fraction(3, 10)
    assert phi < phi4


def test_dstudy_phi_ignores_n_and_rises_in_k():
    from gstudy import dstudy_curve
    a = dstudy_curve(9 / 64, 1 / 16, 5 / 4, 4)
    b = dstudy_curve(9 / 64, 1 / 16, 5 / 4, 8)
    assert a == b[:4]
    assert a[0]["Phi"] < a[-1]["Phi"]
    assert a[0]["Phi"] == pytest.approx(float(Fraction(3, 31)), abs=1e-12)
    assert a[3]["Phi"] == pytest.approx(0.3, abs=1e-12)
