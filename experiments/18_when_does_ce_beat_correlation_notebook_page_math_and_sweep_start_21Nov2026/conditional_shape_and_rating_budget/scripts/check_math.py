"""Reproduce the symbolic and numerical examples in MATH_11-21-2026.md."""
from itertools import product

import numpy as np
import sympy as sp


def check_derivative():
    """Change one predicted mean, including its effect on centering and scaling."""
    t, c, u, v = sp.symbols("t c u v", real=True)
    n = sp.symbols("n", positive=True)
    # a and b are centered vector lengths, c the original centered dot product.
    a, b = sp.symbols("a b", positive=True)
    rho_t = (c + t * v) / (b * sp.sqrt(a**2 + 2 * u * t + (1 - 1 / n) * t**2))
    derivative = sp.diff(rho_t, t).subs(t, 0)
    assert sp.simplify(derivative - (v / (a * b) - c * u / (a**3 * b))) == 0

    ratings = np.arange(6)
    logits = np.random.default_rng(2906).normal(size=(7, 6))
    target = np.array([1.0, 2.5, 1.5, 5.0, 0.0, 3.5, 1.0])

    def softmax(values):
        weights = np.exp(values - np.max(values, axis=1, keepdims=True))
        return weights / np.sum(weights, axis=1, keepdims=True)

    def loss(values):
        return 1 - np.corrcoef(softmax(values) @ ratings, target)[0, 1]

    forecast = softmax(logits)
    means = forecast @ ratings
    z = (means - means.mean()) / means.std()
    h = (target - target.mean()) / target.std()
    rho = 1 - loss(logits)
    np.testing.assert_allclose(1 - rho, np.mean((z - h) ** 2) / 2)
    dmean = (rho * z - h) / (len(target) * means.std())
    analytic = dmean[:, None] * forecast * (ratings - means[:, None])
    step = 1e-5
    numerical = np.empty_like(analytic)
    for index in np.ndindex(logits.shape):
        delta = np.zeros_like(logits)
        delta[index] = step
        numerical[index] = (loss(logits + delta) - loss(logits - delta)) / (2 * step)
    max_error = np.max(abs(analytic - numerical))
    assert max_error < 1e-8
    print(f"Symbolic derivative residual = 0. Logit gradient max error ({analytic.size} entries) = {max_error:.3g}.")


def check_example():
    """Compute finite population scores from exact rational probabilities."""
    rat = sp.Rational
    r = sp.Matrix(range(6))
    uniform = sp.ones(6, 1) / 6
    transfer = sp.Matrix([-1, 0, 0, 0, 0, 1])
    shape = sp.Matrix([0, 1, -2, 1, 0, 0])
    p_population, q_population = [], []
    for x in (-1, 0, 1):
        base = uniform + rat(2 * x**2 - 1, 10) * transfer
        indicator = int(x == 1)
        p = base + rat(indicator, 24) * shape
        q = base - rat(indicator, 24) * shape
        assert min(p) > 0 and min(q) > 0 and sum(p) == sum(q) == 1
        assert r.dot(p) == r.dot(q) == 2 + x**2
        p_population.append(p)
        q_population.append(q)

    # RPS and CE evaluated directly from individual outcomes under the true p.
    def outcome_rps(forecast, y):
        return sum((sum(forecast[:k + 1]) - int(y <= k))**2 for k in range(5)) / 5

    p, q = p_population[-1], q_population[-1]
    excess_rps = sum(p[y] * (outcome_rps(q, y) - outcome_rps(p, y)) for y in range(6))
    excess_ce = sp.expand_log(sum(py * sp.log(py / qy) for py, qy in zip(p, q)), force=True)
    assert excess_rps == rat(1, 360)
    assert sum(abs(py - qy) for py, qy in zip(p, q)) / 2 == rat(1, 6)
    assert sp.simplify(excess_ce - (10 * sp.log(rat(5, 3)) - 2 * sp.log(3)) / 24) == 0
    print(f"Pearson versus true means across x = 1. At x=1: RPS excess = {excess_rps}; "
          f"CE excess = {float(excess_ce):.9f} nats; TVD = 1/6.")
    p = np.array([list(row) for row in p_population], dtype=float)
    q = np.array([list(row) for row in q_population], dtype=float)
    alpha = np.array([[0.2], [0.5], [0.8]])
    scaled = 1 / 6 + alpha * (p - 1 / 6)
    for p_row, scaled_row in zip(p, scaled):
        np.testing.assert_allclose(np.corrcoef(p_row, scaled_row)[0, 1], 1)
    assert np.corrcoef(p.ravel(), scaled.ravel())[0, 1] < 1
    common_scale = 1 / 6 + 0.5 * (p - 1 / 6)
    np.testing.assert_allclose(np.corrcoef(p.ravel(), common_scale.ravel())[0, 1], 1)
    return p, q


def check_budget_bounds(p, q):
    """Enumerate all observations for two artifacts with two ratings each."""
    n = k = 2
    one_artifact = []
    for x in range(len(p)):
        for ratings in product(range(6), repeat=k):
            pair = [np.prod(m[x, list(ratings)]) / len(m) for m in (p, q)]
            one_artifact.append(pair)
    p_obs, q_obs = np.array(one_artifact).T
    p_data = np.outer(p_obs, p_obs)
    q_data = np.outer(q_obs, q_obs)
    np.testing.assert_allclose([p_data.sum(), q_data.sum()], 1)
    error = np.minimum(p_data, q_data).sum() / 2
    affinity = np.sqrt(p * q).sum(axis=1)
    upper = np.mean(affinity**k)**n / 2
    lower = (1 - 1 / 3)**n / 2
    assert lower < error < upper
    print(f"n=k=2, equal priors: lower bound {lower:.9f}, enumerated Bayes error "
          f"{error:.9f}, upper bound {upper:.9f} ({p_data.size} outcomes).")


if __name__ == "__main__":
    check_derivative()
    check_budget_bounds(*check_example())
