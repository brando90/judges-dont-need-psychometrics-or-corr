#!/usr/bin/env python3
"""TLDR: Shared helpers for experiment 04.

Exact (Fraction) and float helpers for six-category rating forecasts on {0,...,5}:
moments, normalized ranked probability score (RPS), cross-entropy (CE), total variation
distance (TVD), Brier score; generalizability-theory (G-theory) variance components for
one-facet (objects x raters) and two-facet (objects x a x b) crossed random designs,
balanced analysis-of-variance (ANOVA) estimators, constrained restricted maximum
likelihood (REML) for balanced data, and decision-study (D-study) coefficients.

Every function here is checked against an independent implementation in test_metrics.py.
"""
from __future__ import annotations

import math
from fractions import Fraction as Fr

import numpy as np

K = 6                    # rating categories 0..5
Y = tuple(range(K))
UNIFORM = [Fr(1, K)] * K
COMPS = ('u', 'a', 'b', 'ua', 'ub', 'ab', 'e')   # two-facet variance components, fixed order


# ---------------------------------------------------------------- exact pmf helpers
def binom_pmf(theta, n: int = 5):
    """Binomial(n, theta) probabilities; exact when theta is a Fraction."""
    return [math.comb(n, y) * theta ** y * (1 - theta) ** (n - y) for y in range(n + 1)]


def mix(p, q, w):
    """(1 - w) * p + w * q, elementwise."""
    return [(1 - w) * a + w * b for a, b in zip(p, q)]


def is_pmf(p, positive: bool = False) -> bool:
    ok = len(p) == K and sum(p) == 1 and all(x >= 0 for x in p)
    return ok and (all(x > 0 for x in p) if positive else True)


def mean(p):
    return sum(y * p[y] for y in Y)


def var(p):
    m = mean(p)
    return sum((y - m) ** 2 * p[y] for y in Y)


def raw_moment(p, r: int):
    return sum(y ** r * p[y] for y in Y)


def cdf(p):
    """Cumulative probabilities Q(k) for k = 0..4 (Q(5) = 1 is omitted)."""
    out, c = [], 0
    for y in Y[:-1]:
        c += p[y]
        out.append(c)
    return out


# ---------------------------------------------------------------- expected proper losses
def rps_excess(q, p):
    """E_{y~p} RPS(q, y) - E_{y~p} RPS(p, y) = (1/5) sum_k (Q(k) - P(k))^2."""
    return sum((a - b) ** 2 for a, b in zip(cdf(q), cdf(p))) / 5


def rps_irreducible(p):
    """E_{y~p} RPS(p, y) = (1/5) sum_k P(k) (1 - P(k))."""
    return sum(c * (1 - c) for c in cdf(p)) / 5


def expected_rps(q, p):
    return rps_excess(q, p) + rps_irreducible(p)


def brier_excess(q, p):
    return sum((a - b) ** 2 for a, b in zip(q, p))


def tvd(q, p):
    return sum(abs(a - b) for a, b in zip(q, p)) / 2


def expected_ce(q, p) -> float:
    """E_{y~p} [-ln q(y)] in nats; infinity when p puts mass where q has none."""
    total = 0.0
    for a, b in zip(q, p):
        if b == 0:
            continue
        if a == 0:
            return math.inf
        total -= float(b) * math.log(float(a))
    return total


def kl(p, q) -> float:
    """KL(p || q) in nats = expected CE excess of forecast q under truth p."""
    total = 0.0
    for a, b in zip(p, q):
        if a == 0:
            continue
        if b == 0:
            return math.inf
        total += float(a) * (math.log(float(a)) - math.log(float(b)))
    return total


# ---------------------------------------------------------------- individual-rating losses
def rps_individual(q, y: int):
    """Normalized individual-rating RPS: (1/5) sum_{k=0}^{4} (Q(k) - 1{y <= k})^2, range 0..1."""
    return sum((c - (1 if y <= k else 0)) ** 2 for k, c in enumerate(cdf(q))) / 5


def ce_individual(q, y: int) -> float:
    return math.inf if q[y] == 0 else -math.log(float(q[y]))


def smooth(q, epsilon: float):
    """Experiment 02/06 floor rule: (1 - 6 eps) q + eps, 0 < eps < 1/6."""
    assert 0 < epsilon < 1 / 6
    return [(1 - 6 * epsilon) * float(x) + epsilon for x in q]


# ---------------------------------------------------------------- coefficients
def ratio(num, den):
    """num / den, or None (undefined) when den == 0. Never reports 0/0 as perfect reliability."""
    return None if den == 0 else num / den


def iid_panel_components(pmfs):
    """Population components of a one-facet objects x raters panel in which each rating is an
    independent draw from the object's pmf (no rater effects). Objects are drawn uniformly from
    the given list, so v_u is the population variance (divisor N) of the conditional means."""
    n = len(pmfs)
    ms = [mean(p) for p in pmfs]
    mbar = sum(ms) / n
    return {'u': sum((m - mbar) ** 2 for m in ms) / n, 'a': 0 * mbar, 'e': sum(var(p) for p in pmfs) / n}


def one_facet_coeffs(v, n: int):
    """Relative (Erho2) and absolute (Phi) coefficients for the mean of n raters."""
    v_delta = v['e'] / n
    v_Delta = (v['a'] + v['e']) / n
    return {'v_delta': v_delta, 'v_Delta': v_Delta,
            'Erho2': ratio(v['u'], v['u'] + v_delta), 'Phi': ratio(v['u'], v['u'] + v_Delta)}


def d_study(v, n_a: int, n_b: int):
    """Two-facet crossed random D study for the average over n_a levels of a and n_b levels of b."""
    v_delta = v['ua'] / n_a + v['ub'] / n_b + v['e'] / (n_a * n_b)
    v_Delta = v_delta + v['a'] / n_a + v['b'] / n_b + v['ab'] / (n_a * n_b)
    return {'v_delta': v_delta, 'v_Delta': v_Delta,
            'Erho2': ratio(v['u'], v['u'] + v_delta), 'Phi': ratio(v['u'], v['u'] + v_Delta)}


# ---------------------------------------------------------------- one-facet ANOVA
def anova2(X):
    """Balanced objects x raters ANOVA (one observation per cell). X: (n_u, n_a)."""
    X = np.asarray(X, dtype=float)
    n_u, n_a = X.shape
    g = X.mean()
    mu, ma = X.mean(1), X.mean(0)
    ss_u = n_a * ((mu - g) ** 2).sum()
    ss_a = n_u * ((ma - g) ** 2).sum()
    ss_e = ((X - mu[:, None] - ma[None, :] + g) ** 2).sum()
    df_u, df_a, df_e = n_u - 1, n_a - 1, (n_u - 1) * (n_a - 1)
    ms_u, ms_a = ss_u / df_u, ss_a / df_a
    ms_e = ss_e / df_e if df_e > 0 else float('nan')
    est = {'u': (ms_u - ms_e) / n_a, 'a': (ms_a - ms_e) / n_u, 'e': ms_e}
    return {'ms_u': ms_u, 'ms_a': ms_a, 'ms_e': ms_e, 'df_u': df_u, 'df_a': df_a, 'df_e': df_e, 'est': est}


def icc_two_way(X):
    """McGraw-Wong two-way single-measure ICC(C,1) and ICC(A,1) from mean squares."""
    r = anova2(X)
    n, k = np.asarray(X).shape
    ms_u, ms_a, ms_e = r['ms_u'], r['ms_a'], r['ms_e']
    c_den = ms_u + (k - 1) * ms_e
    a_den = ms_u + (k - 1) * ms_e + k * (ms_a - ms_e) / n
    return {'ICC_C1': None if c_den == 0 else (ms_u - ms_e) / c_den,
            'ICC_A1': None if a_den == 0 else (ms_u - ms_e) / a_den}


# ---------------------------------------------------------------- two-facet ANOVA / REML
def ems_matrix(n_u: int, n_a: int, n_b: int) -> np.ndarray:
    """Expected-mean-square coefficients; rows = strata, columns = components, both in COMPS order."""
    rows = {
        'u': {'u': n_a * n_b, 'ua': n_b, 'ub': n_a, 'e': 1},
        'a': {'a': n_u * n_b, 'ua': n_b, 'ab': n_u, 'e': 1},
        'b': {'b': n_u * n_a, 'ub': n_a, 'ab': n_u, 'e': 1},
        'ua': {'ua': n_b, 'e': 1},
        'ub': {'ub': n_a, 'e': 1},
        'ab': {'ab': n_u, 'e': 1},
        'e': {'e': 1},
    }
    return np.array([[rows[s].get(c, 0) for c in COMPS] for s in COMPS], dtype=float)


def strata_df(n_u: int, n_a: int, n_b: int) -> np.ndarray:
    du, da, db = n_u - 1, n_a - 1, n_b - 1
    return np.array([du, da, db, du * da, du * db, da * db, du * da * db], dtype=float)


def anova3(X):
    """Balanced crossed objects x a x b sums of squares, mean squares and raw ANOVA estimates.
    Raw estimates can be negative; they are returned unmodified."""
    X = np.asarray(X, dtype=float)
    n_u, n_a, n_b = X.shape
    g = X.mean()
    mu, ma, mb = X.mean((1, 2)), X.mean((0, 2)), X.mean((0, 1))
    mua, mub, mab = X.mean(2), X.mean(1), X.mean(0)
    ss = np.array([
        n_a * n_b * ((mu - g) ** 2).sum(),
        n_u * n_b * ((ma - g) ** 2).sum(),
        n_u * n_a * ((mb - g) ** 2).sum(),
        n_b * ((mua - mu[:, None] - ma[None, :] + g) ** 2).sum(),
        n_a * ((mub - mu[:, None] - mb[None, :] + g) ** 2).sum(),
        n_u * ((mab - ma[:, None] - mb[None, :] + g) ** 2).sum(),
        ((X - mua[:, :, None] - mub[:, None, :] - mab[None, :, :]
          + mu[:, None, None] + ma[None, :, None] + mb[None, None, :] - g) ** 2).sum(),
    ])
    df = strata_df(n_u, n_a, n_b)
    ms = ss / df
    est = np.linalg.solve(ems_matrix(n_u, n_a, n_b), ms)
    return {'ss': ss, 'df': df, 'ms': ms, 'est': dict(zip(COMPS, est.tolist()))}


def reml3_negloglik(v, ss, df, C):
    """Stratum form of the balanced-design REML negative log-likelihood (additive constant dropped)
    and its gradient. lambda_s = expected mean square of stratum s."""
    lam = C @ v
    f = 0.5 * float((df * np.log(lam) + ss / lam).sum())
    grad = 0.5 * C.T @ (df / lam - ss / lam ** 2)
    return f, grad


def reml3(ss, df, n_u, n_a, n_b, start=None):
    """Constrained REML (all components >= 0; residual >= 1e-12) for a balanced crossed design.
    Prespecified: L-BFGS-B from the ANOVA estimates clipped to >= 1e-3 * MS_e."""
    from scipy.optimize import minimize
    C = ems_matrix(n_u, n_a, n_b)
    ss, df = np.asarray(ss, float), np.asarray(df, float)
    if start is None:
        raw = np.linalg.solve(C, ss / df)
        start = np.maximum(raw, 1e-3 * ss[-1] / df[-1])
    bounds = [(0.0, None)] * 6 + [(1e-12, None)]
    res = minimize(reml3_negloglik, np.asarray(start, float), args=(ss, df, C), jac=True,
                   method='L-BFGS-B', bounds=bounds,
                   options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 20000, 'maxfun': 40000})
    return {'est': dict(zip(COMPS, res.x.tolist())), 'success': bool(res.success),
            'nit': int(res.nit), 'message': str(res.message), 'negloglik': float(res.fun)}


def dense_reml_loglik(X, v) -> float:
    """Direct REML log-likelihood from the full covariance matrix (verification only; tiny designs).
    Fixed effect: intercept. Additive constant -(n-1)/2 log(2 pi) dropped."""
    X = np.asarray(X, dtype=float)
    n_u, n_a, n_b = X.shape
    y = X.reshape(-1)
    idx = np.array([(i, j, k) for i in range(n_u) for j in range(n_a) for k in range(n_b)])

    def same(*cols):
        m = np.ones((len(idx), len(idx)), dtype=bool)
        for c in cols:
            m &= idx[:, None, c] == idx[None, :, c]
        return m.astype(float)

    V = (v['u'] * same(0) + v['a'] * same(1) + v['b'] * same(2) + v['ua'] * same(0, 1)
         + v['ub'] * same(0, 2) + v['ab'] * same(1, 2) + v['e'] * np.eye(len(y)))
    one = np.ones(len(y))
    Vi = np.linalg.inv(V)
    s = one @ Vi @ one
    P = Vi - np.outer(Vi @ one, one @ Vi) / s
    _, logdet = np.linalg.slogdet(V)
    return -0.5 * (logdet + math.log(s) + y @ P @ y)


# ---------------------------------------------------------------- serialization
def fr_to_str(x) -> str:
    x = Fr(x)
    return f'{x.numerator}/{x.denominator}'


def str_to_fr(s: str) -> Fr:
    return Fr(s)


def as_float(x):
    return None if x is None else float(x)
