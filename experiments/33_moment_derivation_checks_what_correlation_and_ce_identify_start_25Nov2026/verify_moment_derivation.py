#!/usr/bin/env python3
"""Numerical checks of the moment derivation in Appendix B of the 11-25-2026 technical memo.

Every identity or bound that the appendix derives is checked on seeded random instances with exact finite sums:
no sampled human ratings, no fitted models, no model calls.  Ratings take the values 0..K-1 with K = 6.
The script prints each check with its worst discrepancy (or smallest slack) and writes the same text to
results/verify_moment_derivation_output.txt and the numbers to results/verify_moment_derivation.json.

Run (any directory; needs only numpy):  python3 -I verify_moment_derivation.py
"""
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
K = 6
R = np.arange(K, dtype=float)
rng = np.random.default_rng(20261125)
LINES, RES = [], {}


def say(s):
    LINES.append(s)
    print(s)


def simplex(n, a=0.7):
    """n strictly positive distributions on K categories (Dirichlet draws with a floor of 1e-9)."""
    p = np.clip(rng.dirichlet(np.full(K, a), size=n), 1e-9, None)
    return p / p.sum(1, keepdims=True)


def softmax(u):
    z = np.asarray(u, float) - np.max(u)
    e = np.exp(z)
    return e / e.sum()


def fd(f, x, h=1e-6):
    """Central finite-difference gradient of a scalar function of a vector."""
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


# ------------------------------------------------------------------ B.2 Pearson sees one conditional moment
def check_tower(trials=200, n=7):
    worst_var, worst_id, worst_ceiling, slack, etas, within_share = 0.0, 0.0, 0.0, np.inf, [], []
    for _ in range(trials):
        P = simplex(n)                                        # item i has rating distribution P[i]; items equally likely
        ER = (P * R).sum() / n
        ER2 = (P * R ** 2).sum() / n
        VarR = ER2 - ER ** 2                                  # total variance of the individual rating R
        mu = P @ R
        Varmu = mu.var()
        within = ((P @ R ** 2) - mu ** 2).mean()              # E[Var(R | X)]
        worst_var = max(worst_var, abs(Varmu + within - VarR))
        m = rng.normal(size=n) * 2 + rng.normal()             # an arbitrary vector of forecast means
        EmR = (m[:, None] * R[None, :] * P).sum() / n         # E[m(X) R] from the joint distribution of (X, R)
        direct = (EmR - m.mean() * ER) / (m.std() * math.sqrt(VarR))
        via = np.corrcoef(m, mu)[0, 1] * math.sqrt(Varmu / VarR)
        worst_id = max(worst_id, abs(direct - via))
        a = 3.0 + 0.7 * mu                                    # positive affine map of the true means
        EaR = (a[:, None] * R[None, :] * P).sum() / n
        ceil = (EaR - a.mean() * ER) / (a.std() * math.sqrt(VarR))
        worst_ceiling = max(worst_ceiling, abs(ceil - math.sqrt(Varmu / VarR)))
        best_random = max(
            ((((mm[:, None] * R[None, :] * P).sum() / n) - mm.mean() * ER) / (mm.std() * math.sqrt(VarR)))
            for mm in rng.normal(size=(300, n)))
        slack = min(slack, math.sqrt(Varmu / VarR) - best_random)
        etas.append(math.sqrt(Varmu / VarR))
        within_share.append(within / VarR)
    RES["tower"] = dict(trials=trials, n_items=n, worst_total_variance_error=worst_var, worst_decomposition_error=worst_id,
                        worst_ceiling_error=worst_ceiling, min_slack_ceiling_minus_random_forecast=slack,
                        mean_eta=float(np.mean(etas)), min_eta=float(np.min(etas)), max_eta=float(np.max(etas)))
    say(f"B.2 law of total variance Var(R) = Var(mu) + E[Var(R|X)]: worst error {worst_var:.2e} over {trials} random instances (n = {n} items, K = {K})")
    say(f"B.2 corr(m(X), R) = corr(m(X), mu(X)) * sqrt(Var mu / Var R): worst error {worst_id:.2e}")
    say(f"B.2 forecast means a + b*mu with b > 0 reach corr = sqrt(Var mu / Var R): worst error {worst_ceiling:.2e}")
    say(f"B.2 no random forecast beat that ceiling (smallest ceiling minus best of 300 random forecasts: {slack:.4f} >= 0); "
        f"the ceiling ranged {min(etas):.3f} to {max(etas):.3f} (mean {np.mean(etas):.3f}) over the instances, always below 1")


# ------------------------------------------------------------------ B.3 Cross-entropy controls every bounded moment
def check_bounds(trials=20000, pop_trials=5000, n_pop=9):
    slack_pinsker = slack_f = slack_var = np.inf
    for _ in range(trials):
        p, q = simplex(2)
        tv = 0.5 * np.abs(p - q).sum()
        kl = float((p * np.log(p / q)).sum())
        slack_pinsker = min(slack_pinsker, math.sqrt(kl / 2) - tv)
        f = rng.uniform(-1, 3, size=K) * rng.choice([0.5, 1.0, 4.0])
        slack_f = min(slack_f, (f.max() - f.min()) * tv - abs((p - q) @ f))
        slack_var = min(slack_var, (K - 1) ** 2 * tv - abs((p @ R ** 2 - (p @ R) ** 2) - (q @ R ** 2 - (q @ R) ** 2)))
    worst_ratio = 0.0
    worst_ratio_var = 0.0
    slack_pop = np.inf
    for _ in range(pop_trials):
        P, Q = simplex(n_pop), simplex(n_pop)
        excess_ce = float((P * np.log(P / Q)).sum(1).mean())         # CE(q) - CE(p) = E_X KL(p_X || q_X)
        f = rng.uniform(0, 1, size=K) * rng.choice([1.0, 3.0])
        width = f.max() - f.min()
        lhs = float((((Q - P) @ f) ** 2).mean())
        rhs = width ** 2 / 2 * excess_ce
        slack_pop = min(slack_pop, rhs - lhs)
        worst_ratio = max(worst_ratio, lhs / rhs)
        var_P, var_Q = P @ R ** 2 - (P @ R) ** 2, Q @ R ** 2 - (Q @ R) ** 2        # row-wise variances
        lhs_v = float(((var_Q - var_P) ** 2).mean())
        worst_ratio_var = max(worst_ratio_var, lhs_v / ((K - 1) ** 4 / 2 * excess_ce))
    RES["bounds"] = dict(trials=trials, pop_trials=pop_trials, min_slack_pinsker=slack_pinsker, min_slack_bounded_function=slack_f,
                         min_slack_variance=slack_var, min_slack_population=slack_pop, largest_ratio_population=worst_ratio,
                         largest_ratio_variance=worst_ratio_var)
    say(f"B.3 Pinsker TV <= sqrt(KL/2): smallest slack {slack_pinsker:.4f} >= 0 over {trials} random pairs of distributions")
    say(f"B.3 |E_p f - E_q f| <= width(f) * TV: smallest slack {slack_f:.4f} >= 0")
    say(f"B.3 |Var_p - Var_q| <= (K-1)^2 * TV: smallest slack {slack_var:.4f} >= 0")
    say(f"B.3 population bound E_X[(m_f(q,X) - m_f(p,X))^2] <= width^2/2 * (CE(q) - CE(p)): smallest slack {slack_pop:.4f} >= 0; "
        f"largest observed ratio of the two sides {worst_ratio:.3f} (n = {n_pop} items, {pop_trials} random populations)")
    say(f"B.3 variance version E_X[(Var_p - Var_q)^2] <= (K-1)^4/2 * (CE(q) - CE(p)): largest observed ratio {worst_ratio_var:.3f} <= 1")


def int_det(M):
    """Exact integer determinant by the Bareiss algorithm."""
    M = [row[:] for row in M]
    n, sign, prev = len(M), 1, 1
    for k in range(n - 1):
        if M[k][k] == 0:
            sw = next((i for i in range(k + 1, n) if M[i][k] != 0), None)
            if sw is None:
                return 0
            M[k], M[sw] = M[sw], M[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]


def check_vandermonde():
    Vint = [[1 if j == 0 else i ** j for j in range(K)] for i in range(K)]     # rows: ratings r = i; columns: powers r^j
    det = int_det(Vint)
    formula = math.prod(math.factorial(k) for k in range(1, K))                 # 1! 2! ... (K-1)!
    V = np.array(Vint, float)
    cond = float(np.linalg.cond(V))
    errs = []
    for _ in range(200):
        p = simplex(1)[0]
        mom = V.T @ p                                                           # raw moments j = 0..K-1
        errs.append(float(np.abs(np.linalg.solve(V.T, mom) - p).max()))
    nullity = {}
    for label, cols in (("mean only (j = 0, 1)", [0, 1]), ("moments j = 0..K-2", list(range(K - 1))), ("moments j = 0..K-1", list(range(K)))):
        nullity[label] = K - int(np.linalg.matrix_rank(V[:, cols].T))
    RES["vandermonde"] = dict(K=K, determinant=det, product_formula=formula, condition_number_2norm=cond,
                              worst_recovery_error=max(errs), nullity=nullity)
    say(f"B.4 Vandermonde matrix of the ratings 0..{K - 1}: exact determinant {det} = 1!2!...{K - 1}! = {formula} (so it is invertible)")
    say(f"B.4 2-norm condition number of that matrix for K = {K}: {cond:.3g}; solving for p from its exact moments recovered 200 random p to within {max(errs):.2e}")
    say("B.4 free directions left by matching the moments: " + "; ".join(f"{k}: {v}" for k, v in nullity.items()) + f" (the mean alone leaves K - 2 = {K - 2})")


def check_polynomial_family():
    """Every positive distribution is q_theta ~ exp(theta . T(r)) with T(r) = (x, x^2, ..., x^{K-1}), x = r/(K-1)."""
    x = R / (K - 1)
    A = np.stack([x ** j for j in range(K)], 1)                  # columns 1, x, ..., x^{K-1}
    worst = 0.0
    for _ in range(200):
        p = simplex(1)[0]
        c = np.linalg.solve(A, np.log(p))                        # log p(r) = c_0 + sum_j c_j x^j exactly
        worst = max(worst, float(np.abs(softmax(A[:, 1:] @ c[1:]) - p).max()))
    cond = float(np.linalg.cond(A))
    RES["polynomial_family"] = dict(worst_reconstruction_error=worst, condition_number_2norm=cond)
    say(f"B.5 polynomial features of degree K-1 reach every positive distribution: worst reconstruction error {worst:.2e} over 200 random p; "
        f"2-norm condition number of the feature matrix {cond:.3g}")


# ------------------------------------------------------------------ B.5 gradients
def check_gradients():
    out = {}
    # (a) CE gradient of the exponential family q_theta(r) ~ exp(theta . T(r)) is E_q[T] - E_p[T]
    T = np.stack([R, R ** 2, R ** 3], 1) / np.array([5.0, 25.0, 125.0])
    worst = 0.0
    for _ in range(100):
        p = simplex(1)[0]
        th = rng.normal(size=3) * 0.5
        ce = lambda t: -float((p * np.log(softmax(T @ t))).sum())
        g = T.T @ softmax(T @ th) - T.T @ p
        worst = max(worst, float(np.abs(g - fd(ce, th)).max()))
    out["expfam_ce_gradient"] = worst
    say(f"B.5 exponential family: grad_theta CE = E_q[T] - E_p[T]: worst error against finite differences {worst:.2e} (100 random instances)")
    # (b) the calibration head q ~ p^tau exp(b v + c v^2), v = (r - 2.5)/2.5, for one observed rating y
    v = (R - 2.5) / 2.5
    worst = 0.0
    for _ in range(200):
        pb = simplex(1)[0]
        tau = float(rng.choice([0.5, 1.0, 2.0]))
        bc = rng.normal(size=2) * 0.7
        y = int(rng.integers(K))
        q_of = lambda x: softmax(tau * np.log(pb) + x[0] * v + x[1] * v ** 2)
        loss = lambda x: -float(np.log(q_of(x)[y]))
        q = q_of(bc)
        analytic = np.array([q @ v - v[y], q @ v ** 2 - v[y] ** 2])
        worst = max(worst, float(np.abs(analytic - fd(loss, bc)).max()))
    out["head_gradient"] = worst
    say(f"B.5 head q ~ p^tau exp(b v + c v^2): d(-log q(y))/db = E_q v - v(y) and d/dc = E_q v^2 - v(y)^2: worst error {worst:.2e}")
    # (c) softmax gradients of the per-rating losses
    w_ce = w_mse = w_rps = w_brier = 0.0
    for _ in range(200):
        u = rng.normal(size=K)
        y = int(rng.integers(K))
        q = softmax(u)
        m = float(q @ R)
        e_y = np.eye(K)[y]
        ce = lambda x: -float(np.log(softmax(x)[y]))
        mse = lambda x: float((softmax(x) @ R - y) ** 2)
        def rps(x):
            F = np.cumsum(softmax(x))[:-1]
            return float(((F - (y <= np.arange(K - 1))) ** 2).sum() / (K - 1))
        brier = lambda x: float(((softmax(x) - e_y) ** 2).sum())
        w_brier = max(w_brier, float(np.abs(2 * q * ((q - e_y) - float(q @ (q - e_y))) - fd(brier, u)).max()))
        w_ce = max(w_ce, float(np.abs((q - e_y) - fd(ce, u)).max()))
        w_mse = max(w_mse, float(np.abs(2 * (m - y) * q * (R - m) - fd(mse, u)).max()))
        F = np.cumsum(q)[:-1]
        a = np.array([2 / (K - 1) * sum(F[k] - float(y <= k) for k in range(r, K - 1)) for r in range(K)])
        w_rps = max(w_rps, float(np.abs(q * (a - q @ a) - fd(rps, u)).max()))
    out.update(ce_logit_gradient=w_ce, brier_logit_gradient=w_brier, mse_logit_gradient=w_mse, rps_logit_gradient=w_rps)
    say(f"B.5 softmax logit gradients: CE q - e_y (worst error {w_ce:.2e}); Brier 2 q(r)[(q(r) - e_y(r)) - sum_s q(s)(q(s) - e_y(s))] ({w_brier:.2e}); mean squared error 2(m - y) q(r)(r - m) ({w_mse:.2e}); "
        f"RPS q(r)(a_r - sum_s q(s) a_s) with a_r = 2/(K-1) sum_{{k>=r}} (F_k - 1[y <= k]) ({w_rps:.2e})")
    # (d) Pearson gradient: d rho / d u_i(r) = g_i q_i(r) (r - m_i), g_i = (zeta_i - rho z_i) / (n s_m)
    worst = 0.0
    for _ in range(100):
        n = 6
        U = rng.normal(size=(n, K))
        t = rng.normal(size=n) * 1.5 + 2.5
        def rho(x):
            Q = np.apply_along_axis(softmax, 1, x.reshape(n, K))
            return float(np.corrcoef(Q @ R, t)[0, 1])
        Q = np.apply_along_axis(softmax, 1, U)
        m = Q @ R
        s_m, s_t = m.std(), t.std()
        r_ = rho(U.ravel())
        g = ((t - t.mean()) / s_t - r_ * (m - m.mean()) / s_m) / (n * s_m)
        analytic = (g[:, None] * Q * (R[None, :] - m[:, None])).ravel()
        worst = max(worst, float(np.abs(analytic - fd(rho, U.ravel())).max()))
    out["pearson_gradient"] = worst
    say(f"B.5 Pearson: d rho / d u_i(r) = g_i q_i(r)(r - m_i) with g_i = (zeta_i - rho z_i)/(n s_m): worst error {worst:.2e} (100 random instances of 6 items)")
    RES["gradients"] = out


# ------------------------------------------------------------------ B.6 Pearson as a squared loss
def check_pearson_squared():
    w1 = w2 = 0.0
    for _ in range(200):
        n = 40
        x, y = rng.normal(size=n) * 2 + 1, rng.normal(size=n) * 0.5 - 3
        r = np.corrcoef(x, y)[0, 1]
        zx, zy = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
        w1 = max(w1, abs(((zx - zy) ** 2).mean() - 2 * (1 - r)))
        mse = ((x - y) ** 2).mean()
        w2 = max(w2, abs(mse - ((x.mean() - y.mean()) ** 2 + (x.std() - y.std()) ** 2 + 2 * x.std() * y.std() * (1 - r))))
    RES["pearson_squared"] = dict(worst_zscore_identity=w1, worst_mse_decomposition=w2)
    say(f"B.6 mean of (z_x - z_y)^2 = 2(1 - r): worst error {w1:.2e}; MSE = (mean diff)^2 + (sd diff)^2 + 2 s_x s_y (1 - r): worst error {w2:.2e} (200 random vector pairs)")


# ------------------------------------------------------------------ B.7 an exactly solvable finite-sample example
def check_threshold():
    tL, tH = Fraction(2, 5), Fraction(4, 5)
    rows = []
    for m in (Fraction(-3, 10), Fraction(-1, 10), Fraction(0), Fraction(1, 5), Fraction(3, 10)):
        pL = [(tL - m) / 2, 1 - tL, (tL + m) / 2]                # probabilities of D = -1, 0, +1 (R = 2 + D) at low spread
        pH = [(tH - m) / 2, 1 - tH, (tH + m) / 2]
        assert all(x > 0 for x in pL + pH)
        high = [h > l for h, l in zip(pH, pL)]                  # one rating: maximum likelihood picks the larger likelihood
        eL = sum(l for l, hi in zip(pL, high) if hi)             # truth low, picks high
        eH = sum(h for h, hi in zip(pH, high) if not hi)         # truth high, picks low
        rows.append((m, eL, eH, eL / (1 + eL - eH)))
    assert len({(r[1], r[2]) for r in rows}) == 1
    eL, eH = rows[0][1], rows[0][2]
    pi_star = eL / (1 + eL - eH)
    delta2 = (tH - tL) ** 2
    pear = lambda pi: delta2 * pi                                 # Pearson tie rule always picks low spread
    cee = lambda pi: delta2 * ((1 - pi) * eL + pi * eH)
    assert pear(pi_star) == cee(pi_star)
    assert cee(Fraction(3, 10)) > pear(Fraction(3, 10)) and cee(Fraction(2, 5)) < pear(Fraction(2, 5))
    RES["threshold"] = dict(t_low=str(tL), t_high=str(tH), e_low=str(eL), e_high=str(eH), pi_star=str(pi_star),
                            closed_form_t_low_over_sum=str(tL / (tL + tH)), means_checked=[str(r[0]) for r in rows])
    say(f"B.7 one rating, known mean, spreads t = {tL} and {tH}: the maximum-likelihood spread choice errs with probability e_L = {eL} (truth low) and e_H = {eH} (truth high) "
        f"for each of {len(rows)} means checked; threshold pi* = e_L / (1 + e_L - e_H) = {pi_star} = t_L / (t_L + t_H) = {tL / (tL + tH)}")
    say(f"B.7 decimals of the same quantities: t_L = {float(tL)}, t_H = {float(tH)}, e_L = {float(eL)}, e_H = {float(eH)}, pi* = {float(pi_star):.4f}; "
        f"mean offsets checked m = {', '.join(str(float(r[0])) for r in rows)}; squared variance error of a wrong spread (t_H - t_L)^2 = {float(delta2)}")
    say(f"B.7 at the threshold the two selectors have equal expected squared variance error ({float(pear(pi_star)):.4f}); "
        f"at prior pi = 0.3 the CE selector is worse ({float(cee(Fraction(3, 10))):.4f} vs {float(pear(Fraction(3, 10))):.4f}), at pi = 0.4 it is better ({float(cee(Fraction(2, 5))):.4f} vs {float(pear(Fraction(2, 5))):.4f})")


def main():
    say(f"Moment derivation checks, seed 20261125, K = {K} ratings 0..{K - 1}; every check uses exact finite sums over random instances.")
    check_tower()
    check_bounds()
    check_vandermonde()
    check_polynomial_family()
    check_gradients()
    check_pearson_squared()
    check_threshold()
    (OUT / "verify_moment_derivation_output.txt").write_text("\n".join(LINES) + "\n")
    (OUT / "verify_moment_derivation.json").write_text(json.dumps(RES, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
