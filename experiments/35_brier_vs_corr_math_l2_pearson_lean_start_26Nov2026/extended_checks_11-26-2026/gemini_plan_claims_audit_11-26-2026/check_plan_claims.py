#!/usr/bin/env python3
"""Seeded checks of the mathematical claims in the "5-dataset real judging suite" plan (relayed 11-26-2026).

Data: synthetic finite laws only (random pmfs with small denominators, a few hand-built instances).
Nothing here uses human ratings.  This is a check of the *mathematics* the plan relies on, not an
empirical result.

Arithmetic: exact `fractions.Fraction` wherever the quantity is rational (enumerating every outcome of
the finite joint law, never assuming the formula being checked).  Where a quantity is irrational
(logarithms, square roots at large N) a float comparison with the named tolerance TOL is used and the
line says so.  Every statement that the plan or the earlier notes claim to be FALSE (or that I found
false) is tested as a failing case and printed as `PASS  ... [claimed-false case]`.

Output: one PASS or FAIL line per check, a SUMMARY line, exit status 1 if any check fails.
Deterministic: fixed seeds, no clock, no hash-order dependence.  Run: python3 -I check_plan_claims.py
"""
from __future__ import annotations

import math
import random
import sys
from fractions import Fraction as F
from itertools import product

SEED = 20261126
TOL = 1e-12  # float tolerance, used only for irrational quantities

RESULTS: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((label, bool(ok)))
    print(("PASS" if ok else "FAIL") + "  " + label + (("  -- " + detail) if detail else ""))


def header(text: str) -> None:
    print()
    print("== " + text)


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def rand_pmf(rng: random.Random, K: int, den: int, zeros: bool = False) -> list[F]:
    """Random pmf over K outcomes with denominator `den` (all entries > 0 unless zeros=True)."""
    while True:
        cuts = sorted(rng.randint(0, den) for _ in range(K - 1))
        parts = [b - a for a, b in zip([0] + cuts, cuts + [den])]
        if zeros or all(x > 0 for x in parts):
            return [F(x, den) for x in parts]


def prod_fr(xs) -> F:
    out = F(1)
    for x in xs:
        out *= x
    return out


def sq_norm(p) -> F:
    return sum((x * x for x in p), F(0))


def brier(q, y) -> F:
    return sum(((qk - (1 if k == y else 0)) ** 2 for k, qk in enumerate(q)), F(0))


def cdf(q) -> list[F]:
    out, s = [], F(0)
    for qk in q:
        s += qk
        out.append(s)
    return out


def rps(q, y) -> F:
    """Ranked probability score ||F_q - F_{e_y}||^2 (sum over all thresholds t; the last term is 0)."""
    Fq = cdf(q)
    return sum(((Fq[t] - (1 if y <= t else 0)) ** 2 for t in range(len(q))), F(0))


def compositions(n: int, K: int):
    if K == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in compositions(n - i, K - 1):
            yield (i,) + rest


def multinom(counts) -> int:
    r = math.factorial(sum(counts))
    for c in counts:
        r //= math.factorial(c)
    return r


def joint_prompt_law(p, k: int, w: F) -> dict:
    """Joint law of the k ratings of one prompt: w * (k iid draws from p) + (1-w) * (k copies of one
    draw from p).  Every single rating has marginal p, whatever w is; w < 1 makes ratings dependent."""
    law = {}
    for ys in product(range(len(p)), repeat=k):
        pr = w * prod_fr(p[y] for y in ys)
        if len(set(ys)) == 1:
            pr += (1 - w) * p[ys[0]]
        if pr:
            law[ys] = pr
    return law


def enumerate_prompts(laws):
    for combo in product(*[list(l.items()) for l in laws]):
        pr = F(1)
        for (_, pc) in combo:
            pr *= pc
        yield [s for (s, _) in combo], pr


def rhat(states, qs, loss) -> object:
    """(1/m) sum_c (1/k_c) sum_j L(q_c, y_cj)."""
    m = len(states)
    tot = 0
    for ys, q in zip(states, qs):
        tot = tot + sum(loss(q, y) for y in ys) / len(ys)
    return tot / m


# ----------------------------------------------------------------------------------------------
# N1. unbiased per-sample proper risk (plan, "statistical details" 1 and rule 1)
# ----------------------------------------------------------------------------------------------
def claim_N1() -> None:
    header("N1  per-sample test risk  R_hat = (1/m) sum_c (1/k_c) sum_j L(q(c), y_cj)  is unbiased")
    rng = random.Random(SEED + 1)
    n_inst = 0
    n_k1 = 0
    all_ok = {"brier": True, "rps": True, "table": True}
    ok_dep = True
    worst_ce = 0.0
    for _ in range(240):
        m = rng.randint(1, 3)
        K = rng.randint(2, 3)
        ks = [rng.randint(1, 3) for _ in range(m)]
        while K ** sum(ks) > 800:  # keep the enumeration small
            ks = [max(1, k - 1) for k in ks]
        ps = [rand_pmf(rng, K, rng.choice([4, 6, 8, 12]), zeros=True) for _ in range(m)]
        qs = [rand_pmf(rng, K, rng.choice([5, 7, 10]), zeros=False) for _ in range(m)]
        ws = [rng.choice([F(1), F(3, 4), F(1, 2), F(1, 4), F(0)]) for _ in range(m)]
        W = [[rng.randint(-3, 3) for _ in range(K)] for _ in range(K)]  # arbitrary (non-proper) loss
        table = lambda q, y, W=W: sum((W[y][k] * q[k] for k in range(len(q))), F(0))
        laws = [joint_prompt_law(p, k, w) for p, k, w in zip(ps, ks, ws)]
        n_inst += 1
        n_k1 += any(k == 1 for k in ks)
        for name, loss in (("brier", brier), ("rps", rps), ("table", table)):
            truth = sum((sum((p[y] * loss(q, y) for y in range(K)), F(0)) for p, q in zip(ps, qs)), F(0)) / m
            exp = sum((pr * rhat(st, qs, loss) for st, pr in enumerate_prompts(laws)), F(0))
            all_ok[name] &= (exp == truth)
        # dependent ratings change nothing for the mean: also compare w=1 (iid) with the same marginals
        laws_iid = [joint_prompt_law(p, k, F(1)) for p, k in zip(ps, ks)]
        e1 = sum((pr * rhat(st, qs, brier) for st, pr in enumerate_prompts(laws)), F(0))
        e2 = sum((pr * rhat(st, qs, brier) for st, pr in enumerate_prompts(laws_iid)), F(0))
        ok_dep &= (e1 == e2)
        # cross-entropy in floats (q strictly positive, p may have zeros)
        qf = [[float(x) for x in q] for q in qs]
        ce = lambda q, y: -math.log(q[y])
        truth_f = sum(sum(float(p[y]) * ce(q, y) for y in range(K)) for p, q in zip(ps, qf)) / m
        exp_f = sum(float(pr) * rhat(st, qf, ce) for st, pr in enumerate_prompts(laws))
        worst_ce = max(worst_ce, abs(exp_f - truth_f))
    ok_ce = worst_ce < TOL * 100
    check("N1a  E[R_hat] = true risk, Brier, exactly (k_c in 1..3, per-prompt laws, mixed iid/dependent ratings)", all_ok["brier"], f"{n_inst} instances, {n_k1} with some k_c = 1")
    check("N1b  E[R_hat] = true risk, RPS, exactly", all_ok["rps"], "same instances")
    check("N1c  E[R_hat] = true risk for an ARBITRARY (non-proper) loss table, exactly (properness is not needed for unbiasedness)", all_ok["table"], "same instances")
    check("N1d  E[R_hat] identical for iid and dependent ratings with the same marginals (independence is not needed)", ok_dep)
    check(f"N1e  E[R_hat] = true risk, cross-entropy (float, tolerance {TOL * 100:g})", ok_ce, f"max abs error {worst_ce:.0e}")

    # variance DOES depend on dependence: one prompt, two ratings, p = (1/2,1/2), q = (3/4,1/4)
    p, q, k = [F(1, 2), F(1, 2)], [F(3, 4), F(1, 4)], 2
    out = {}
    for w in (F(1), F(0)):
        law = joint_prompt_law(p, k, w)
        e = sum((pr * rhat([s], [q], brier) for s, pr in law.items()), F(0))
        e2 = sum((pr * rhat([s], [q], brier) ** 2 for s, pr in law.items()), F(0))
        out[w] = (e, e2 - e * e)
    check("N1f  variance of R_hat depends on the dependence between ratings even though the mean does not (iid var < identical-copies var)", out[F(1)][0] == out[F(0)][0] and out[F(1)][1] < out[F(0)][1], f"mean {out[F(1)][0]} both; var iid {out[F(1)][1]} < var copies {out[F(0)][1]}")

    # claimed-false: unbiasedness when the NUMBER of ratings depends on the ratings (HelpSteer2-style adaptive stopping)
    theta = F(3, 10)
    pr = lambda y: theta if y == 1 else 1 - theta
    exp = F(0)  # enumerate the stopping rule directly
    for y1, y2 in product((0, 1), repeat=2):
        if y1 == y2:
            exp += pr(y1) * pr(y2) * F(y1 + y2, 2)
        else:
            for y3, y4 in product((0, 1), repeat=2):
                exp += pr(y1) * pr(y2) * pr(y3) * pr(y4) * F(y1 + y2 + y3 + y4, 4)
    check("N1g  [claimed-false case] two raters, two more only if they disagree: E[item mean] = 129/500 != theta = 3/10 (adaptive k_c biases the estimator)", exp == F(129, 500) and exp != theta, f"E = {exp} = {float(exp):.3f} vs theta = {float(theta):.3f}")

    # claimed-false, generic loss: ask a second rater only if the first said 1
    rng = random.Random(SEED + 11)
    ok_all = True
    for _ in range(60):
        th = F(rng.randint(1, 9), 10)
        q = rand_pmf(rng, 2, 10)
        L0, L1 = brier(q, 0), brier(q, 1)
        e = (1 - th) * L0 + th * (th * L1 + (1 - th) * (L1 + L0) / 2)
        truth = (1 - th) * L0 + th * L1
        ok_all &= (e - truth == th * (1 - th) * (L0 - L1) / 2) and (e != truth or L0 == L1)
    check("N1h  [claimed-false case] 'second rater only if the first rated 1' gives bias theta(1-theta)(L0-L1)/2, nonzero whenever L0 != L1", ok_all, "60 random (theta, q)")

    # k_c random but independent of the ratings: still unbiased
    rng = random.Random(SEED + 12)
    ok_all = True
    for _ in range(40):
        K = 3
        p = rand_pmf(rng, K, 8, zeros=True)
        q = rand_pmf(rng, K, 7)
        truth = sum((p[y] * brier(q, y) for y in range(K)), F(0))
        pk = {1: F(1, 2), 2: F(1, 3), 3: F(1, 6)}
        e = F(0)
        for k, pkv in pk.items():
            law = joint_prompt_law(p, k, F(1))
            e += pkv * sum((pr * rhat([s], [q], brier) for s, pr in law.items()), F(0))
        ok_all &= (e == truth)
    check("N1i  k_c drawn at random INDEPENDENTLY of the ratings keeps E[R_hat] = true risk", ok_all, "40 random (p, q)")


# ----------------------------------------------------------------------------------------------
# N2. plug-in Brier divergence bias and the exact ERM rate
# ----------------------------------------------------------------------------------------------
def claim_N2() -> None:
    header("N2  plug-in D_B(p_hat || q) bias = (1/k)(1 - ||p||^2);  exact ERM rate E||p_hat_N - p||^2 = (1/N)(1 - ||p||^2)")
    rng = random.Random(SEED + 2)
    ok_bias = ok_erm = ok_pos = ok_deg = True
    n = 0
    for _ in range(70):
        K = rng.randint(2, 4)
        k = rng.randint(1, 6)
        p = rand_pmf(rng, K, rng.choice([4, 6, 9]), zeros=True)
        q = rand_pmf(rng, K, rng.choice([5, 8]), zeros=True)
        ephat_q = F(0)
        ephat_p = F(0)
        for cnt in compositions(k, K):
            pr = multinom(cnt) * prod_fr(p[a] ** cnt[a] for a in range(K))
            phat = [F(c, k) for c in cnt]
            ephat_q += pr * sum(((phat[a] - q[a]) ** 2 for a in range(K)), F(0))
            ephat_p += pr * sum(((phat[a] - p[a]) ** 2 for a in range(K)), F(0))
        bias = ephat_q - sum(((p[a] - q[a]) ** 2 for a in range(K)), F(0))
        target = (1 - sq_norm(p)) / k
        ok_bias &= (bias == target)
        ok_erm &= (ephat_p == target)
        if sq_norm(p) < 1:
            ok_pos &= (bias > 0)
        else:
            ok_deg &= (bias == 0)
        n += 1
    check("N2a  E||p_hat_k - q||^2 = ||p - q||^2 + (1-||p||^2)/k exactly (multinomial enumeration)", ok_bias, f"{n} random (K, k, p, q), K<=4, k<=6")
    check("N2b  E||p_hat_N - p||^2 = (1 - ||p||^2)/N exactly", ok_erm, "same instances")
    check("N2c  [claimed-false case] plug-in D_B is NOT unbiased: bias > 0 whenever p is not a point mass", ok_pos)
    check("N2d  bias is exactly 0 when p is a point mass (then 1 - ||p||^2 = 0)", ok_deg)

    # k_c varying across prompts: average of the per-prompt terms
    p1, p2 = [F(1, 2), F(1, 2)], [F(1, 4), F(3, 4)]
    q1, q2 = [F(1, 3), F(2, 3)], [F(1, 2), F(1, 2)]
    k1, k2 = 2, 5
    tot = F(0)
    for (p, q, k) in ((p1, q1, k1), (p2, q2, k2)):
        e = F(0)
        for cnt in compositions(k, 2):
            pr = multinom(cnt) * prod_fr(p[a] ** cnt[a] for a in range(2))
            e += pr * sum(((F(cnt[a], k) - q[a]) ** 2 for a in range(2)), F(0))
        tot += e
    tot /= 2
    base = (sum(((p1[a] - q1[a]) ** 2 for a in range(2)), F(0)) + sum(((p2[a] - q2[a]) ** 2 for a in range(2)), F(0))) / 2
    pred = ((1 - sq_norm(p1)) / k1 + (1 - sq_norm(p2)) / k2) / 2
    check("N2e  with k_c varying across prompts the bias is the average of (1-||p_c||^2)/k_c", tot - base == pred, f"bias {tot - base}")

    # large N, binary, float: sum the binomial pmf directly (independent of the formula)
    p = 0.3
    worst = 0.0
    for N in (10, 100, 1000, 10000, 100000):
        s = 0.0
        tot_pr = 0.0
        lp, lq = math.log(p), math.log(1 - p)
        for j in range(N + 1):
            lg = math.lgamma(N + 1) - math.lgamma(j + 1) - math.lgamma(N - j + 1) + j * lp + (N - j) * lq
            pr = math.exp(lg)
            tot_pr += pr
            s += pr * 2 * (j / N - p) ** 2  # ||p_hat - p||^2 for K = 2 equals 2 (p_hat_1 - p)^2
        formula = (1 - p * p - (1 - p) ** 2) / N
        worst = max(worst, abs(s / formula - 1), abs(tot_pr - 1))
    check("N2f  ERM rate for N in {10,100,1e3,1e4,1e5} (binary, float, summing the binomial pmf; relative tolerance 1e-9)", worst < 1e-9, f"worst relative error {worst:.0e}")


# ----------------------------------------------------------------------------------------------
# N3. total variance and anti-properness of correlation (plan, Stage 1 item 4)
# ----------------------------------------------------------------------------------------------
def stats(joint):
    """joint: list of (prob, x, y).  Returns exact (EX, EY, VarX, VarY, Cov)."""
    EX = sum((pr * x for pr, x, _ in joint), F(0))
    EY = sum((pr * y for pr, _, y in joint), F(0))
    VX = sum((pr * (x - EX) ** 2 for pr, x, _ in joint), F(0))
    VY = sum((pr * (y - EY) ** 2 for pr, _, y in joint), F(0))
    CV = sum((pr * (x - EX) * (y - EY) for pr, x, y in joint), F(0))
    return EX, EY, VX, VY, CV


def build_model(rng, nC, K, den, zeros):
    pi = rand_pmf(rng, nC, rng.choice([4, 6, 8]))
    v = [F(i) for i in range(K)] if rng.random() < 0.5 else sorted({F(rng.randint(-6, 12), rng.choice([1, 2, 3])) for _ in range(K * 3)})[:K]
    while len(v) < K:
        v.append(v[-1] + 1)
    p = [rand_pmf(rng, K, den, zeros=zeros) for _ in range(nC)]
    return pi, v, p


def claim_N3() -> None:
    header("N3  law of total variance; Corr(X_p*, Y) = Var(mu)/Var(Y) in (0,1); collapsed X_delta has Corr = sqrt of it")
    rng = random.Random(SEED + 3)
    ok_tv = ok_cov = ok_true = ok_col = ok_gap = True
    n_strict = 0
    n_det = 0
    for _ in range(150):
        nC, K = rng.randint(2, 4), rng.randint(2, 5)
        pi, v, p = build_model(rng, nC, K, rng.choice([5, 6, 8, 10]), zeros=True)
        mu = [sum((p[c][y] * v[y] for y in range(K)), F(0)) for c in range(nC)]
        Emu = sum((pi[c] * mu[c] for c in range(nC)), F(0))
        varMu = sum((pi[c] * (mu[c] - Emu) ** 2 for c in range(nC)), F(0))
        eVar = sum((pi[c] * sum((p[c][y] * (v[y] - mu[c]) ** 2 for y in range(K)), F(0)) for c in range(nC)), F(0))
        # judge X ~ p*(.|c) drawn independently of Y given c: joint over (c, x, y)
        joint = [(pi[c] * p[c][x] * p[c][y], v[x], v[y]) for c in range(nC) for x in range(K) for y in range(K)]
        EX, EY, VX, VY, CV = stats(joint)
        ok_tv &= (VY == eVar + varMu)
        ok_cov &= (CV == varMu and VX == VY)
        # collapsed judge X_delta = a mu(c) + b, a > 0
        a = F(rng.randint(1, 5), rng.randint(1, 4))
        b = F(rng.randint(-4, 4), rng.randint(1, 3))
        jc = [(pi[c] * p[c][y], a * mu[c] + b, v[y]) for c in range(nC) for y in range(K)]
        _, _, VXd, VYd, CVd = stats(jc)
        if eVar > 0 and varMu > 0:
            r = varMu / VY
            ok_true &= (0 < r < 1) and CV > 0 and (CV * CV) / (VX * VY) == r * r
            corr_d_sq = (CVd * CVd) / (VXd * VYd)
            ok_col &= (CVd > 0 and corr_d_sq == r)           # Corr(X_delta, Y)^2 = r, Corr > 0  =>  Corr = sqrt(r)
            ok_gap &= (corr_d_sq > (CV * CV) / (VX * VY))      # sqrt(r) > r  <=>  r^2 < r
            n_strict += 1
        elif eVar == 0 and varMu > 0:
            n_det += 1
            r = varMu / VY
            # Y is a function of c: r = 1 and the collapsed judge is just as correlated (no strict gain)
            ok_col &= (r == 1 and (CVd * CVd) / (VXd * VYd) == 1)
    check("N3a  Var(Y) = E[Var(Y|C)] + Var(mu_Y(C)) (finite joint law, exact)", ok_tv, "150 random models")
    check("N3b  X ~ p*(.|c) independent of Y given c:  Cov(X,Y) = Var(mu_Y(C))  and  Var(X) = Var(Y)", ok_cov)
    check("N3c  Corr(X_p*, Y) = Var(mu)/Var(Y) strictly inside (0,1) when E[Var(Y|C)] > 0 and Var(mu) > 0", ok_true, f"{n_strict} models with both variances > 0")
    check("N3d  collapsed X_delta = a mu(C) + b (a > 0):  Corr(X_delta,Y)^2 = Corr(X_p*,Y), Corr > 0  =>  Corr(X_delta,Y) = sqrt(Corr(X_p*,Y))", ok_col)
    check("N3e  sqrt(rho) > rho strictly in every such model (squares compared exactly: r > r^2)", ok_gap)
    # explicit deterministic model (random pmfs almost never produce one): Y = 0 on prompt A, Y = 2 on prompt B
    pi_d, v_d = [F(1, 2), F(1, 2)], [F(0), F(1), F(2)]
    p_d = [[F(1), F(0), F(0)], [F(0), F(0), F(1)]]
    mu_d = [sum((p_d[c][y] * v_d[y] for y in range(3)), F(0)) for c in range(2)]
    js_d = [(pi_d[c] * p_d[c][x] * p_d[c][y], v_d[x], v_d[y]) for c in range(2) for x in range(3) for y in range(3)]
    jc_d = [(pi_d[c] * p_d[c][y], F(3) * mu_d[c] + F(1), v_d[y]) for c in range(2) for y in range(3)]
    _, _, VXs, VYs, CVs = stats(js_d)
    _, _, VXc, VYc, CVc = stats(jc_d)
    r_d = CVs * CVs / (VXs * VYs)
    c_d = CVc * CVc / (VXc * VYc)
    check("N3f  [claimed-false case] if Y is a function of c (E[Var(Y|C)] = 0) there is NO strict gain: r = 1 and the collapsed correlation is also 1", r_d == 1 and c_d == 1 and CVs > 0 and CVc > 0, f"Corr^2 sampled = {r_d}, collapsed = {c_d}; {n_det} random models also met")

    # a < 0 flips the sign; a = 0 makes the correlation undefined
    pi, v, p = [F(1, 2), F(1, 2)], [F(0), F(1), F(2)], [[F(1, 2), F(0), F(1, 2)], [F(0), F(1, 2), F(1, 2)]]
    mu = [sum((p[c][y] * v[y] for y in range(3)), F(0)) for c in range(2)]
    cov_neg = stats([(pi[c] * p[c][y], -mu[c], v[y]) for c in range(2) for y in range(3)])[4]
    var_zero = stats([(pi[c] * p[c][y], F(7), v[y]) for c in range(2) for y in range(3)])[2]
    check("N3g  [claimed-false case] the theorem needs a > 0: with a < 0 the collapsed correlation is negative (sign flip)", cov_neg < 0, f"Cov = {cov_neg}")
    check("N3h  [claimed-false case] a = 0 (constant judge): Var(X_delta) = 0, so Pearson is undefined (0/0), not 'sqrt(rho)'", var_zero == 0)

    # conditional independence is essential: X = Y (comonotone) attains Corr 1 >= sqrt(r)
    pi, v, p = [F(1, 2), F(1, 2)], [F(0), F(1), F(2)], [[F(1, 2), F(0), F(1, 2)], [F(0), F(1, 2), F(1, 2)]]
    mu = [sum((p[c][y] * v[y] for y in range(3)), F(0)) for c in range(2)]
    jx = [(pi[c] * p[c][y], v[y], v[y]) for c in range(2) for y in range(3)]
    _, _, VXx, VYx, CVx = stats(jx)
    check("N3i  [claimed-false case] without X independent of Y given c, a judge with the right marginal can have Corr = 1 (X = Y), so the hypothesis is essential", (CVx * CVx) / (VXx * VYx) == 1)


# ----------------------------------------------------------------------------------------------
# N3j. what the plan's Experiment 1(b) needs: collapse to the judge's OWN mean, and the greedy mode
# ----------------------------------------------------------------------------------------------
def claim_N3j() -> None:
    header("N3j  (not in the plan) collapsing ANY sampled judge to its own conditional mean raises |Pearson| by 1/sqrt(Var(mu_q)/Var(X)); greedy MODE collapse carries no such guarantee")
    rng = random.Random(SEED + 31)
    ok_ratio = ok_ge = True
    n_strict = 0
    for _ in range(120):
        nC, K = rng.randint(2, 4), rng.randint(2, 5)
        pi, v, p = build_model(rng, nC, K, rng.choice([5, 6, 8]), zeros=True)
        q = [rand_pmf(rng, K, rng.choice([4, 6, 7]), zeros=True) for _ in range(nC)]
        mu_q = [sum((q[c][x] * v[x] for x in range(K)), F(0)) for c in range(nC)]
        # sampled judge X ~ q(.|c), independent of Y given c
        js = [(pi[c] * q[c][x] * p[c][y], v[x], v[y]) for c in range(nC) for x in range(K) for y in range(K)]
        jm = [(pi[c] * p[c][y], mu_q[c], v[y]) for c in range(nC) for y in range(K)]
        _, _, VX, VY, CV = stats(js)
        _, _, VM, _, CM = stats(jm)
        if CV == 0 or VM == 0:
            continue
        ratio = (CM * CM / (VM * VY)) / (CV * CV / (VX * VY))  # corr(mean)^2 / corr(sample)^2
        ok_ratio &= (CM == CV and ratio == VX / VM)
        ok_ge &= (ratio >= 1)
        n_strict += (ratio > 1)
    check("N3j-a  Corr(mean readout, Y)^2 / Corr(sampled readout, Y)^2 = Var(X)/Var(mu_q) >= 1 for an arbitrary judge q (exact)", ok_ratio and ok_ge, f"120 random (p*, q); {n_strict} strict")

    # greedy MODE is a different object: it can LOWER Pearson below the sampled judge, even flip its sign
    pi = [F(1, 2), F(1, 2)]
    v = [F(0), F(1), F(2)]
    p = [[F(45, 100), F(0), F(55, 100)], [F(0), F(6, 10), F(4, 10)]]   # mu = 1.1 and 1.4, but mode = 2 and 1
    mode = [2, 1]
    mu = [sum((p[c][y] * v[y] for y in range(3)), F(0)) for c in range(2)]
    js = [(pi[c] * p[c][x] * p[c][y], v[x], v[y]) for c in range(2) for x in range(3) for y in range(3)]
    jg = [(pi[c] * p[c][y], v[mode[c]], v[y]) for c in range(2) for y in range(3)]
    _, _, VX, VY, CV = stats(js)
    _, _, VG, _, CG = stats(jg)
    check("N3j-b  [claimed-false case] 'greedy decoding always inflates Pearson over sampling' is false: here the mode (2,1) is anti-ordered to the mean (1.1,1.4)",
          mu[0] < mu[1] and mode[0] > mode[1] and CV > 0 and CG < 0,
          f"sampled judge Cov = {CV} > 0; greedy-mode judge Cov = {CG} < 0")


# ----------------------------------------------------------------------------------------------
# N4. the ordinal {0..4} example of the plan (Stage 1 item 4, last line)
# ----------------------------------------------------------------------------------------------
def claim_N4() -> None:
    header("N4  ordinal {0,...,4} example: Pearson 0.5000 vs 0.7071, Brier 4x, RPS 2x, cross-entropy 0.6931 vs +inf")
    K = 5
    pi = [F(1, 2), F(1, 2)]
    p = [[F(1, 2), F(0), F(1, 2), F(0), F(0)], [F(0), F(0), F(1, 2), F(0), F(1, 2)]]  # A: uniform {0,2};  B: uniform {2,4}
    v = [F(i) for i in range(K)]
    mu = [sum((p[c][y] * v[y] for y in range(K)), F(0)) for c in range(2)]
    qc = [[F(1) if k == int(mu[c]) else F(0) for k in range(K)] for c in range(2)]   # point mass at the conditional mean (1 and 3)
    joint = [(pi[c] * p[c][x] * p[c][y], v[x], v[y]) for c in range(2) for x in range(K) for y in range(K)]
    jc = [(pi[c] * p[c][y], mu[c], v[y]) for c in range(2) for y in range(K)]
    _, _, VX, VY, CV = stats(joint)
    _, _, VXd, _, CVd = stats(jc)
    r = CV / VY
    corr_d_sq = CVd * CVd / (VXd * VY)
    check("N4a  Corr(X_p*, Y) = 1/2 and Corr(X_delta, Y)^2 = 1/2 (so Corr(X_delta, Y) = sqrt(1/2) = 0.7071)", r == F(1, 2) and corr_d_sq == F(1, 2), f"{float(r):.4f} and {math.sqrt(float(corr_d_sq)):.4f}")
    risk = lambda q, loss: sum((pi[c] * sum((p[c][y] * loss(q[c], y) for y in range(K)), F(0)) for c in range(2)), F(0))
    bt, bc = risk(p, brier), risk(qc, brier)
    rt, rc = risk(p, rps), risk(qc, rps)
    check("N4b  Brier of the true forecast 1/2 vs 2 for the collapsed forecast: ratio 4", (bt, bc) == (F(1, 2), F(2)) and bc / bt == 4, f"{bt} vs {bc}")
    check("N4c  RPS of the true forecast 1/2 vs 1 for the collapsed forecast: ratio 2", (rt, rc) == (F(1, 2), F(1)) and rc / rt == 2, f"{rt} vs {rc}")
    # cross-entropy: the collapsed forecast puts probability 0 on every rating that can occur
    zero_mass_all = all(qc[c][y] == 0 for c in range(2) for y in range(K) if p[c][y] > 0)
    ce_true = sum(float(pi[c]) * sum(float(p[c][y]) * -math.log(float(p[c][y])) for y in range(K) if p[c][y] > 0) for c in range(2))
    check("N4d  cross-entropy: true forecast ln 2 = 0.6931, collapsed forecast +inf (zero probability on every realised rating)", zero_mass_all and abs(ce_true - math.log(2)) < TOL, f"{ce_true:.4f} vs +inf")


# ----------------------------------------------------------------------------------------------
# N5. binary example, p* in {0.1, 0.5, 0.9}, q2 = 0.01 p* + 0.90
# ----------------------------------------------------------------------------------------------
def claim_N5() -> None:
    header("N5  binary example p* in {0.1,0.5,0.9}, q2 = 0.01 p* + 0.90: exact Pearson tie 0.6532, Brier 0.1433 vs 0.4119 (2.87x)")
    ps = [F(1, 10), F(1, 2), F(9, 10)]
    pi = [F(1, 3)] * 3
    q1 = ps
    q2 = [F(1, 100) * x + F(9, 10) for x in ps]

    def corr_sq_and_sign(q):
        # judge draws X ~ Bernoulli(q_c) independent of Y ~ Bernoulli(p*_c) given c
        joint = [(pi[c] * (q[c] if x else 1 - q[c]) * (ps[c] if y else 1 - ps[c]), F(x), F(y)) for c in range(3) for x in (0, 1) for y in (0, 1)]
        _, _, VX, VY, CV = stats(joint)
        return CV * CV / (VX * VY), CV > 0

    c1, s1 = corr_sq_and_sign(q1)
    c2, s2 = corr_sq_and_sign(q2)
    # mean-readout correlation (deterministic judge mu_q(c) = q_c): identical for both because q2 is an increasing affine map of q1
    def mean_corr_sq(q):
        joint = [(pi[c] * (ps[c] if y else 1 - ps[c]), q[c], F(y)) for c in range(3) for y in (0, 1)]
        _, _, VX, VY, CV = stats(joint)
        return CV * CV / (VX * VY)
    m1, m2 = mean_corr_sq(q1), mean_corr_sq(q2)
    check("N5a  mean-readout Pearson is identical for q1 and q2 (affine, a = 0.01 > 0): Corr^2 = 32/75, Corr = 0.6532", m1 == m2 == F(32, 75), f"sqrt(32/75) = {math.sqrt(32 / 75):.4f}")
    check("N5b  [claimed-false case] the tie is specific to the MEAN (probability-as-score) readout: for a SAMPLED 0/1 judge X ~ Bernoulli(q) the Pearson values are NOT tied", s1 and s2 and c1 != c2 and c1 > c2, f"sampled Corr^2: q1 = {c1} ({float(c1):.4f}), q2 = {c2} ({float(c2):.6f})")
    bs = lambda q: sum((pi[c] * sum(((ps[c] if y else 1 - ps[c]) * (q[c] - y) ** 2 for y in (0, 1)), F(0)) for c in range(3)), F(0))
    b1, b2 = bs(q1), bs(q2)
    check("N5c  binary Brier E (q - Y)^2: 43/300 = 0.1433 for q1 and 1235707/3000000 = 0.4119 for q2; ratio 2.87", b1 == F(43, 300) and b2 == F(1235707, 3000000) and 2.87 < float(b2 / b1) < 2.88, f"{float(b1):.4f}, {float(b2):.4f}, ratio {float(b2 / b1):.4f}")
    kb = lambda q: sum((pi[c] * sum(((ps[c] if y else 1 - ps[c]) * brier([1 - q[c], q[c]], y) for y in (0, 1)), F(0)) for c in range(3)), F(0))
    check("N5d  [imprecise in the plan] the plan's Brier ||q - e_Y||^2 has two components, so for binary data it is 2 x E(q-Y)^2: 43/150 = 0.2867 and 1235707/1500000 = 0.8238 (ratio unchanged); 0.1433 and 0.4119 use the one-component convention", kb(q1) == 2 * b1 and kb(q2) == 2 * b2, f"{float(kb(q1)):.4f} and {float(kb(q2)):.4f}")
    ce = lambda q: sum(float(pi[c]) * sum(float(ps[c] if y else 1 - ps[c]) * -math.log(float(q[c] if y else 1 - q[c])) for y in (0, 1)) for c in range(3))
    print(f"INFO  cross-entropy (nats, finite for both): q1 {ce(q1):.4f}, q2 {ce(q2):.4f}")
    # scalar L2 of a SAMPLED 0/1 judge, E (X - Y)^2 = p + q(1 - 2p) per prompt: linear in q, so it is improper
    sq = lambda q: sum((pi[c] * sum((q[c] if x else 1 - q[c]) * (ps[c] if y else 1 - ps[c]) * (x - y) ** 2 for x in (0, 1) for y in (0, 1)) for c in range(3)), F(0))
    q_pt = [F(1) if x > F(1, 2) else F(0) for x in ps]  # point-mass judge: always the more likely rating (tie at 1/2 broken to 0)
    check("N5e  [claimed-false case] scalar L2 on SAMPLED draws is improper: the 0/1 point-mass judge (7/30 = 0.2333) beats the true forecast q = p* (43/150 = 0.2867)", sq(q1) == F(43, 150) and sq(q_pt) == F(7, 30) and sq(q_pt) < sq(q1), f"{float(sq(q_pt)):.4f} < {float(sq(q1)):.4f}")


# ----------------------------------------------------------------------------------------------
# N6. sqrt must be applied after averaging
# ----------------------------------------------------------------------------------------------
def claim_N6() -> None:
    header("N6  sqrt(RPS) must be taken AFTER averaging over items (Jensen): mean of sqrt < sqrt of mean")
    rng = random.Random(SEED + 6)
    ok = True
    for _ in range(300):
        a, b = F(rng.randint(0, 40), 8), F(rng.randint(0, 40), 8)
        # ((sqrt a + sqrt b)/2)^2 = (a+b+2 sqrt(ab))/4 < (a+b)/2  <=>  4ab < (a+b)^2  <=>  (a-b)^2 > 0
        ok &= ((4 * a * b < (a + b) ** 2) == (a != b))
    check("N6a  (sqrt a + sqrt b)/2 < sqrt((a+b)/2) iff a != b (exact: 4ab < (a+b)^2 iff (a-b)^2 > 0)", ok, "300 random pairs")
    # a concrete RPS example: two items, one perfect (RPS 0), one with RPS 1
    mean_of_sqrt = (math.sqrt(0) + math.sqrt(1)) / 2
    sqrt_of_mean = math.sqrt((0 + 1) / 2)
    check("N6b  [claimed-false case] 'average per-item sqrt(RPS)' != 'sqrt of average RPS': 0.5000 vs 0.7071 for per-item RPS (0, 1)", mean_of_sqrt < sqrt_of_mean, f"{mean_of_sqrt:.4f} vs {sqrt_of_mean:.4f}")


# ----------------------------------------------------------------------------------------------
# N7. Pearson affine invariance needs a > 0
# ----------------------------------------------------------------------------------------------
def claim_N7() -> None:
    header("N7  Corr(aX+b, Y) = Corr(X, Y) needs a > 0 (a < 0 flips the sign; a = 0 is undefined)")
    rng = random.Random(SEED + 7)
    ok_pos = ok_neg = True
    for _ in range(200):
        K = rng.randint(2, 4)
        pr = rand_pmf(rng, K * K, 12, zeros=True)
        pts = [(pr[i * K + j], F(rng.randint(-5, 5)), F(rng.randint(-5, 5))) for i in range(K) for j in range(K)]
        _, _, VX, VY, CV = stats(pts)
        if VX == 0 or VY == 0:
            continue
        a = F(rng.randint(1, 6), rng.randint(1, 3))
        b = F(rng.randint(-5, 5))
        _, _, VXa, _, CVa = stats([(q, a * x + b, y) for q, x, y in pts])
        ok_pos &= (CVa * CVa / (VXa * VY) == CV * CV / (VX * VY)) and ((CVa > 0) == (CV > 0))
        _, _, VXn, _, CVn = stats([(q, -a * x + b, y) for q, x, y in pts])
        ok_neg &= (CVn * CVn / (VXn * VY) == CV * CV / (VX * VY)) and ((CVn > 0) == (CV < 0)) and (CV != 0 or CVn == 0)
    check("N7a  a > 0: Corr^2 and the sign of the covariance are unchanged", ok_pos, "200 random joint laws")
    check("N7b  [claimed-false case] a < 0: Corr(aX+b, Y) = -Corr(X, Y), so 'invariant for all affine maps' is false", ok_neg)


# ----------------------------------------------------------------------------------------------
# N8. L2 identities, coupling counterexample, independent-judge formula
# ----------------------------------------------------------------------------------------------
def claim_N8() -> None:
    header("N8  standardized L2, MSE decomposition, fair-coin couplings, independent judge, nearest-grid minimiser")
    rng = random.Random(SEED + 8)
    worst1 = worst2 = 0.0
    for _ in range(200):
        K = rng.randint(2, 4)
        pr = rand_pmf(rng, K * K, 12, zeros=True)
        pts = [(pr[i * K + j], F(rng.randint(-5, 5)), F(rng.randint(-5, 5))) for i in range(K) for j in range(K)]
        EX, EY, VX, VY, CV = stats(pts)
        if VX == 0 or VY == 0:
            continue
        sx, sy = math.sqrt(float(VX)), math.sqrt(float(VY))
        corr = float(CV) / (sx * sy)
        l2std = sum(float(q) * ((float(x - EX)) / sx - (float(y - EY)) / sy) ** 2 for q, x, y in pts)
        worst1 = max(worst1, abs(l2std - 2 * (1 - corr)))
        mse = sum(float(q) * float(x - y) ** 2 for q, x, y in pts)
        dec = (float(EX - EY)) ** 2 + (sx - sy) ** 2 + 2 * sx * sy * (1 - corr)
        worst2 = max(worst2, abs(mse - dec) / max(1.0, abs(mse)))
    check(f"N8a  E[(z_X - z_Y)^2] = 2(1 - Corr)  (float, tolerance {TOL * 1e3:g})", worst1 < TOL * 1e3, f"max abs error {worst1:.0e}")
    check(f"N8b  E[(X-Y)^2] = (mu_X-mu_Y)^2 + (s_X-s_Y)^2 + 2 s_X s_Y (1-Corr)  (float, tolerance {TOL * 1e3:g})", worst2 < TOL * 1e3, f"max rel error {worst2:.0e}")

    half = F(1, 2)
    px = [half, half]
    cases = {
        "Y = X": [(half, F(0), F(0)), (half, F(1), F(1))],
        "Y independent of X": [(F(1, 4), F(x), F(y)) for x in (0, 1) for y in (0, 1)],
        "Y = 1 - X": [(half, F(0), F(1)), (half, F(1), F(0))],
    }
    vals = {k: sum((q * (x - y) ** 2 for q, x, y in v), F(0)) for k, v in cases.items()}
    db = sum(((a - b) ** 2 for a, b in zip(px, px)), F(0))
    check("N8c  fair coins: D_B(p_X||p_Y) = 0 yet E[(X-Y)^2] = 0, 1/2, 1 for Y = X, independent, 1 - X", db == 0 and list(vals.values()) == [F(0), F(1, 2), F(1)], str({k: str(v) for k, v in vals.items()}))

    ok_ind = True
    for _ in range(100):
        K = rng.randint(2, 4)
        vx = [F(rng.randint(-4, 6)) for _ in range(K)]
        vy = [F(rng.randint(-4, 6)) for _ in range(K)]
        px_, py_ = rand_pmf(rng, K, 8, zeros=True), rand_pmf(rng, K, 9, zeros=True)
        mse = sum((px_[i] * py_[j] * (vx[i] - vy[j]) ** 2 for i in range(K) for j in range(K)), F(0))
        mx = sum((px_[i] * vx[i] for i in range(K)), F(0))
        my = sum((py_[j] * vy[j] for j in range(K)), F(0))
        sx2 = sum((px_[i] * (vx[i] - mx) ** 2 for i in range(K)), F(0))
        sy2 = sum((py_[j] * (vy[j] - my) ** 2 for j in range(K)), F(0))
        ok_ind &= (mse == (mx - my) ** 2 + sx2 + sy2)
    check("N8d  independent X, Y: E[(X-Y)^2] = (mu_X-mu_Y)^2 + sigma_X^2 + sigma_Y^2 exactly", ok_ind, "100 random laws")

    # minimiser over judge laws supported on the grid {0..4}: a POINT MASS at the grid point nearest mu_Y, not delta_{mu_Y} unless mu_Y is on the grid
    grid = [F(i) for i in range(5)]
    ok_min = True
    n_off = 0
    for _ in range(80):
        py_ = rand_pmf(rng, 5, rng.choice([7, 9, 11]), zeros=True)
        my = sum((py_[j] * grid[j] for j in range(5)), F(0))
        sy2 = sum((py_[j] * (grid[j] - my) ** 2 for j in range(5)), F(0))
        cost = lambda px_: sum((px_[i] * py_[j] * (grid[i] - grid[j]) ** 2 for i in range(5) for j in range(5)), F(0))
        point_costs = [cost([F(1) if i == k else F(0) for i in range(5)]) for k in range(5)]
        best = min(point_costs)
        near = min(abs(g - my) for g in grid)
        ok_min &= (best == near ** 2 + sy2)
        for _ in range(8):
            ok_min &= (cost(rand_pmf(rng, 5, rng.choice([6, 10]), zeros=True)) >= best)
        if my.denominator != 1:
            n_off += 1
            # the mixture of the two neighbouring grid points with mean exactly mu_Y (the 'delta_{mu_Y}' proxy) is strictly worse
            lo = int(my)  # floor for non-negative
            w = my - lo
            mix = [F(0)] * 5
            mix[lo], mix[lo + 1] = 1 - w, w
            ok_min &= (cost(mix) > best) if w != F(1, 2) else (cost(mix) >= best)
    check("N8e  independent judge on the grid {0..4}: the minimiser is a point mass at the grid point nearest mu_Y; the exact-mean mixture is worse (so 'delta_{mu_Y}' needs mu_Y on the support)", ok_min, f"80 random laws, {n_off} with mu_Y off the grid")


# ----------------------------------------------------------------------------------------------
# N9. properness decompositions
# ----------------------------------------------------------------------------------------------
def claim_N9() -> None:
    header("N9  Brier and RPS: expected loss = divergence + irreducible term; strictly minimised at q = p; cross-entropy = KL + entropy")
    rng = random.Random(SEED + 9)
    ok_b = ok_r = ok_sb = ok_sr = True
    ok_ce = True
    for _ in range(200):
        K = rng.randint(2, 5)
        p = rand_pmf(rng, K, rng.choice([5, 8, 12]), zeros=True)
        q = rand_pmf(rng, K, rng.choice([5, 7, 11]), zeros=True)
        eb = lambda r: sum((p[y] * brier(r, y) for y in range(K)), F(0))
        er = lambda r: sum((p[y] * rps(r, y) for y in range(K)), F(0))
        ok_b &= (eb(q) == sq_norm([a - b for a, b in zip(p, q)]) + 1 - sq_norm(p))
        Fp, Fq = cdf(p), cdf(q)
        ok_r &= (er(q) == sum(((Fq[t] - Fp[t]) ** 2 for t in range(K)), F(0)) + sum((Fp[t] * (1 - Fp[t]) for t in range(K)), F(0)))
        if q != p:
            ok_sb &= (eb(q) > eb(p))
            ok_sr &= (er(q) > er(p))
        # cross-entropy (float): E[-log q_Y] - H(p) = KL(p||q) >= 0, q strictly positive
        qs = rand_pmf(rng, K, 11)
        ce = sum(float(p[y]) * -math.log(float(qs[y])) for y in range(K) if p[y] > 0)
        H = sum(float(p[y]) * -math.log(float(p[y])) for y in range(K) if p[y] > 0)
        kl = sum(float(p[y]) * math.log(float(p[y]) / float(qs[y])) for y in range(K) if p[y] > 0)
        ok_ce &= abs(ce - H - kl) < TOL * 100 and kl >= -TOL
    check("N9a  E_p[Brier(q, Y)] = ||p - q||^2 + 1 - ||p||^2 (exact)", ok_b, "200 random (K, p, q)")
    check("N9b  E_p[RPS(q, Y)] = sum_t (F_q(t) - F_p(t))^2 + sum_t F_p(t)(1 - F_p(t)) (exact)", ok_r)
    check("N9c  Brier strictly proper: q != p implies strictly larger expected Brier", ok_sb)
    check("N9d  RPS strictly proper: q != p implies strictly larger expected RPS", ok_sr)
    check(f"N9e  cross-entropy = entropy + KL >= entropy (float, tolerance {TOL * 100:g})", ok_ce)


# ----------------------------------------------------------------------------------------------
# N10. M-sample rollout versus single-pass logit readout (plan, Experiment 4)
# ----------------------------------------------------------------------------------------------
def claim_N10() -> None:
    header("N10  an M-sample rollout pmf q_M of a forecast q has E||q_M - q||^2 = (1 - ||q||^2)/M; the single-pass logit readout has Monte-Carlo error 0")
    rng = random.Random(SEED + 10)
    ok = True
    cfg = [(6, 1), (6, 5), (3, 20), (4, 12)]
    for K, M in cfg:
        q = rand_pmf(rng, K, 10, zeros=False)
        e = F(0)
        for cnt in compositions(M, K):
            pr = multinom(cnt) * prod_fr(q[a] ** cnt[a] for a in range(K))
            e += pr * sum(((F(cnt[a], M) - q[a]) ** 2 for a in range(K)), F(0))
        ok &= (e == (1 - sq_norm(q)) / M)
    check("N10a  rollout Brier excess over the readout's own forecast is (1-||q||^2)/M exactly, for (K, M) in {(6,1),(6,5),(3,20),(4,12)}", ok, "M = 1, 5, 20 as in the plan's Experiment 4")
    # the excess of the rollout forecast over the readout forecast is paid at EVERY M < inf: the single pass is the M -> inf limit
    q = [F(6, 10), F(3, 10), F(1, 10)]
    vals = [(1 - sq_norm(q)) / M for M in (1, 5, 20)]
    check("N10b  the excess shrinks like 1/M but never reaches 0: single-pass (M = infinity) is at least as good in expected Brier as any finite M", vals[0] > vals[1] > vals[2] > 0, "excess " + ", ".join(f"{float(x):.4f}" for x in vals))


def main() -> int:
    print("check_plan_claims.py  seed =", SEED, " float tolerance =", TOL)
    claim_N1()
    claim_N2()
    claim_N3()
    claim_N3j()
    claim_N4()
    claim_N5()
    claim_N6()
    claim_N7()
    claim_N8()
    claim_N9()
    claim_N10()
    n_ok = sum(1 for _, ok in RESULTS if ok)
    n_bad = len(RESULTS) - n_ok
    print()
    print(f"SUMMARY: {n_ok} PASS, {n_bad} FAIL, {len(RESULTS)} checks")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
