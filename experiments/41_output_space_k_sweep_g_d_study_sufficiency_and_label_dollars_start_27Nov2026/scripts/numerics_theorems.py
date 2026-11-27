#!/usr/bin/env python3
"""Deterministic numerical checks of every Experiment 41 claim (and of the 11-25 blog identities).

Exact rational arithmetic (fractions.Fraction) wherever the quantity is rational; a named tolerance otherwise.
Every claim prints PASS or FAIL, a false claim is tested as a failing case, the exit code is nonzero on any FAIL,
and two runs print identical output (fixed seeds, no clock, no paths).

    python3 -I numerics_theorems.py

Claim labels match lean/E40_theorems.lean where a Lean theorem exists ("Lean: <name>").
The Monte Carlo block (C9) is a SYNTHETIC known-distribution check; it says nothing about human rating data.
11-27-2026 revision after an independent review: E3, F3, D3 and G3 were rewritten because they were tautological or weaker than their labels (E3 and F3 now enumerate
the exact joint law of (item, draw, rating) instead of re-deriving the same algebra); B6 (decision accuracy is proper), F0 (the tower step) and C9f (chi-squared
excess is (K-1)/N) were added. A claim below is "independent" of the Lean theorem only where it enumerates or simulates the quantity instead of re-using the same formula.
"""
from __future__ import annotations

import itertools
import math
import random
import sys
from fractions import Fraction as F

import numpy as np

FAILS: list[str] = []
TOL = 1e-10  # float tolerance used only where a square root or logarithm appears


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def rand_pmf(rng: random.Random, k: int, den: int = 24) -> list[F]:
    cuts = sorted(rng.randint(0, den) for _ in range(k - 1))
    parts = [b - a for a, b in zip([0] + cuts, cuts + [den])]
    return [F(x, den) for x in parts]


def mean(p, v=None):
    v = list(range(len(p))) if v is None else v
    return sum(F(a) * b for a, b in zip(v, p))


def sqnorm(a):
    return sum(x * x for x in a)


# ------------------------------------------------------------------------------------------------------------------
# A. K = 2 versus K >= 3  (Lean: binary_mean_identifies, mean_not_identifying)
# ------------------------------------------------------------------------------------------------------------------
rng = random.Random(20261127)
pairs, collisions = 0, 0
ok = True
for _ in range(4000):
    p, q = rand_pmf(rng, 2, 8), rand_pmf(rng, 2, 8)
    pairs += 1
    if mean(p) == mean(q):
        collisions += 1
        ok &= (p == q)
check("A1 K=2: equal mean implies equal law (Lean: binary_mean_identifies)", ok and collisions > 100,
      f"{collisions} equal-mean pairs among {pairs}")

ok = True
for K in range(3, 41):
    p = [F(0)] * K
    p[1] = F(1)
    q = [F(0)] * K
    q[0] = q[2] = F(1, 2)
    ok &= (sum(p) == 1 and sum(q) == 1 and mean(p) == mean(q) == 1 and p != q)
check("A2 every K from 3 to 40: a point mass at 1 and a half-half law on {0,2} share the mean 1 (Lean: mean_not_identifying)", ok)

# the claim "the mean identifies the law for K>=3" is FALSE: count equal-mean distinct pairs found at random
coll3 = 0
for _ in range(4000):
    p, q = rand_pmf(rng, 3, 8), rand_pmf(rng, 3, 8)
    if mean(p) == mean(q) and p != q:
        coll3 += 1
check("A3 K=3: random search finds equal-mean pairs with different laws (the false claim fails)", coll3 > 50, f"{coll3} found")

# ------------------------------------------------------------------------------------------------------------------
# B. what accuracy elicits  (Lean: sampled_accuracy_le_max, sampled_accuracy_eq_max_iff, truthful_..., accuracy_prefers_..., decision_...)
# ------------------------------------------------------------------------------------------------------------------
def sampled_acc(p, q):
    return sum(a * b for a, b in zip(p, q))


ok_le, ok_iff, strict_seen = True, True, 0
for _ in range(3000):
    K = rng.randint(2, 7)
    p = rand_pmf(rng, K, 12)
    q = rand_pmf(rng, K, 12)
    m = max(p)
    acc = sampled_acc(p, q)
    ok_le &= acc <= m
    supported_on_argmax = all((qi == 0) or (pi == m) for pi, qi in zip(p, q))
    ok_iff &= ((acc == m) == supported_on_argmax)
    strict_seen += acc < m
check("B1 sampled accuracy never exceeds max p (Lean: sampled_accuracy_le_max)", ok_le, f"strict in {strict_seen} of 3000")
check("B2 sampled accuracy equals max p exactly when q is supported on argmax p (Lean: sampled_accuracy_eq_max_iff)", ok_iff)

ok_truth = True
uniform_support_hits = 0
for _ in range(3000):
    K = rng.randint(2, 7)
    p = rand_pmf(rng, K, 12)
    m = max(p)
    truthful_optimal = sampled_acc(p, p) == m
    uniform_on_support = all((pi == 0) or (pi == m) for pi in p)
    ok_truth &= (truthful_optimal == uniform_on_support)
    uniform_support_hits += uniform_on_support
check("B3 honest forecast q=p is accuracy-optimal exactly when p is uniform on its support (Lean: truthful_sampled_accuracy_optimal_iff)",
      ok_truth, f"{uniform_support_hits} uniform-on-support laws among 3000")
p = [F(3, 5), F(2, 5)]
check("B4 instance (3/5,2/5): honest 13/25 < overconfident 3/5 (Lean: accuracy_prefers_overconfidence)",
      sampled_acc(p, p) == F(13, 25) and sampled_acc(p, [F(1), F(0)]) == F(3, 5) and F(13, 25) < F(3, 5))

def decision_acc(p, q):  # argmax with ties to class 0, two classes
    return p[0] if q[1] <= q[0] else p[1]

check("B5 decision accuracy: honest and point-mass forecasts with the same argmax earn the same score (Lean: decision_accuracy_not_strict)",
      decision_acc(p, p) == decision_acc(p, [F(1), F(0)]) and p != [F(1), F(0)])
ok_proper, strict_seen2, tie_seen = True, 0, 0
for _ in range(3000):
    pp2, qq2 = rand_pmf(rng, 2, 12), rand_pmf(rng, 2, 12)
    ok_proper &= decision_acc(pp2, qq2) <= decision_acc(pp2, pp2)
    strict_seen2 += decision_acc(pp2, qq2) < decision_acc(pp2, pp2)
    tie_seen += (decision_acc(pp2, qq2) == decision_acc(pp2, pp2)) and pp2 != qq2
check("B6 decision accuracy is proper: no forecast q beats the true law p on expected decision accuracy, K=2 (Lean: decision_accuracy_proper)", ok_proper,
      f"strictly worse in {strict_seen2} of 3000, tied with a different forecast in {tie_seen}")

# ------------------------------------------------------------------------------------------------------------------
# C. Brier locality  (Lean: brier_binary_local, brier_not_local_three)
# ------------------------------------------------------------------------------------------------------------------
def brier(q, y):
    return sum((qi - (1 if j == y else 0)) ** 2 for j, qi in enumerate(q))

ok = True
for _ in range(2000):
    q = rand_pmf(rng, 2, 20)
    for y in (0, 1):
        ok &= brier(q, y) == 2 * (1 - q[y]) ** 2
check("C1 K=2: Brier at outcome y equals 2(1-q_y)^2 (Lean: brier_binary_local)", ok)
q1, q2 = [F(1, 2), F(1, 2), F(0)], [F(1, 2), F(1, 4), F(1, 4)]
check("C2 K=3 instance: same q_0=1/2, Brier 1/2 versus 3/8 (Lean: brier_not_local_three)",
      q1[0] == q2[0] and brier(q1, 0) == F(1, 2) and brier(q2, 0) == F(3, 8))
nonlocal_pairs = 0
for _ in range(3000):
    qa, qb = rand_pmf(rng, 3, 8), rand_pmf(rng, 3, 8)
    if qa[0] == qb[0] and brier(qa, 0) != brier(qb, 0):
        nonlocal_pairs += 1
check("C3 K=3: random search finds non-local pairs (locality fails for K>=3)", nonlocal_pairs > 20, f"{nonlocal_pairs} found")

# ------------------------------------------------------------------------------------------------------------------
# D. moment-blindness  (Lean: same_two_moments_different_law)
# ------------------------------------------------------------------------------------------------------------------
pp = [F(1, 4), F(0), F(1, 2), F(0), F(1, 4)]
qq = [F(1, 5)] * 5
m1p, m1q = mean(pp), mean(qq)
m2p = sum(F(i * i) * x for i, x in enumerate(pp))
m2q = sum(F(i * i) * x for i, x in enumerate(qq))
check("D1 same mean (2), same second moment (6), different law, Brier divergence 7/40 (Lean: same_two_moments_different_law)",
      m1p == m1q == 2 and m2p == m2q == 6 and pp != qq and sqnorm([a - b for a, b in zip(pp, qq)]) == F(7, 40))


def variance_components(item_laws, v):
    """Population one-way components: sigma_c^2 = Var(item mean), sigma_w^2 = mean within-item variance."""
    n = len(item_laws)
    mus = [sum(F(vv) * x for vv, x in zip(v, p)) for p in item_laws]
    sec = [sum(F(vv) ** 2 * x for vv, x in zip(v, p)) for p in item_laws]
    mbar = sum(mus) / n
    sc2 = sum(m * m for m in mus) / n - mbar ** 2
    sw2 = sum(s - m * m for s, m in zip(sec, mus)) / n
    return sc2, sw2


v5 = [0, 1, 2, 3, 4]
other = [F(0), F(0), F(0), F(0), F(1)]  # a second item type: point mass at 4
pop_p = [pp, other, other, pp]
pop_q = [qq, other, other, qq]
sc_p, sw_p = variance_components(pop_p, v5)
sc_q, sw_q = variance_components(pop_q, v5)
same_G = all(sc_p / (sc_p + sw_p / k) == sc_q / (sc_q + sw_q / k) for k in range(1, 61))
check("D2 two populations that differ only by replacing p with q (equal mean, equal variance) have identical variance components and identical G(k) for k=1..60",
      (sc_p, sw_p) == (sc_q, sw_q) and same_G and sc_p > 0,
      f"sigma_c^2={sc_p}, sigma_w^2={sw_p}")
pop_div = sum(sqnorm([a - b for a, b in zip(lp, lq)]) for lp, lq in zip(pop_p, pop_q)) / len(pop_p)
check("D3 ...while the population-average squared distance between the two populations' item laws is 7/80 > 0 (p and q differ on half of the four items), "
      "although every G-study quantity above is identical", pop_div == F(7, 80) and pop_div > 0, f"average over items = {pop_div}")

# ------------------------------------------------------------------------------------------------------------------
# E. law of total variance and ICC(1) = rho  (Lean: total_variance, icc_eq_rho)
# F. the ceiling  (Lean: cov_sq_le, ceiling_sq, ceiling_rho, ceiling_attained)
# ------------------------------------------------------------------------------------------------------------------
ok_tv = ok_icc = ok_cs = ok_attain = ok_sampled = ok_collapsed = ok_tower = True
strict_ceiling = 0
collapsed_gt_sampled = 0
for _ in range(600):
    n, K = rng.randint(2, 7), rng.randint(2, 6)
    v = sorted(rng.sample(range(0, 10), K))
    laws = [rand_pmf(rng, K, 10) for _ in range(n)]
    mus = [sum(F(vv) * x for vv, x in zip(v, p)) for p in laws]
    sec = [sum(F(vv) ** 2 * x for vv, x in zip(v, p)) for p in laws]
    mbar = sum(mus) / n
    total = sum(sec) / n - mbar ** 2
    between = sum(m * m for m in mus) / n - mbar ** 2
    within = sum(s - m * m for s, m in zip(sec, mus)) / n
    ok_tv &= total == between + within
    if total > 0:
        rho = between / total
        ok_icc &= rho == between / (between + within)
        # sampled judge X ~ p_c independent of Y given c: Corr(X, Y) = Cov / Var = between / total = rho
        # E3 (independent of the algebra above): enumerate the exact joint law of (item c, draw X, rating Y), X and Y independent draws from p_c given c
        ex = ey = exy_j = ex2 = ey2 = F(0)
        for c in range(n):
            for xi in range(K):
                for yi in range(K):
                    w = F(1, n) * laws[c][xi] * laws[c][yi]
                    ex += w * v[xi]
                    ey += w * v[yi]
                    exy_j += w * v[xi] * v[yi]
                    ex2 += w * v[xi] ** 2
                    ey2 += w * v[yi] ** 2
        ok_sampled &= (ex2 - ex * ex == ey2 - ey * ey == total) and ((exy_j - ex * ey) / total == rho)
        # F3 (independent): the collapsed judge f(c) = mu_c against Y, from the enumerated joint law of (c, Y)
        ef = ey_c = efy = ef2 = ey2_c = F(0)
        for c in range(n):
            for yi in range(K):
                w = F(1, n) * laws[c][yi]
                ef += w * mus[c]
                ey_c += w * v[yi]
                efy += w * mus[c] * v[yi]
                ef2 += w * mus[c] ** 2
                ey2_c += w * v[yi] ** 2
        varf_c, vary_c, cov_c = ef2 - ef * ef, ey2_c - ey_c * ey_c, efy - ef * ey_c
        if varf_c > 0:
            ok_collapsed &= (cov_c * cov_c / (varf_c * vary_c) == rho)
            if rho < 1:
                collapsed_gt_sampled += 1
                ok_collapsed &= float(rho) < math.sqrt(float(rho))
        # F0 (independent of the Lean-side algebra): the tower step, Cov(f(c), Y) from the joint (c, y) equals Cov(f(c), mu(c))
        fz = [F(rng.randint(-6, 6), rng.randint(1, 4)) for _ in range(n)]
        efz = sum(fz) / n
        e_fy = sum(F(1, n) * laws[c][yi] * fz[c] * v[yi] for c in range(n) for yi in range(K))
        ok_tower &= (e_fy - efz * ey_c) == (sum(fc * m for fc, m in zip(fz, mus)) / n - efz * mbar)
        # ceiling: Cov(f(c), Y)^2 <= Var(f) * between for random item-level f
        for _ in range(5):
            f = [F(rng.randint(-6, 6), rng.randint(1, 4)) for _ in range(n)]
            fbar = sum(f) / n
            varf = sum(x * x for x in f) / n - fbar ** 2
            cov = sum(fc * m for fc, m in zip(f, mus)) / n - fbar * mbar
            ok_cs &= cov * cov <= varf * between
            strict_ceiling += cov * cov < varf * between
        f = [2 * m + 3 for m in mus]  # affine copy of the item mean
        fbar = sum(f) / n
        varf = sum(x * x for x in f) / n - fbar ** 2
        cov = sum(fc * m for fc, m in zip(f, mus)) / n - fbar * mbar
        ok_attain &= (varf == 0) or (cov * cov == varf * between)
check("E1 Var(Y) = Var(item mean) + mean within-item variance on random finite populations (Lean: total_variance)", ok_tv)
check("E2 rho = Var(mu)/Var(Y) equals the one-way intraclass correlation sigma_c^2/(sigma_c^2+sigma_w^2) (Lean: icc_eq_rho)", ok_icc)
check("E3 sampled judge X~p_c (independent of Y given c), enumerated over the exact joint law of (c, X, Y): Var(X) = Var(Y) and Corr(X,Y) = rho exactly", ok_sampled)
check("F1 ceiling: Cov(f(c),Y)^2 <= Var(f) Var(mu) for every item-level f (Lean: cov_sq_le, ceiling_sq, ceiling_rho, ceiling_pooled_rho)", ok_cs,
      f"strict in {strict_ceiling} random predictors")
check("F2 the deterministic item mean (and any affine copy) attains the ceiling (Lean: ceiling_attained)", ok_attain)
check("F0 tower step: Cov(f(c), Y) from the enumerated joint law of (c, Y) equals Cov(f(c), mu(c)) exactly on random populations (Lean: covPooled_eq_covF)", ok_tower)
check("F3 collapsed judge f(c)=mu_c, enumerated over the exact joint law of (c, Y): Corr^2 = rho exactly, hence Corr = sqrt(rho) > rho = Corr of the sampled judge whenever 0 < rho < 1 "
      "(Lean: ceiling_attained_pooled)", ok_collapsed and collapsed_gt_sampled > 100, f"{collapsed_gt_sampled} populations with 0 < rho < 1")

# ------------------------------------------------------------------------------------------------------------------
# G. standardized L2, MSE decomposition, affine invariance, binary Brier  (Exp 35 Lean: standardized_mse_corr, mse_decomposition,
#    pearson_affine_invariant, binary_brier_eq_two_l2) -- re-run here for the MSE + L2 comment
# ------------------------------------------------------------------------------------------------------------------
nrng = np.random.default_rng(20261127)
ok_std = ok_mse = ok_aff = ok_aff_b = True
mse_changed = brier_changed = n_bin = 0
for _ in range(500):
    n = int(nrng.integers(5, 60))
    x, y = nrng.normal(size=n), nrng.normal(size=n) + 0.3 * nrng.normal(size=n)
    mx, my, sx, sy = x.mean(), y.mean(), x.std(), y.std()
    r = np.mean((x - mx) * (y - my)) / (sx * sy)
    zx, zy = (x - mx) / sx, (y - my) / sy
    ok_std &= abs(np.mean((zx - zy) ** 2) - 2 * (1 - r)) < 1e-10
    ok_mse &= abs(np.mean((x - y) ** 2) - ((mx - my) ** 2 + (sx - sy) ** 2 + 2 * sx * sy * (1 - r))) < 1e-9
    a, b = float(nrng.uniform(0.2, 5)), float(nrng.normal())
    r2 = np.mean(((a * x + b) - (a * x + b).mean()) * (y - my)) / ((a * x + b).std() * sy)
    ok_aff &= abs(r2 - r) < 1e-10
    mse_changed += abs(np.mean((a * x + b - y) ** 2) - np.mean((x - y) ** 2)) > 1e-9
    # a probability forecast q in (0,1) of a binary label, and the positive-affine copy q' = a q + (1-a)/2 (still in (0,1)): same Corr, different Brier
    yb = (nrng.uniform(size=n) < 0.5).astype(float)
    qb = np.clip(0.5 + 0.3 * (yb - 0.5) + 0.2 * nrng.normal(size=n), 0.02, 0.98)
    a2 = float(nrng.uniform(0.3, 0.9))
    qb2 = a2 * qb + (1 - a2) * 0.5
    if yb.std() > 0:
        ok_aff_b &= abs(np.corrcoef(qb2, yb)[0, 1] - np.corrcoef(qb, yb)[0, 1]) < 1e-10
        brier_changed += abs(np.mean((qb2 - yb) ** 2) - np.mean((qb - yb) ** 2)) > 1e-9
        n_bin += 1
check("G1 standardized L2 identity: mean (z_x - z_y)^2 = 2(1 - Corr)  (Exp 35 Lean: standardized_mse_corr)", ok_std)
check("G2 MSE decomposition: (mu_x-mu_y)^2 + (s_x-s_y)^2 + 2 s_x s_y (1-Corr)  (Exp 35 Lean: mse_decomposition)", ok_mse)
check("G3 Corr(a x + b, y) = Corr(x, y) for a>0, yet the mean squared error changes (on at least 99% of 500 draws) and so does the binary Brier score of the affine-shrunk "
      "probability forecast (on at least 95% of the binary draws)  (Exp 35 Lean: pearson_affine_invariant)",
      ok_aff and ok_aff_b and mse_changed >= 0.99 * 500 and brier_changed >= 0.95 * n_bin, f"MSE changed in {mse_changed} of 500; Brier changed in {brier_changed} of {n_bin}")
ok = True
for _ in range(1000):
    q = F(rng.randint(0, 20), 20)
    for y in (0, 1):
        two_class = (q - y) ** 2 + ((1 - q) - (1 - y)) ** 2
        ok &= two_class == 2 * (q - y) ** 2
check("G4 two-class Brier equals twice the scalar squared error (Exp 35 Lean: binary_brier_eq_two_l2); equal only up to the factor 2", ok)

# the 11-25 blog sentence "on a binary label the second line is the Brier score": sampled draw versus forecast number
ok = True
for _ in range(2000):
    p_true, q = F(rng.randint(0, 20), 20), F(rng.randint(0, 20), 20)
    forecast_brier_one_sided = (q - p_true) ** 2 + p_true * (1 - p_true)       # E[(q - Y)^2], q a number
    sampled_draw_sq_error = q + p_true - 2 * q * p_true                         # E[(X - Y)^2], X ~ Bernoulli(q) independent of Y
    ok &= (forecast_brier_one_sided - sampled_draw_sq_error == q * (q - 1))
check("G5 binary label: E[(X-Y)^2] for a SAMPLED draw X~Bern(q) exceeds E[(q-Y)^2] for the forecast number q by exactly q(1-q)", ok)

# ------------------------------------------------------------------------------------------------------------------
# C9. estimation rate versus K  (SYNTHETIC known distributions; exact formulas plus Monte Carlo)
# ------------------------------------------------------------------------------------------------------------------
mc = np.random.default_rng(20261127)
REPS = 3000
rows = []
ok_brier = ok_rps = ok_rps_cap = ok_brier_cap = ok_chi = True
chi_rows = []
kl_ratios = []
for K in (2, 4, 8, 16, 32, 64):
    p = mc.dirichlet(np.full(K, 5.0))
    Fcdf = np.cumsum(p)[:-1]
    for N in (64, 256, 1024):
        counts = mc.multinomial(N, p, size=REPS)
        qhat = counts / N
        brier_ex = ((qhat - p) ** 2).sum(axis=1)
        exact_brier = (1 - (p ** 2).sum()) / N
        se = brier_ex.std(ddof=1) / math.sqrt(REPS)
        ok_brier &= abs(brier_ex.mean() - exact_brier) < 4 * se + 1e-12
        ok_brier_cap &= exact_brier <= 1 / N + 1e-15
        chi = ((qhat - p) ** 2 / p).sum(axis=1)
        se_chi = chi.std(ddof=1) / math.sqrt(REPS)
        ok_chi &= abs(chi.mean() - (K - 1) / N) < 4 * se_chi + 1e-12
        chi_rows.append((K, N, chi.mean(), (K - 1) / N))
        cdf_hat = np.cumsum(qhat, axis=1)[:, :-1]
        rps_ex = ((cdf_hat - Fcdf) ** 2).sum(axis=1) / (K - 1)
        exact_rps = (Fcdf * (1 - Fcdf)).sum() / (N * (K - 1))
        se2 = rps_ex.std(ddof=1) / math.sqrt(REPS)
        ok_rps &= abs(rps_ex.mean() - exact_rps) < 4 * se2 + 1e-12
        ok_rps_cap &= exact_rps <= 1 / (4 * N) + 1e-15
        if N >= 16 * K:
            qs = (counts + 1.0) / (N + K)
            kl = (p * (np.log(p) - np.log(qs))).sum(axis=1)
            kl_ratios.append((K, N, kl.mean() / ((K - 1) / (2 * N))))
        rows.append((K, N, exact_brier, exact_rps, 1 / N, 1 / (4 * N)))
check("C9a Brier excess ||qhat-p||^2 has mean (1-||p||^2)/N within 4 standard errors, for K=2..64 (Exp 36 Lean: empirical_l2_rate)", ok_brier)
check("C9b ...and that mean is at most 1/N for every K (dimension-free; a bound of order K/N is valid but loose for this metric)", ok_brier_cap)
check("C9c normalized RPS excess has mean sum_t F(t)(1-F(t))/(N(K-1)) within 4 standard errors, for K=2..64", ok_rps)
check("C9d ...and that mean is at most 1/(4N) for every K (dimension-free for normalized RPS)", ok_rps_cap)
ratio_ok = all(0.7 <= r <= 1.5 for (_, _, r) in kl_ratios)
check("C9e cross-entropy excess KL(p||add-one qhat) is about (K-1)/(2N) when N >= 16K and p is not tiny (grows with K, unlike C9a to C9d)",
      ratio_ok and len(kl_ratios) >= 8, f"ratios {min(r for *_, r in kl_ratios):.2f}..{max(r for *_, r in kl_ratios):.2f} over {len(kl_ratios)} cells")
# (The first version of this growth test compared cells with different N and failed; the claim was right and the check was wrong. The 4-standard-error
#  condition had passed in all 18 cells, largest |z| = 2.84. The growth test below compares K at a fixed N = 256.)
chi256 = [(K, m) for (K, N, m, _) in chi_rows if N == 256]
grows = all(b[1] > a[1] for a, b in zip(chi256, chi256[1:])) and chi256[-1][1] > 30 * chi256[0][1]
check("C9f chi-squared excess sum_j (qhat_j - p_j)^2 / p_j has mean (K-1)/N within 4 standard errors for K=2..64 (exact: E = sum_j (1-p_j)/N), and at N=256 it grows "
      "roughly linearly in K, 63-fold from K=2 to K=64 (the O(K/N) label is right for this metric)", ok_chi and grows,
      f"N=256 means: " + ", ".join(f"K={K}: {m:.4f}" for K, m in chi256))
for K, N, b, rps, ib, irps in rows:
    if N == 256 and K in (2, 8, 64):
        print(f"        K={K:2d} N={N}: exact Brier excess {b:.6f} (bound {ib:.6f}) | exact normalized RPS excess {rps:.6f} (bound {irps:.6f})")

print()
if FAILS:
    print(f"FAILED {len(FAILS)} claim(s): " + "; ".join(FAILS))
    sys.exit(1)
print("ALL CLAIMS PASS")
