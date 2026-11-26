#!/usr/bin/env python3
"""Checks for SQUARED_DIFFERENCE_VS_DISTRIBUTION_11-26-2026.md (experiment 35).

Page three of Brando's 11-26-2026 derivation pages asks to show that optimizing a
difference of random variables (squared difference, mean difference) does not
yield distribution matching.  Each check below prints PASS or FAIL with the
numbers the note quotes.  Exact rational arithmetic (fractions.Fraction) is used
wherever no square root is needed; floats are used for standard deviations and
for the linear program in check 6.  No human ratings, no randomness except the
seeded numpy generators named in the checks.  Run:  python3 -I <this file>

Notation (the PS on page one): X~ = X - mu_X is the centered variable and
Z_X = X~ / sigma_X is the standardized variable.
"""
from __future__ import annotations

import itertools
import math
import sys
from fractions import Fraction as F

import numpy as np
from scipy.optimize import linprog

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


# ---------------------------------------------------------------- helpers
def mean(pmf, support):
    return sum(w * x for w, x in zip(pmf, support))


def var(pmf, support):
    m = mean(pmf, support)
    return sum(w * (x - m) ** 2 for w, x in zip(pmf, support))


def brier_dist(p, q):
    """Squared Euclidean distance sum_a (p(a) - q(a))^2 = D_B(p || q)."""
    return sum((a - b) ** 2 for a, b in zip(p, q))


def tvd(p, q):
    return sum(abs(a - b) for a, b in zip(p, q)) / 2


def random_joint(rng, nx, ny, zero_prob=0.3):
    """Random joint pmf on an nx-by-ny grid with exact rational weights."""
    w = rng.integers(0, 6, size=(nx, ny))
    w = np.where(rng.random((nx, ny)) < zero_prob, 0, w)
    if w.sum() == 0:
        w[0, 0] = 1
    total = int(w.sum())
    return [[F(int(w[i, j]), total) for j in range(ny)] for i in range(nx)]


def joint_moments(J, xs, ys):
    px = [sum(J[i][j] for j in range(len(ys))) for i in range(len(xs))]
    py = [sum(J[i][j] for i in range(len(xs))) for j in range(len(ys))]
    mx, my = mean(px, xs), mean(py, ys)
    vx, vy = var(px, xs), var(py, ys)
    exy = sum(J[i][j] * xs[i] * ys[j] for i in range(len(xs)) for j in range(len(ys)))
    cov = exy - mx * my
    sq = sum(J[i][j] * (xs[i] - ys[j]) ** 2 for i in range(len(xs)) for j in range(len(ys)))
    return px, py, mx, my, vx, vy, cov, sq


# ------------------------------------------------ check 1: moment expansion
def check1():
    rng = np.random.default_rng(1)
    xs = [F(v) for v in range(5)]
    ys = [F(v) for v in range(5)]
    worst_exact = F(0)
    worst_float = 0.0
    n = 0
    for _ in range(300):
        J = random_joint(rng, 5, 5)
        px, py, mx, my, vx, vy, cov, sq = joint_moments(J, xs, ys)
        # exact: E[(X-Y)^2] = (mx-my)^2 + vx + vy - 2 cov
        rhs = (mx - my) ** 2 + vx + vy - 2 * cov
        worst_exact = max(worst_exact, abs(sq - rhs))
        # float: (mx-my)^2 + (sx-sy)^2 + 2 sx sy (1 - rho)  (needs vx, vy > 0)
        if vx > 0 and vy > 0:
            sx, sy = math.sqrt(vx), math.sqrt(vy)
            rho = float(cov) / (sx * sy)
            rhs2 = float((mx - my) ** 2) + (sx - sy) ** 2 + 2 * sx * sy * (1 - rho)
            worst_float = max(worst_float, abs(float(sq) - rhs2))
            n += 1
    ok = worst_exact == 0 and worst_float < 1e-12
    record(
        "1 moment expansion",
        ok,
        f"300 random joint pmfs on a 5x5 grid: E[(X-Y)^2] = (mu_X-mu_Y)^2 + var_X + var_Y - 2cov "
        f"holds exactly (max exact gap {worst_exact}); the three-term form "
        f"(mu gap)^2 + (sigma gap)^2 + 2 sigma_X sigma_Y (1-corr) agrees to {worst_float:.1e} on {n} pairs",
    )


# ------------------------- check 2: zero squared difference iff X = Y a.s.
def check2():
    rng = np.random.default_rng(2)
    xs = [F(v) for v in range(5)]
    ok = True
    n_diag = n_off = 0
    for _ in range(300):
        J = random_joint(rng, 5, 5)
        px, py, mx, my, vx, vy, cov, sq = joint_moments(J, xs, xs)
        off_mass = sum(J[i][j] for i in range(5) for j in range(5) if i != j)
        if sq == 0:
            ok &= off_mass == 0 and px == py
            n_diag += 1
        else:
            ok &= off_mass > 0
            n_off += 1
    # force the diagonal case so it is exercised
    for _ in range(100):
        w = rng.integers(0, 6, size=5)
        if w.sum() == 0:
            w[0] = 1
        p = [F(int(v), int(w.sum())) for v in w]
        J = [[p[i] if i == j else F(0) for j in range(5)] for i in range(5)]
        px, py, *_rest, sq = joint_moments(J, xs, xs)
        ok &= sq == 0 and px == py
        n_diag += 1
    record(
        "2 zero squared difference forces X=Y",
        ok,
        f"{n_diag} couplings with E[(X-Y)^2]=0 all had no off-diagonal mass and equal marginals; "
        f"{n_off} couplings with off-diagonal mass all had E[(X-Y)^2]>0. "
        "With a free coupling, a zero squared difference does imply equal laws",
    )


# ------------------------- check 3: the mean-matching pair of page three
P6 = [F(1, 2), F(0), F(0), F(0), F(0), F(1, 2)]
Q6 = [F(0), F(0), F(1, 2), F(1, 2), F(0), F(0)]
SUP6 = [F(v) for v in range(6)]


def check3():
    mp, mq = mean(P6, SUP6), mean(Q6, SUP6)
    vp, vq = var(P6, SUP6), var(Q6, SUP6)
    b, t = brier_dist(P6, Q6), tvd(P6, Q6)
    ok = (mp == mq == F(5, 2)) and P6 != Q6 and b == 1 and t == 1 and sum(P6) == 1 and sum(Q6) == 1
    record(
        "3 equal means, different laws",
        ok,
        f"p=(1/2,0,0,0,0,1/2), q=(0,0,1/2,1/2,0,0) on {{0..5}}: mean {mp} = {mq}, so the mean difference "
        f"is 0; variances {vp} and {vq}; Brier distance sum_a (p-q)^2 = {b}; total variation distance = {t}",
    )


# ------------------------------------- check 4: dimension of mean-matching set
def rank_exact(rows):
    M = [list(map(F, r)) for r in rows]
    r = 0
    ncols = len(M[0])
    for c in range(ncols):
        piv = None
        for i in range(r, len(M)):
            if M[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c] / M[r][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        r += 1
    return r


def check4():
    ok = True
    dims = {}
    for K in range(2, 9):
        A = [[1] * K, list(range(K))]
        rk = rank_exact(A)
        dims[K] = K - rk
        ok &= rk == 2 and dims[K] == K - 2
    # moments 0..j (normalisation plus j moments) leave K-1-j free directions: K = 6
    K = 6
    free = {}
    for j in range(1, K):
        A = [[F(x) ** order for x in range(K)] for order in range(j + 1)]
        free[j] = K - rank_exact(A)
    ok &= free == {1: 4, 2: 3, 3: 2, 4: 1, 5: 0}
    # a full-support pmf with mean m moves along a null vector and keeps mean and total mass
    p = [F(1, 6)] * 6
    v = [F(1), F(-2), F(1), F(0), F(0), F(0)]  # sums to 0, first moment 0 - 2 + 2 = 0
    assert sum(v) == 0 and sum(i * v[i] for i in range(6)) == 0
    t = F(1, 100)
    q = [a + t * b for a, b in zip(p, v)]
    ok &= sum(q) == 1 and min(q) > 0 and mean(q, SUP6) == mean(p, SUP6) and q != p and brier_dist(p, q) > 0
    record(
        "4 free directions after matching moments",
        ok,
        f"pmfs on K outcomes with a given mean form a set of dimension K-2 (K=2..8: {dims}); for K=6 "
        f"matching the first j moments plus normalisation leaves {free} free directions (j=5 pins the pmf); "
        f"moving the uniform pmf by t*v keeps mean {mean(q, SUP6)} and changes it by Brier distance {brier_dist(p, q)}",
    )


# -------- check 5: standardized difference zero, different laws (Y = 2X)
def check5():
    xs = [F(v) for v in range(5)]
    px = [F(1, 4), F(1, 2), F(1, 4), F(0), F(0)]
    ys = [F(2 * v) for v in range(5)]  # Y = 2X lives on {0,2,4,6,8}; fold onto the grid {0..4} only where it lands
    # law of Y on {0,...,4}: Y = 2X with X in {0,1,2} gives {0,2,4}
    py = [F(1, 4), F(0), F(1, 2), F(0), F(1, 4)]
    sup = [F(v) for v in range(5)]
    mx, my = mean(px, sup), mean(py, sup)
    vx, vy = var(px, sup), var(py, sup)
    # coupling Y = 2X
    J = [[F(0)] * 5 for _ in range(5)]
    for i in range(3):
        J[i][2 * i] = px[i]
    _, _, _, _, _, _, cov, sq = joint_moments(J, sup, sup)
    corr_sq = cov**2 / (vx * vy)
    # exact equality of the standardized values: (x-mx)^2/vx == (2x-my)^2/vy and same sign (2>0)
    z_equal = all((x - mx) ** 2 * vy == (2 * x - my) ** 2 * vx for x in sup[:3])
    b, t = brier_dist(px, py), tvd(px, py)
    ok = z_equal and corr_sq == 1 and cov > 0 and px != py and b == F(3, 8) and t == F(1, 2)
    record(
        "5 correlation one, different laws",
        ok,
        f"X~(1/4,1/2,1/4,0,0), Y=2X~(1/4,0,1/2,0,1/4): means {mx}, {my}; variances {vx}, {vy}; corr^2 = {corr_sq}; "
        f"Z_X = Z_Y exactly (E[(Z_X-Z_Y)^2] = 2(1-corr) = 0) yet Brier distance = {b} and total variation = {t}",
    )


# ------------- check 6: minimum over couplings (squared Wasserstein distance)
def monotone_coupling_cost(p, q, sup):
    """Exact cost of the north-west-corner (monotone) coupling for cost (x-y)^2."""
    i = j = 0
    a, b = list(p), list(q)
    cost = F(0)
    exy = F(0)
    while i < len(a) and j < len(b):
        m = min(a[i], b[j])
        cost += m * (sup[i] - sup[j]) ** 2
        exy += m * sup[i] * sup[j]
        a[i] -= m
        b[j] -= m
        if a[i] == 0:
            i += 1
        if b[j] == 0:
            j += 1
    return cost, exy


def lp_min_coupling(p, q, sup):
    K = len(p)
    c = [float((sup[i] - sup[j]) ** 2) for i in range(K) for j in range(K)]
    A_eq, b_eq = [], []
    for i in range(K):
        row = [0.0] * (K * K)
        for j in range(K):
            row[i * K + j] = 1.0
        A_eq.append(row)
        b_eq.append(float(p[i]))
    for j in range(K):
        row = [0.0] * (K * K)
        for i in range(K):
            row[i * K + j] = 1.0
        A_eq.append(row)
        b_eq.append(float(q[j]))
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return res.fun


def decompose(p, q, sup):
    """Minimum of E[(X-Y)^2] over couplings, from the exact monotone coupling and from a linear program,
    with its split into (mean gap)^2, (sd gap)^2 and the shape term 2 sd_p sd_q (1 - rho_max)."""
    cost, exy = monotone_coupling_cost(p, q, sup)
    lp = lp_min_coupling(p, q, sup)
    mp, mq = mean(p, sup), mean(q, sup)
    vp, vq = var(p, sup), var(q, sup)
    sp, sq_ = math.sqrt(float(vp)), math.sqrt(float(vq))
    rho_max = float(exy - mp * mq) / (sp * sq_)
    mean_gap2 = float((mp - mq) ** 2)
    sd_gap2 = (sp - sq_) ** 2
    shape = 2 * sp * sq_ * (1 - rho_max)
    return cost, lp, rho_max, mean_gap2, sd_gap2, shape


def check6():
    pairs = {
        "A": (P6, Q6, SUP6),  # the pair of check 3: equal means, different variances, two-point laws
        "B": ([F(1, 4), F(0), F(1, 2), F(0), F(1, 4)], [F(0), F(1, 2), F(0), F(1, 2), F(0)], [F(v) for v in range(5)]),
        "C": (
            [F(1, 4), F(0), F(1, 2), F(0), F(1, 4)],
            [F(1, 8), F(3, 8), F(1, 8), F(1, 8), F(1, 4)],
            [F(v) for v in range(5)],
        ),
    }
    ok = True
    parts = []
    for name, (p, q, sup) in pairs.items():
        cost, lp, rho, mg, sg, sh = decompose(p, q, sup)
        ok &= abs(float(cost) - lp) < 1e-9 and abs(float(cost) - (mg + sg + sh)) < 1e-9 and cost > 0
        parts.append(
            f"pair {name}: min over couplings {cost} (LP {lp:.6f}) = mean gap^2 {mg:.4f} + sd gap^2 {sg:.4f} "
            f"+ shape term {sh:.4f} (rho_max {rho:.4f})"
        )
    # the equal-mean, equal-variance pair C has no location or scale gap, so its whole cost is shape
    cost_c, _, rho_c, mg_c, sg_c, sh_c = decompose(*pairs["C"])
    ok &= mg_c == 0.0 and abs(sg_c) < 1e-12 and rho_c < 1
    mp, vp = mean(pairs["C"][0], pairs["C"][2]), var(pairs["C"][0], pairs["C"][2])
    mq, vq = mean(pairs["C"][1], pairs["C"][2]), var(pairs["C"][1], pairs["C"][2])
    ok &= (mp, vp) == (mq, vq) == (F(2), F(2))
    # identical laws cost nothing, and the independent coupling of pair A is not a minimiser
    cost_pp, _ = monotone_coupling_cost(P6, P6, SUP6)
    prod_cost = sum(P6[i] * Q6[j] * (SUP6[i] - SUP6[j]) ** 2 for i in range(6) for j in range(6))
    ok &= cost_pp == 0 and abs(lp_min_coupling(P6, P6, SUP6)) < 1e-12 and prod_cost > decompose(*pairs["A"])[0]
    record(
        "6 minimum over couplings",
        ok,
        "; ".join(parts)
        + f". Pair C has equal mean {mp} and equal variance {vp} yet a positive minimum: shape alone costs {cost_c}. "
        f"Identical laws cost {cost_pp}; the independent coupling of pair A costs {prod_cost} > its minimum. "
        "A minimiser over couplings sees location, scale and shape; a coupling fixed by the data is not that minimiser",
    )


# ---------------------- check 7: fixed pairing identifies the conditional mean
def check7():
    rng = np.random.default_rng(7)
    ok = True
    worst = F(0)
    n_pert = 0
    xs = SUP6
    for _ in range(100):
        # joint pmf over (Z in {0,1,2}, X in {0..5}) with rational weights
        w = rng.integers(0, 5, size=(3, 6))
        w[:, 0] += 1  # keep every conditional well defined
        total = int(w.sum())
        J = [[F(int(w[z, x]), total) for x in range(6)] for z in range(3)]
        pz = [sum(row) for row in J]
        cm = [sum(J[z][x] * xs[x] for x in range(6)) / pz[z] for z in range(3)]
        cv = [sum(J[z][x] * (xs[x] - cm[z]) ** 2 for x in range(6)) / pz[z] for z in range(3)]
        f = [F(int(v), 4) for v in rng.integers(-4, 24, size=3)]
        lhs = sum(J[z][x] * (xs[x] - f[z]) ** 2 for z in range(3) for x in range(6))
        rhs = sum(pz[z] * cv[z] for z in range(3)) + sum(pz[z] * (cm[z] - f[z]) ** 2 for z in range(3))
        worst = max(worst, abs(lhs - rhs))
        best = sum(pz[z] * cv[z] for z in range(3))
        for _k in range(5):
            g = [cm[z] + F(int(v), 8) for z, v in enumerate(rng.integers(-5, 6, size=3))]
            if g != cm:
                risk = sum(J[z][x] * (xs[x] - g[z]) ** 2 for z in range(3) for x in range(6))
                ok &= risk > best
                n_pert += 1
    ok &= worst == 0
    # same optimal judge output for two different conditional rating laws, different Brier-optimal forecasts
    fstar_p = mean(P6, SUP6)
    fstar_q = mean(Q6, SUP6)
    ok &= fstar_p == fstar_q == F(5, 2) and P6 != Q6
    # expected squared error of the shared best point output 5/2 under each law (irreducible noise)
    mse_p = sum(w * (x - F(5, 2)) ** 2 for w, x in zip(P6, SUP6))
    mse_q = sum(w * (x - F(5, 2)) ** 2 for w, x in zip(Q6, SUP6))
    record(
        "7 fixed pairing identifies only the conditional mean",
        ok,
        f"E[(X-f(Z))^2] = E[Var(X|Z)] + E[(E[X|Z]-f(Z))^2] exactly on 100 random joint pmfs (max gap {worst}); "
        f"{n_pert} perturbations of the conditional mean all had strictly larger risk. The two laws of check 3 give "
        f"the same best point output {fstar_p} but irreducible squared error {mse_p} and {mse_q}; "
        "the point output cannot tell them apart, while the Brier-optimal forecasts p and q differ",
    )


# --------------- check 8: squared difference between independent draws
def check8():
    ok = True
    p = P6
    mp, vp = mean(p, SUP6), var(p, SUP6)
    matched = sum(p[i] * p[j] * (SUP6[i] - SUP6[j]) ** 2 for i in range(6) for j in range(6))
    ok &= matched == 2 * vp == F(25, 2)
    point = {}
    for y in range(6):
        point[y] = sum(p[i] * (SUP6[i] - y) ** 2 for i in range(6))
    best_y = min(point, key=lambda k: point[k])
    ok &= point[2] == point[3] == F(13, 2) and min(point.values()) == F(13, 2)
    # linear in q, so the minimum over all pmfs q on {0..5} is at a vertex (a point mass)
    rng = np.random.default_rng(8)
    worst_lin = F(0)
    for _ in range(200):
        w = rng.integers(0, 6, size=6)
        if w.sum() == 0:
            w[0] = 1
        q = [F(int(v), int(w.sum())) for v in w]
        val = sum(p[i] * q[j] * (SUP6[i] - SUP6[j]) ** 2 for i in range(6) for j in range(6))
        formula = vp + var(q, SUP6) + (mp - mean(q, SUP6)) ** 2
        worst_lin = max(worst_lin, abs(val - formula))
        ok &= val >= F(13, 2)
    ok &= worst_lin == 0
    record(
        "8 independent draws reward collapse",
        ok,
        f"p=(1/2)delta_0+(1/2)delta_5: E[(X-Y)^2] for independent X~p, Y~q equals var_p + var_q + (mu_p-mu_q)^2 "
        f"(max gap {worst_lin} on 200 random q); matched q=p scores {matched}; the best pmf on the integer grid is the "
        f"point mass at 2 or 3 and scores {point[best_y]}; on a continuous support the point mass at the mean 5/2 would score {vp}",
    )


# ------------------------------------------- check 9: Brier sees the difference
def check9():
    ok = True
    rng = np.random.default_rng(9)
    worst = F(0)
    for _ in range(200):
        K = 4
        wp = rng.integers(1, 9, size=K)
        wq = rng.integers(1, 9, size=K)
        p = [F(int(v), int(wp.sum())) for v in wp]
        q = [F(int(v), int(wq.sum())) for v in wq]

        def exp_brier(report, truth):
            return sum(truth[x] * sum((report[a] - (1 if a == x else 0)) ** 2 for a in range(K)) for x in range(K))

        gap = exp_brier(q, p) - exp_brier(p, p)
        worst = max(worst, abs(gap - brier_dist(p, q)))
        ok &= gap >= 0 and (gap == 0) == (p == q)
    ok &= worst == 0
    b = brier_dist(P6, Q6)
    ok &= b == 1 and mean(P6, SUP6) - mean(Q6, SUP6) == 0
    record(
        "9 excess expected Brier is the squared distance",
        ok,
        f"excess expected Brier of report q under truth p equals sum_a (p-q)^2 exactly on 200 random pairs (max gap {worst}); "
        f"for the pair of check 3 the mean difference is 0 while the Brier distance is {b}",
    )


# ------------------ check 10: three levels (raw, centered, standardized)
def check10():
    rng = np.random.default_rng(10)
    xs = [F(v) for v in range(5)]
    worst_c = F(0)
    worst_z = 0.0
    n = 0
    for _ in range(300):
        J = random_joint(rng, 5, 5)
        px, py, mx, my, vx, vy, cov, sq = joint_moments(J, xs, xs)
        # centered: E[(X~-Y~)^2] = var_X + var_Y - 2cov, exact
        cen = sum(J[i][j] * ((xs[i] - mx) - (xs[j] - my)) ** 2 for i in range(5) for j in range(5))
        worst_c = max(worst_c, abs(cen - (vx + vy - 2 * cov)))
        # raw = bias^2 + centered, exact
        worst_c = max(worst_c, abs(sq - ((mx - my) ** 2 + cen)))
        if vx > 0 and vy > 0:
            sx, sy = math.sqrt(vx), math.sqrt(vy)
            zz = sum(
                float(J[i][j]) * (float(xs[i] - mx) / sx - float(xs[j] - my) / sy) ** 2 for i in range(5) for j in range(5)
            )
            corr = float(cov) / (sx * sy)
            worst_z = max(worst_z, abs(zz - 2 * (1 - corr)))
            n += 1
    ok = worst_c == 0 and worst_z < 1e-12
    record(
        "10 raw, centered and standardized squared difference",
        ok,
        f"300 random joint pmfs: raw = (mu gap)^2 + centered and centered = var_X + var_Y - 2cov hold exactly "
        f"(max gap {worst_c}); standardized E[(Z_X-Z_Y)^2] = 2(1-corr) to {worst_z:.1e} on {n} pairs",
    )


if __name__ == "__main__":
    for fn in (check1, check2, check3, check4, check5, check6, check7, check8, check9, check10):
        fn()
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_pass} of {len(results)} checks PASS")
    sys.exit(0 if n_pass == len(results) else 1)
