"""Seeded numerical checks of the proper-score (Murphy/Broecker) decomposition used in appendix `app:calibration` (28 Nov 2026).

Claim checked (status: numerically checked, Lean pending, for cross-entropy and the ranked probability score; the scalar squared-error
and multiclass Brier forms are Lean-checked in lean/Point2/Murphy.lean):

    E S(q, Y) = E S(pbar, Y) - E d(pbar, p_q) + E d(q, p_q)           (uncertainty - resolution + calibration error)

for a finite joint law of (forecast q, outcome Y), where p_q is the law of Y given the forecast VALUE q, pbar the marginal law of Y and
d(a, b) = E_{Y~b}[S(a, Y)] - E_{Y~b}[S(b, Y)]. The identity is algebraic (linearity of expectation and the tower property) for ANY score;
propriety only gives d >= 0 and, when strict, d(a, b) = 0 iff a = b.

Exact rational arithmetic (fractions.Fraction) for the Brier score and the normalized RPS; floats with a named tolerance for cross-entropy.
Every statement claimed false is tested as a failing case. Seed fixed; output is identical on a second run; exit code is nonzero on any FAIL.
Run: python3 proper_score_decomposition_numerics_28Nov2026.py
"""
import math
import random
import sys
from fractions import Fraction as F

SEED = 20261128
rng = random.Random(SEED)
N_TRIALS = 300
TOL_CE = 1e-12
results = []


def record(name, ok, detail):
    results.append((name, bool(ok), detail))


# ---------------------------------------------------------------- scores (loss orientation, lower is better)
def brier(q, y):
    return sum((qk - (1 if k == y else 0)) ** 2 for k, qk in enumerate(q))


def rps(q, y):
    K = len(q)
    cum, total = F(0), F(0)
    for j in range(K - 1):
        cum += q[j]
        total += (cum - (1 if y <= j else 0)) ** 2
    return total / (K - 1)


def ce(q, y):
    return -math.log(float(q[y]))


SCORES = {"brier": (brier, True), "rps": (rps, True), "ce": (ce, False)}  # (function, exact?)


def e(S, a, b):
    """expected score of forecast a when Y ~ b"""
    return sum(bk * S(a, k) for k, bk in enumerate(b) if bk != 0)


def d(S, a, b):
    return e(S, a, b) - e(S, b, b)


# ---------------------------------------------------------------- random finite joint laws
def rand_dist(K, positive):
    while True:
        w = [F(rng.randint(1 if positive else 0, 9)) for _ in range(K)]
        s = sum(w)
        if s > 0:
            return [x / s for x in w]


def rand_cells(positive, calibrated=False, shared=True):
    """cells (w, q, pi): weight, forecast vector, outcome law. Forecast values repeat across cells when `shared`."""
    K = rng.randint(2, 5)
    m = rng.randint(2, 7)
    pool = [rand_dist(K, positive) for _ in range(rng.randint(1, 4) if shared else m)]
    raw = [F(rng.randint(1, 9)) for _ in range(m)]
    tot = sum(raw)
    cells = []
    for g in range(m):
        q = pool[rng.randrange(len(pool))] if shared else pool[g]
        pi = list(q) if calibrated else rand_dist(K, positive)
        cells.append((raw[g] / tot, q, pi))
    return K, cells


def decompose(S, cells, key):
    """returns (lhs, unc, res, cal) with p_key = law of Y given the value of key(cell)"""
    K = len(cells[0][1])
    pbar = [sum(w * pi[y] for w, _, pi in cells) for y in range(K)]
    groups = {}
    for idx, (w, q, pi) in enumerate(cells):
        groups.setdefault(key(idx, q), []).append((w, q, pi))
    lhs = sum(w * e(S, q, pi) for w, q, pi in cells)
    unc = e(S, pbar, pbar)
    res = cal = 0
    for members in groups.values():
        W = sum(w for w, _, _ in members)
        pq = [sum(w * pi[y] for w, _, pi in members) / W for y in range(K)]
        q = members[0][1]  # the forecast value (all members share it when key is the forecast value)
        res += W * d(S, pbar, pq)
        cal += W * d(S, q, pq)
    return lhs, unc, res, cal, pbar, groups


def close(a, b, exact):
    return a == b if exact else abs(float(a) - float(b)) <= TOL_CE


def less_eq(a, b, exact):
    return a <= b if exact else float(a) <= float(b) + TOL_CE


by_forecast = lambda idx, q: tuple(q)
by_cell = lambda idx, q: idx

# ---------------------------------------------------------------- (1) the identity, three scores
for name, (S, exact) in SCORES.items():
    ok, n = True, 0
    for _ in range(N_TRIALS):
        K, cells = rand_cells(positive=True)
        lhs, unc, res, cal, _, _ = decompose(S, cells, by_forecast)
        ok &= close(lhs, unc - res + cal, exact)
        # the identity also holds for the finer key (cell id), where the terms change but the sum does not
        lhs2, unc2, res2, cal2, _, _ = decompose(S, cells, by_cell)
        ok &= close(lhs2, unc2 - res2 + cal2, exact) and close(unc, unc2, exact)
        n += 1
    record(f"(1) E S = UNC - RES + CAL, score={name}, forecast-value key and cell key", ok,
           f"{n} random laws, {'exact fractions' if exact else 'tol 1e-12'}")

# ---------------------------------------------------------------- (2) propriety: d >= 0, d = 0 iff a = b; closed forms
for name, (S, exact) in SCORES.items():
    ok, zero_ok, strict_ok, n = True, True, True, 0
    for _ in range(N_TRIALS):
        K = rng.randint(2, 5)
        a, b = rand_dist(K, True), rand_dist(K, True)
        dab = d(S, a, b)
        ok &= (dab >= 0) if exact else (float(dab) >= -TOL_CE)
        zero_ok &= close(d(S, b, b), 0, exact)
        if a != b:
            strict_ok &= (dab > 0) if exact else (float(dab) > 0)
        n += 1
    record(f"(2) d >= 0, d(b,b) = 0, d(a,b) > 0 for a != b, score={name}", ok and zero_ok and strict_ok, f"{n} pairs")

ok = True
for _ in range(N_TRIALS):
    K = rng.randint(2, 6)
    a, b = rand_dist(K, True), rand_dist(K, True)
    ok &= d(brier, a, b) == sum((x - y) ** 2 for x, y in zip(a, b))
    cumA, cumB, t = F(0), F(0), F(0)
    for j in range(K - 1):
        cumA += a[j]
        cumB += b[j]
        t += (cumA - cumB) ** 2
    ok &= d(rps, a, b) == t / (K - 1)
    ok &= abs(d(ce, a, b) - sum(float(bk) * math.log(float(bk) / float(ak)) for ak, bk in zip(a, b))) <= TOL_CE
record("(3) closed forms: d_Brier = squared L2, d_RPS = mean squared CDF gap over K-1 thresholds, d_CE = KL(b||a)", ok,
       f"{N_TRIALS} pairs")

# ---------------------------------------------------------------- (4) calibration error is zero iff calibrated; resolution >= 0
def is_calibrated(groups, K):
    """exact test (Fractions): every forecast value q equals the law of Y among the cells that issued it"""
    for members in groups.values():
        W = sum(w for w, _, _ in members)
        pq = [sum(w * pi[k] for w, _, pi in members) / W for k in range(K)]
        if list(members[0][1]) != pq:
            return False
    return True


for name, (S, exact) in SCORES.items():
    ok_cal, ok_miscal, ok_res, n, n_miscal = True, True, True, 0, 0
    for _ in range(N_TRIALS):
        K, cells = rand_cells(positive=True, calibrated=True)
        _, unc, res, cal, _, groups = decompose(S, cells, by_forecast)
        ok_cal &= is_calibrated(groups, K) and close(cal, 0, exact)
        ok_res &= (res >= 0) if exact else (float(res) >= -TOL_CE)
        K, cells = rand_cells(positive=True)  # generic cells: miscalibrated unless the law happens to match
        _, unc, res, cal, _, groups = decompose(S, cells, by_forecast)
        if not is_calibrated(groups, K):
            n_miscal += 1
            ok_miscal &= (cal > 0) if exact else (float(cal) > 0)
        ok_res &= (res >= 0) if exact else (float(res) >= -TOL_CE)
        n += 1
    record(f"(4) CAL = 0 when pi = q (calibrated), CAL > 0 when not, RES >= 0, score={name}",
           ok_cal and ok_miscal and ok_res and n_miscal > 0, f"{n} laws each, {n_miscal} miscalibrated")

# ---------------------------------------------------------------- (5) Jensen: the finer key never lowers the third term; claimed-false: it never changes it
for name, (S, exact) in SCORES.items():
    ok_ge, changed = True, 0
    for _ in range(N_TRIALS):
        K, cells = rand_cells(positive=True)
        _, _, _, cal_f, _, _ = decompose(S, cells, by_forecast)
        _, _, _, cal_c, _, _ = decompose(S, cells, by_cell)
        ok_ge &= less_eq(cal_f, cal_c, exact)
        changed += (cal_c > cal_f) if exact else (float(cal_c) > float(cal_f) + TOL_CE)
    record(f"(5) CAL(forecast-value key) <= CAL(cell key), score={name}", ok_ge, f"{N_TRIALS} laws")
    record(f"(5f) claimed-false: 'CAL does not depend on the conditioning key' FAILS, score={name}", changed > 0,
           f"CAL(cell key) > CAL(forecast key) on {changed} of {N_TRIALS} laws")

# ---------------------------------------------------------------- (6) a restricted recalibration family removes at most the third term
for name, (S, exact) in SCORES.items():
    ok, strict_seen, n = True, 0, 0
    for _ in range(N_TRIALS):
        K, cells = rand_cells(positive=True)
        lhs, unc, res, cal, pbar, groups = decompose(S, cells, by_forecast)
        # a random map g sends each forecast value to some forecast
        gmap = {key: rand_dist(K, True) for key in groups}
        r_cal = sum(w * e(S, gmap[tuple(q)], pi) for w, q, pi in cells)
        ok &= less_eq(lhs - r_cal, cal, exact)               # R_raw - R_g <= CAL
        # the best map is q -> p_q and attains the bound exactly
        best = {key: [sum(w * pi[k] for w, _, pi in m) / sum(w for w, _, _ in m) for k in range(K)] for key, m in groups.items()}
        r_best = sum(w * e(S, best[tuple(q)], pi) for w, q, pi in cells)
        ok &= close(lhs - r_best, cal, exact)
        strict_seen += (lhs - r_cal < cal) if exact else (float(lhs - r_cal) < float(cal) - TOL_CE)
        n += 1
    record(f"(6) R_raw - R_g <= CAL for every map g, equality at g(q) = p_q, score={name}", ok and strict_seen > 0,
           f"{n} laws; strict inequality for a random map on {strict_seen}")

# ---------------------------------------------------------------- scalar squared-error forms (Lean-checked; numerics here are a cross-check)
def rand_scalar_law():
    m = rng.randint(3, 8)
    pool = [F(rng.randint(-4, 4)) for _ in range(rng.randint(2, 4))]
    raw = [F(rng.randint(1, 9)) for _ in range(m)]
    tot = sum(raw)
    w = [x / tot for x in raw]
    s = [pool[rng.randrange(len(pool))] for _ in range(m)]
    y = [F(rng.randint(-5, 5)) for _ in range(m)]
    return w, s, y


def Ex(w, f):
    return sum(a * b for a, b in zip(w, f))


def cov(w, f, g):
    return Ex(w, [a * b for a, b in zip(f, g)]) - Ex(w, f) * Ex(w, g)


def condE(w, s, y):
    groups = {}
    for wi, si, yi in zip(w, s, y):
        a = groups.setdefault(si, [F(0), F(0)])
        a[0] += wi
        a[1] += wi * yi
    return [groups[si][1] / groups[si][0] for si in s]


ok_m, ok_r, n = True, True, 0
for _ in range(N_TRIALS):
    w, s, y = rand_scalar_law()
    g = condE(w, s, y)
    vS, vY = cov(w, s, s), cov(w, y, y)
    mse = Ex(w, [(yi - si) ** 2 for yi, si in zip(y, s)])
    ok_m &= mse == vY - cov(w, g, g) + Ex(w, [(gi - si) ** 2 for gi, si in zip(g, s)])
    if vS == 0 or vY == 0:
        continue
    b = cov(w, s, y) / vS
    a = Ex(w, y) - b * Ex(w, s)
    r2 = cov(w, s, y) ** 2 / (vS * vY)
    ok_r &= r2 * vY == cov(w, g, g) - Ex(w, [(gi - (a + b * si)) ** 2 for gi, si in zip(g, s)])
    n += 1
record("(7) scalar Murphy: E(Y-S)^2 = Var Y - Var g(S) + E(g(S)-S)^2 (cf. Lean `murphy_scalar`)", ok_m, f"{N_TRIALS} laws, exact")
record("(8) r^2 Var Y = Var g(S) - E(g(S) - S')^2 with S' the best affine refit (cf. Lean `r_sq_resolution`)", ok_r, f"{n} laws, exact")

# claimed-false: 'r^2 Var Y equals the resolution Var g(S)' (correlation sees all of the resolution)
w = [F(1, 3)] * 3
s = [F(-1), F(0), F(1)]
y = [x * x for x in s]  # Y = S^2: a perfectly informative but non-monotone forecast
g = condE(w, s, y)
r2VY = cov(w, s, y) ** 2 / cov(w, s, s)
record("(8f) claimed-false: 'r^2 Var Y = Var g(S)' FAILS (Y = S^2 gives r = 0 but Var g(S) = Var Y > 0)",
       r2VY == 0 and cov(w, g, g) == cov(w, y, y) and cov(w, y, y) > 0,
       f"r^2 Var Y = {r2VY}, Var g(S) = {cov(w, g, g)}")

# affine relabelling: r^2 and the calibration curve are unchanged, the calibration error E(g(S)-S)^2 is not
ok_inv, ok_chg, n = True, 0, 0
for _ in range(N_TRIALS):
    w, s, y = rand_scalar_law()
    vS, vY = cov(w, s, s), cov(w, y, y)
    if vS == 0 or vY == 0:
        continue
    a2, b2 = F(rng.randint(-5, 5)), F(rng.choice([-1, 1]) * rng.randint(1, 5), rng.randint(1, 3))
    s2 = [a2 + b2 * si for si in s]
    g, g2 = condE(w, s, y), condE(w, s2, y)
    r2 = cov(w, s, y) ** 2 / (vS * vY)
    r2b = cov(w, s2, y) ** 2 / (cov(w, s2, s2) * vY)
    ok_inv &= (r2 == r2b) and (g == g2)
    c1 = Ex(w, [(gi - si) ** 2 for gi, si in zip(g, s)])
    c2 = Ex(w, [(gi - si) ** 2 for gi, si in zip(g2, s2)])
    ok_chg += (c1 != c2)
    n += 1
record("(9) an affine relabelling of S leaves r^2 and E[Y|S] unchanged", ok_inv, f"{n} laws, exact")
record("(9f) claimed-false: 'the calibration error E(g(S)-S)^2 is unchanged by an affine relabelling' FAILS", ok_chg > 0,
       f"changed on {ok_chg} of {n} laws")

# claimed-false: Pearson r is unchanged by every strictly increasing relabelling (only affine ones are safe)
w = [F(1, 3)] * 3
s = [F(-1), F(0), F(2)]
y = [F(0), F(1), F(1)]
s3 = [x ** 3 for x in s]
r2a = cov(w, s, y) ** 2 / (cov(w, s, s) * cov(w, y, y))
r2b = cov(w, s3, y) ** 2 / (cov(w, s3, s3) * cov(w, y, y))
record("(9g) claimed-false: 'Pearson r is invariant to every strictly increasing relabelling' FAILS (S -> S^3)", r2a != r2b,
       f"r^2 = {r2a} vs {r2b}")

# ---------------------------------------------------------------- (10) a finer conditioning key (added 28 Nov 2026 after review)
# Claim: for a proper score, replacing the forecast value by a finer key (here the cell, i.e. the item) leaves UNC - RES + CAL unchanged
# and ADDS THE SAME NONNEGATIVE AMOUNT E H(p_q) - E H(p_key), with H(b) = E_{Y~b} S(b, Y), to the resolution and to the calibration error.
# It does not move calibration error into resolution. These checks run after all earlier ones, so earlier random draws are unchanged.
ok_same, ok_nonneg, fell, opposite, n_laws = True, True, 0, 0, 0
for name, (S, exact) in SCORES.items():
    for _ in range(N_TRIALS):
        K, cells = rand_cells(positive=True)
        _, unc1, res1, cal1, _, _ = decompose(S, cells, by_forecast)
        _, unc2, res2, cal2, _, _ = decompose(S, cells, by_cell)
        dres, dcal = res2 - res1, cal2 - cal1
        ok_same &= close(dres, dcal, exact)
        ok_nonneg &= less_eq(0, dres, exact) and less_eq(0, dcal, exact)
        tiny = 0 if exact else TOL_CE
        fell += float(dcal) < -tiny
        opposite += (float(dres) > tiny and float(dcal) < -tiny) or (float(dres) < -tiny and float(dcal) > tiny)
        n_laws += 1
record("(10) finer key (cell instead of forecast value): resolution and calibration error both rise by the same amount >= 0, scores Brier, RPS, CE",
       ok_same and ok_nonneg, f"{n_laws} random laws, exact fractions for Brier and RPS, tol 1e-12 for CE")

# worked example (unnormalized Brier = the sum of squared errors over categories, as defined above)
q_const = [F(1, 2), F(1, 2)]
cells_ex = [(F(1, 2), q_const, [F(1), F(0)]), (F(1, 2), q_const, [F(0), F(1)])]  # two equally likely items, Y = 0 and Y = 1
_, u_a, r_a, c_a, _, _ = decompose(brier, cells_ex, by_forecast)
_, u_b, r_b, c_b, _, _ = decompose(brier, cells_ex, by_cell)
record("(10w) worked example: constant forecast (1/2, 1/2) on two items with Y = 0 and Y = 1 has (UNC, RES, CAL) = (1/2, 0, 0) by forecast value and (1/2, 1/2, 1/2) by item",
       (u_a, r_a, c_a) == (F(1, 2), 0, 0) and (u_b, r_b, c_b) == (F(1, 2), F(1, 2), F(1, 2)),
       f"by forecast value ({u_a}, {r_a}, {c_a}); by item ({u_b}, {r_b}, {c_b})")
record("(10f) claimed-false: 'a finer key moves calibration error into resolution' FAILS (calibration error never fell; the two terms never moved in opposite directions)",
       fell == 0 and opposite == 0, f"calibration error fell on {fell} of {n_laws} laws; opposite moves on {opposite}")

bad = 0
for name, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  [{detail}]")
    bad += (not ok)
print(f"seed={SEED}; {len(results) - bad}/{len(results)} PASS")
sys.exit(1 if bad else 0)
