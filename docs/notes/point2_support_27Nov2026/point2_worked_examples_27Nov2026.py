#!/usr/bin/env python3
"""Exact-arithmetic checks for the Point 2 note (27 Nov 2026).

Everything below uses fractions.Fraction, so the printed numbers are exact; the only floats are
the final decimal renderings and the cross-entropy (natural logs).  Seeded random checks use
seed 0.  Run:  python3 point2_worked_examples_27Nov2026.py

Conventions (same as experiments/36_*/metrics.py):
  one-hot Brier   B(q, y)   = sum_k (q_k - 1[y=k])^2
  RPS             R(q, y)   = sum_{t=0}^{K-2} (F_q(t) - 1[y <= t])^2     (normalized: R/(K-1))
  cross-entropy   CE(q, y)  = -ln q_y                                   (nats)
Ratings are integers 0..K-1 with K = 6.
"""
from fractions import Fraction as Fr
import math, random

K = 6

def cdf(q):
    out, s = [], Fr(0)
    for x in q:
        s += x
        out.append(s)
    return out

def mean(p):   return sum(Fr(k) * p[k] for k in range(K))
def var(p):    return sum((Fr(k) - mean(p)) ** 2 * p[k] for k in range(K))
def l2sq(a, b): return sum((x - y) ** 2 for x, y in zip(a, b))

def exp_brier(q, p):   # E_{Y~p} ||q - e_Y||^2 ,  = ||q-p||^2 + 1 - ||p||^2
    return sum(p[y] * sum((q[k] - (1 if k == y else 0)) ** 2 for k in range(K)) for y in range(K))

def exp_rps(q, p):
    Fq = cdf(q)
    tot = Fr(0)
    for y in range(K):
        tot += p[y] * sum((Fq[t] - (1 if y <= t else 0)) ** 2 for t in range(K - 1))
    return tot

def exp_ce(q, p):
    tot = 0.0
    for y in range(K):
        if p[y] > 0:
            if q[y] == 0:
                return math.inf
            tot += float(p[y]) * -math.log(float(q[y]))
    return tot

def smooth(q, eps=Fr(1, 10)):  # mix with uniform; used ONLY to give finite cross-entropies
    return [(1 - eps) * x + eps / K for x in q]

def fmt(x):
    if isinstance(x, float):
        return "inf" if math.isinf(x) else f"{x:.4f}"
    return f"{x} (= {float(x):.4f})" if x.denominator != 1 else f"{x}"

def pm(*pairs):
    p = [Fr(0)] * K
    for k, w in pairs:
        p[k] += Fr(w)
    assert sum(p) == 1
    return p

print("=" * 70)
print("PART 1. Single-item worked example (f): same mean, different spread, K=6")
A = pm((2, 1))                       # point mass at 2
B = pm((0, Fr(1, 2)), (4, Fr(1, 2))) # half on 0, half on 4
print("mean(A), mean(B) =", mean(A), mean(B), "| var(A), var(B) =", var(A), var(B))
print("cdf(A) =", [str(x) for x in cdf(A)[:-1]], " cdf(B) =", [str(x) for x in cdf(B)[:-1]])
print("Scalar mean-readout squared error E(Y-2)^2 : truth A =", sum(A[k] * (k - 2) ** 2 for k in range(K)),
      "; truth B =", sum(B[k] * (k - 2) ** 2 for k in range(K)), " (identical for forecast A and forecast B, both read out as 2)")
print()
print("rows = forecast, columns = truth; entries are expected score under the truth (lower is better)")
for name, fn in (("Brier  ||q-e_Y||^2", exp_brier), ("RPS (raw, 0..5)", exp_rps)):
    print(f"-- {name}")
    for fn_name, q in (("forecast A", A), ("forecast B", B)):
        print(f"   {fn_name}: truth A = {fmt(fn(q, A))}, truth B = {fmt(fn(q, B))}")
print("-- RPS normalized by K-1 = 5")
for fn_name, q in (("forecast A", A), ("forecast B", B)):
    print(f"   {fn_name}: truth A = {fmt(exp_rps(q, A) / 5)}, truth B = {fmt(exp_rps(q, B) / 5)}")
print("-- cross-entropy, nats (exact: infinite when the forecast gives zero mass to a realized rating)")
for fn_name, q in (("forecast A", A), ("forecast B", B)):
    print(f"   {fn_name}: truth A = {fmt(exp_ce(q, A))}, truth B = {fmt(exp_ce(q, B))}")
print("-- cross-entropy with 10% uniform smoothing of the FORECAST (labelled numerical device, not part of the theory)")
for fn_name, q in (("forecast A_s", smooth(A)), ("forecast B_s", smooth(B))):
    print(f"   {fn_name}: truth A = {fmt(exp_ce(q, A))}, truth B = {fmt(exp_ce(q, B))}")
# identities
assert exp_brier(B, A) == l2sq(B, A) + 1 - sum(x * x for x in A)
assert exp_brier(A, B) == l2sq(A, B) + 1 - sum(x * x for x in B)
assert l2sq(A, B) == Fr(3, 2) and exp_rps(B, A) - exp_rps(A, A) == 1
assert mean(A) == mean(B) == 2

print()
print("=" * 70)
print("PART 2. Population example: 3 equiprobable items, truthful judge T vs mean-collapsed judge D")
P = [pm((0, Fr(1, 2)), (2, Fr(1, 2))),   # c1, mean 1
     pm((0, Fr(1, 2)), (4, Fr(1, 2))),   # c2, mean 2
     pm((2, Fr(1, 2)), (4, Fr(1, 2)))]   # c3, mean 3
pi = [Fr(1, 3)] * 3
mus = [mean(p) for p in P]
print("conditional means =", [str(m) for m in mus], " conditional variances =", [str(var(p)) for p in P])
EY = sum(w * m for w, m in zip(pi, mus))
EY2 = sum(w * sum(Fr(k) ** 2 * p[k] for k in range(K)) for w, p in zip(pi, P))
VarY = EY2 - EY ** 2
VarMu = sum(w * m * m for w, m in zip(pi, mus)) - EY ** 2
EVar = sum(w * var(p) for w, p in zip(pi, P))
assert VarY == VarMu + EVar
print(f"Var(Y) = {VarY}, Var(mu) = {VarMu}, E Var(Y|C) = {EVar}")
r_true = VarMu / VarY
print(f"Expected-score readout (score = mean of the forecast, a function of c alone): both judges give s(c)=mu(c)")
print(f"   Pearson r(s,Y) = sqrt(Var mu / Var Y) = sqrt({r_true}) = {math.sqrt(r_true):.4f};  r^2 = {r_true}")
print(f"   Spearman/Kendall/MSE-of-the-mean: identical for T and D because s is identical")
D = [pm((int(m), 1)) for m in mus]
T = P
def popscore(fn, J):
    return sum(w * fn(q, p) for w, q, p in zip(pi, J, P))
print("Population-average expected scores (lower is better):")
for nm, fn in (("Brier", exp_brier), ("RPS raw", exp_rps)):
    print(f"   {nm:8s} T = {fmt(popscore(fn, T))}   D = {fmt(popscore(fn, D))}")
print(f"   RPS normalized (/5)  T = {fmt(popscore(exp_rps, T) / 5)}   D = {fmt(popscore(exp_rps, D) / 5)}")
print("   CE (nats)           T =", fmt(popscore(exp_ce, T)), "  D =", fmt(popscore(exp_ce, D)), " (D has zero mass on realized ratings)")
Ts = [smooth(q) for q in T]; Ds = [smooth(q) for q in D]
print("   CE, smoothed forecasts: T_s =", fmt(popscore(exp_ce, Ts)), "  D_s =", fmt(popscore(exp_ce, Ds)))
# skill vs base rate (pooled marginal law)
marg = [sum(w * p[k] for w, p in zip(pi, P)) for k in range(K)]
print("   pooled base-rate forecast =", [str(x) for x in marg])
for nm, fn in (("Brier", exp_brier), ("RPS", exp_rps)):
    base = popscore(fn, [marg] * 3)
    print(f"   skill 1 - S(judge)/S(base) [{nm}]: T = {fmt(1 - popscore(fn, T) / base)}   D = {fmt(1 - popscore(fn, D) / base)}   (base score {fmt(base)})")

print()
print("Sampled-score readout: the judge's score is a draw X ~ q(.|c), independent of Y given c")
r_sampled_T = VarMu / VarY
r_sampled_D = math.sqrt(VarMu) / math.sqrt(VarY)   # D's score is the constant mu(c) -> same as expected readout
print(f"   truthful T:   r = Var mu / Var Y = {r_sampled_T} = {float(r_sampled_T):.4f}")
print(f"   collapsed D:  r = sqrt(Var mu / Var Y) = {r_sampled_D:.4f}   -> D beats T on correlation, loses on every proper score")
# brute-force exact check of r_sampled_T from the joint law
def joint_corr_sampled():
    EX = EXY = EX2_ = Fr(0)
    for w, p in zip(pi, P):
        for x in range(K):
            for y in range(K):
                pr = w * p[x] * p[y]
                EX += pr * x; EXY += pr * x * y; EX2_ += pr * x * x
    return (EXY - EX * EY) / (EX2_ - EX ** 2)   # Var X = Var Y here since marginals coincide
assert joint_corr_sampled() == r_sampled_T

print()
print("=" * 70)
print("PART 3. Distance sensitivity (g): realized rating Y=0")
e0 = pm((0, 1)); F1 = pm((1, 1)); F5 = pm((5, 1))
for nm, q in (("point mass at 1", F1), ("point mass at 5", F5)):
    print(f"   forecast {nm}:  Brier = {exp_brier(q, e0)},  RPS raw = {exp_rps(q, e0)},  RPS normalized = {exp_rps(q, e0) / 5}")
assert exp_brier(F1, e0) == exp_brier(F5, e0) == 2
assert exp_rps(F1, e0) == 1 and exp_rps(F5, e0) == 5

print()
print("=" * 70)
print("PART 4. Seeded numeric checks of the identities used in the note (seed 0)")
random.seed(0)
import itertools
def rand_simplex(k):
    w = [Fr(random.randint(0, 9)) for _ in range(k)]
    if sum(w) == 0: w[0] = Fr(1)
    s = sum(w); return [x / s for x in w]
# (i) RPS expected = sum_k (Fq-Fp)^2 + sum_k Fp(1-Fp);  Brier expected = ||q-p||^2 + 1 - ||p||^2
for _ in range(200):
    p, q = rand_simplex(K), rand_simplex(K)
    Fp, Fq = cdf(p), cdf(q)
    assert exp_rps(q, p) == sum((Fq[t] - Fp[t]) ** 2 for t in range(K - 1)) + sum(Fp[t] * (1 - Fp[t]) for t in range(K - 1))
    assert exp_brier(q, p) == l2sq(q, p) + 1 - sum(x * x for x in p)
print("   200 random pairs (p,q): exact expected-RPS and expected-Brier identities hold")
# (ii) r^2 = 1 - min_{a,b} MSE / Var  on random finite data (exact normal equations)
for _ in range(200):
    n = 8
    s = [Fr(random.randint(0, 5)) for _ in range(n)]
    y = [Fr(random.randint(0, 5)) for _ in range(n)]
    ms, my = sum(s) / n, sum(y) / n
    vs = sum((x - ms) ** 2 for x in s) / n
    if vs == 0: continue
    vy = sum((x - my) ** 2 for x in y) / n
    if vy == 0: continue
    cov = sum((a - ms) * (b - my) for a, b in zip(s, y)) / n
    b_ = cov / vs; a_ = my - b_ * ms
    mse = sum((yy - a_ - b_ * ss) ** 2 for ss, yy in zip(s, y)) / n
    assert mse == vy - cov ** 2 / vs
    assert 1 - mse / vy == cov ** 2 / (vs * vy)
    # perturbing (a,b) never helps
    for da, db in ((Fr(1, 10), 0), (0, Fr(1, 10)), (Fr(-1, 7), Fr(1, 9))):
        assert sum((yy - (a_ + da) - (b_ + db) * ss) ** 2 for ss, yy in zip(s, y)) / n >= mse
print("   200 random finite datasets: r^2 = 1 - MSE(best affine)/Var(Y), exactly")
# (iii) affine invariance and the two examples
ys = [Fr(k) for k in (0, 1, 2, 3, 4, 5, 2, 3)]
def corr2(x, z):
    n = len(x); mx, mz = sum(x) / n, sum(z) / n
    c = sum((a - mx) * (b - mz) for a, b in zip(x, z)) / n
    vx = sum((a - mx) ** 2 for a in x) / n; vz = sum((b - mz) ** 2 for b in z) / n
    return c * c / (vx * vz)
assert corr2([y + 1 for y in ys], ys) == 1 and corr2([2 * y for y in ys], ys) == 1
mse_shift = sum(((y + 1) - y) ** 2 for y in ys) / len(ys)
mse_dbl = sum((2 * y - y) ** 2 for y in ys) / len(ys)
print(f"   s = Y+1: r^2 = 1, raw MSE = {mse_shift};   s = 2Y: r^2 = 1, raw MSE = E[Y^2] = {mse_dbl} on the 8-item toy list {[int(y) for y in ys]}")
# (iv) encoding-rank proposition: scalar encoding kernel dimension
print("   scalar encoding row (0,1,2,3,4,5) has rank 1; on the sum-zero subspace (dim 5) its kernel has dim", K - 2)
# (v) all-in-one: rank of cumulative map restricted to sum-zero subspace is K-1 (full)
print("   cumulative-sum map L: R^6 -> R^5 is injective on the sum-zero subspace (dim 5): kernel dim 0")
print("ALL ASSERTIONS PASSED")
