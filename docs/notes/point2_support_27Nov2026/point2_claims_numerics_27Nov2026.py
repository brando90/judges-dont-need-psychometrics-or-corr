"""Quick randomized numerical checks of every Point 2 claim, run before the Lean proofs (27 Nov 2026).

Exact rational arithmetic (fractions.Fraction) on random finite laws, plus a randomized
counterexample search for each strict-propriety and anti-properness claim. Seed fixed; output is
identical on a second run. Exit code is nonzero if any claim FAILS.
Run: python3 point2_claims_numerics_27Nov2026.py
"""
import random, sys, itertools, math
from fractions import Fraction as F

SEED = 20261127
rng = random.Random(SEED)
N_TRIALS = 300
results = []

def rand_law(n, zeros=True):
    w = [F(rng.randint(0 if zeros else 1, 9)) for _ in range(n)]
    if sum(w) == 0:
        w[0] = F(1)
    s = sum(w)
    return [x / s for x in w]

def E(w, f): return sum(wi * fi for wi, fi in zip(w, f))
def cov(w, f, g): return E(w, [a * b for a, b in zip(f, g)]) - E(w, f) * E(w, g)
def var(w, f): return cov(w, f, f)

def record(name, ok, detail):
    results.append((name, ok, detail))

# (1) r^2 = 1 - min_{a,b} E(Y-a-bS)^2 / Var(Y)
ok = True; n_done = 0
for _ in range(N_TRIALS):
    n = rng.randint(2, 8); w = rand_law(n)
    S = [F(rng.randint(-5, 5)) for _ in range(n)]; Y = [F(rng.randint(-5, 5)) for _ in range(n)]
    vS, vY, c = var(w, S), var(w, Y), cov(w, S, Y)
    if vS == 0 or vY == 0: continue
    n_done += 1
    b = c / vS; a = E(w, Y) - b * E(w, S)
    mse_star = E(w, [(y - a - b * s) ** 2 for s, y in zip(S, Y)])
    r2 = c * c / (vS * vY)
    ok &= (mse_star == vY * (1 - r2))
    for _ in range(20):  # random competitors never do better
        a2 = F(rng.randint(-20, 20), rng.randint(1, 5)); b2 = F(rng.randint(-20, 20), rng.randint(1, 5))
        ok &= E(w, [(y - a2 - b2 * s) ** 2 for s, y in zip(S, Y)]) >= mse_star
record("(1) r^2 identity, minimizer (a*,b*), random (a,b) never lower", ok, f"{n_done} laws x 20 competitors")

# (2) affine invariance r(aS+b,Y) = sgn(a) r(S,Y), a != 0 (exact via r^2 and sign of cov)
ok = True
for _ in range(N_TRIALS):
    n = rng.randint(2, 8); w = rand_law(n)
    S = [F(rng.randint(-5, 5)) for _ in range(n)]; Y = [F(rng.randint(-5, 5)) for _ in range(n)]
    if var(w, S) == 0 or var(w, Y) == 0: continue
    a = F(rng.choice([-1, 1]) * rng.randint(1, 9), rng.randint(1, 4)); b = F(rng.randint(-9, 9))
    T = [a * s + b for s in S]
    r2S = cov(w, S, Y) ** 2 / (var(w, S) * var(w, Y)); r2T = cov(w, T, Y) ** 2 / (var(w, T) * var(w, Y))
    ok &= (r2S == r2T) and (cov(w, T, Y) == a * cov(w, S, Y))
record("(2) Pearson affine invariance with sign, a<0 included", ok, f"{N_TRIALS} trials")

# (3) Spearman/Kendall: functions of the readout; invariant under strictly increasing maps
def midrank(w, s):
    return [sum(wj * (F(1) if sj < si else (F(1, 2) if sj == si else F(0))) for wj, sj in zip(w, s)) for si in s]
def sgn(x): return (x > 0) - (x < 0)
def kendall(w, s, y): return sum(w[i] * w[j] * sgn((s[i] - s[j]) * (y[i] - y[j])) for i in range(len(s)) for j in range(len(s)))
def spear_num(w, s, y): return cov(w, midrank(w, s), midrank(w, y))
ok = True
for _ in range(N_TRIALS):
    K = 6; nC = rng.randint(2, 5)
    # forecasts q, q' with the same mean per item (q' = q + t v, v in ker(mean) cap D)
    items = []
    for c in range(nC):
        q = rand_law(K, zeros=False)
        v = [F(1), F(-2), F(1), F(0), F(0), F(0)]; t = min(q) / 4
        q2 = [qi + t * vi for qi, vi in zip(q, v)]
        assert min(q2) >= 0 and sum(q2) == 1
        items.append((q, q2))
    mu = [sum(k * qk for k, qk in enumerate(q)) for q, _ in items]
    mu2 = [sum(k * qk for k, qk in enumerate(q2)) for _, q2 in items]
    ok &= (mu == mu2)
    pi = rand_law(nC, zeros=False); P = [rand_law(K) for _ in range(nC)]
    w = [pi[c] * P[c][y] for c in range(nC) for y in range(K)]
    Y = [F(y) for c in range(nC) for y in range(K)]
    s1 = [mu[c] for c in range(nC) for y in range(K)]; s2 = [mu2[c] for c in range(nC) for y in range(K)]
    ok &= kendall(w, s1, Y) == kendall(w, s2, Y) and spear_num(w, s1, Y) == spear_num(w, s2, Y)
    ok &= E(w, [(a - b) ** 2 for a, b in zip(s1, Y)]) == E(w, [(a - b) ** 2 for a, b in zip(s2, Y)])
    f = lambda x: x ** 3 + 7 * x  # strictly increasing
    s3 = [f(x) for x in s1]
    ok &= kendall(w, s3, Y) == kendall(w, s1, Y) and midrank(w, s3) == midrank(w, s1)
record("(3) same-mean forecasts tie on Pearson/Spearman/Kendall/MSE of the expected readout; rank stats invariant under strictly increasing maps", ok, f"{N_TRIALS} trials, K=6")

# (4)(5)(7) encoding identity, Brier and RPS strict propriety, scalar ties
def S_A(A, q, y):
    K = len(q); return sum(sum(A[j][k] * (q[k] - (1 if k == y else 0)) for k in range(K)) ** 2 for j in range(len(A)))
def expS(A, q, p): return sum(p[y] * S_A(A, q, y) for y in range(len(p)))
def formula(A, q, p):
    K = len(q)
    t1 = sum(sum(A[j][k] * (q[k] - p[k]) for k in range(K)) ** 2 for j in range(len(A)))
    t2 = sum(sum(A[j][k] ** 2 * p[k] for k in range(K)) - sum(A[j][k] * p[k] for k in range(K)) ** 2 for j in range(len(A)))
    return t1 + t2
def cdf(q, t): return sum(q[: t + 1])
def rps(q, y): return sum((cdf(q, t) - (1 if y <= t else 0)) ** 2 for t in range(len(q) - 1))
ok_enc = ok_brier = ok_rps = ok_scalar = True; counterex = 0
for _ in range(N_TRIALS):
    K = rng.randint(2, 7); p = rand_law(K); q = rand_law(K); m = rng.randint(1, 4)
    A = [[F(rng.randint(-3, 3)) for _ in range(K)] for _ in range(m)]
    ok_enc &= expS(A, q, p) == formula(A, q, p)
    I = [[F(int(j == k)) for k in range(K)] for j in range(K)]
    L = [[F(int(k <= t)) for k in range(K)] for t in range(K - 1)]
    ok_enc &= all(S_A(L, q, y) == rps(q, y) for y in range(K))
    eb = expS(I, q, p); ok_brier &= eb == sum((a - b) ** 2 for a, b in zip(q, p)) + 1 - sum(b * b for b in p)
    er = sum(p[y] * rps(q, y) for y in range(K))
    ok_rps &= er == sum((cdf(q, t) - cdf(p, t)) ** 2 + cdf(p, t) * (1 - cdf(p, t)) for t in range(K - 1))
    if q != p:  # counterexample search for strict propriety
        if not (expS(I, q, p) > expS(I, p, p)): counterex += 1
        if not (sum(p[y] * rps(q, y) for y in range(K)) > sum(p[y] * rps(p, y) for y in range(K))): counterex += 1
    if K >= 3:
        Sc = [[F(k) for k in range(K)]]
        v = [F(0)] * K; v[0], v[1], v[2] = F(1), F(-2), F(1)
        ok_scalar &= sum(Sc[0][k] * v[k] for k in range(K)) == 0 and sum(v) == 0
        qq = [x + min(q) / 4 * vi for x, vi in zip(q, v)]
        ok_scalar &= expS(Sc, qq, p) == expS(Sc, q, p)
record("(7) encoding identity E S_A = ||A(q-p)||^2 + tr(A Sigma_p A^T); RPS = S_L", ok_enc, f"{N_TRIALS} random A,p,q, K in 2..7")
record("(5) Brier expectation ||q-p||^2 + 1 - ||p||^2, any K", ok_brier, f"{N_TRIALS} trials")
record("(4) RPS expectation sum (F_q-F_p)^2 + sum F_p(1-F_p)", ok_rps, f"{N_TRIALS} trials")
record("(4)(5) strict propriety: counterexample search (q != p never ties or wins)", counterex == 0, f"{counterex} counterexamples")
record("(7) scalar encoding not strictly proper for K>=3 (v=(1,-2,1,0..) in ker cap D, ties)", ok_scalar, "K in 3..7")
ok_k2 = True
for a0, a1 in itertools.product(range(-3, 4), repeat=2):
    v = [F(a0), F(a1)]
    if sum(v) == 0 and 0 * v[0] + 1 * v[1] == 0: ok_k2 &= v == [0, 0]
record("(7) scalar encoding injective on D for K=2", ok_k2, "grid search")

# (6) Murphy decomposition, finite support, plus calibrated <=> reliability 0, plus r^2 Var Y = res - E(g-S')^2
ok = True; ok_cal = True; ok_aff = True
for _ in range(N_TRIALS):
    n = rng.randint(2, 9); w = rand_law(n)
    S = [F(rng.randint(0, 3)) for _ in range(n)]; Y = [F(rng.randint(0, 5)) for _ in range(n)]
    def g_of(Sv):
        out = []
        for i in range(n):
            ws = sum(w[j] for j in range(n) if Sv[j] == Sv[i]); wy = sum(w[j] * Y[j] for j in range(n) if Sv[j] == Sv[i])
            out.append(wy / ws if ws != 0 else F(0))
        return out
    g = g_of(S)
    lhs = E(w, [(y - s) ** 2 for s, y in zip(S, Y)])
    rel = E(w, [(gi - s) ** 2 for gi, s in zip(g, S)])
    ok &= lhs == var(w, Y) - var(w, g) + rel
    cal = all(w[i] == 0 or g[i] == S[i] for i in range(n))
    ok_cal &= (rel == 0) == cal
    gS = g  # calibrated forecast g(S) has zero reliability
    ok_cal &= E(w, [(a - b) ** 2 for a, b in zip(g_of(gS), gS)]) == 0
    if var(w, S) != 0 and var(w, Y) != 0 and cov(w, S, Y) != 0:
        b = cov(w, S, Y) / var(w, S); a = E(w, Y) - b * E(w, S); Sp = [a + b * s for s in S]
        r2 = cov(w, S, Y) ** 2 / (var(w, S) * var(w, Y))
        ok_aff &= g_of(Sp) == g and r2 * var(w, Y) == var(w, g) - E(w, [(gi - s) ** 2 for gi, s in zip(g, Sp)])
record("(6) Murphy: E(Y-S)^2 = Var Y - Var g(S) + E(g(S)-S)^2", ok, f"{N_TRIALS} finite laws")
record("(6) reliability = 0 iff calibrated on the support", ok_cal, f"{N_TRIALS} finite laws")
record("(6) r^2 Var Y = Var g(S) - E(g(S)-S')^2 after best affine refit", ok_aff, "trials with Cov != 0")

# Prop. expected: r(s,Y) <= r(mu,Y) for item-level readouts (exact via squares and signs)
ok = True; viol = 0
for _ in range(N_TRIALS):
    K = 6; nC = rng.randint(2, 5); pi = rand_law(nC, zeros=False); P = [rand_law(K) for _ in range(nC)]
    mu = [sum(k * P[c][k] for k in range(K)) for c in range(nC)]
    w = [pi[c] * P[c][y] for c in range(nC) for y in range(K)]; Y = [F(y) for c in range(nC) for y in range(K)]
    sv = [F(rng.randint(-5, 5)) for _ in range(nC)]
    s = [sv[c] for c in range(nC) for y in range(K)]; m = [mu[c] for c in range(nC) for y in range(K)]
    if var(w, s) == 0 or var(w, m) == 0: continue
    cs, cm = cov(w, s, Y), cov(w, m, Y)
    # r(s,Y) <= r(mu,Y) <=> cs/sqrt(vs) <= cm/sqrt(vm); cm = var(m) > 0
    if cs > 0 and cs * cs * var(w, m) > cm * cm * var(w, s): viol += 1
    ok &= cm == var(w, m)
record("Prop. expected: r(s,Y) <= r(mu,Y) for item-level s (search), Cov(mu,Y)=Var mu", ok and viol == 0, f"{viol} violations")

# (8) sampled readout: r_truthful = rho, r_collapsed = sqrt(rho) > rho
ok = True; viol = 0
for _ in range(N_TRIALS):
    K = 6; nC = rng.randint(2, 5); pi = rand_law(nC, zeros=False); P = [rand_law(K) for _ in range(nC)]
    mu = [sum(k * P[c][k] for k in range(K)) for c in range(nC)]
    EY = sum(pi[c] * mu[c] for c in range(nC)); EY2 = sum(pi[c] * sum(k * k * P[c][k] for k in range(K)) for c in range(nC))
    vY = EY2 - EY ** 2; vmu = sum(pi[c] * mu[c] ** 2 for c in range(nC)) - EY ** 2
    eV = vY - vmu
    if vmu == 0 or eV == 0: continue
    # truthful sampled judge: joint law over (c, x, y), x and y iid from P[c]
    EXY = sum(pi[c] * P[c][x] * P[c][y] * x * y for c in range(nC) for x in range(K) for y in range(K))
    rT = (EXY - EY * EY) / vY  # Var X = Var Y (same marginal)
    a = F(rng.randint(1, 5)); b = F(rng.randint(-3, 3))
    cD = sum(pi[c] * P[c][y] * (a * mu[c] + b) * y for c in range(nC) for y in range(K)) - (a * EY + b) * EY
    vD = sum(pi[c] * (a * mu[c] + b) ** 2 for c in range(nC)) - (a * EY + b) ** 2
    rho = vmu / vY
    rD2 = cD * cD / (vD * vY)
    ok &= rT == rho and rD2 == rho and cD > 0 and 0 < rho < 1
    if not (math.sqrt(rD2) > rT): viol += 1
record("(8) sampled readout: r_truthful = rho, r_collapsed^2 = rho, sqrt(rho) > rho", ok and viol == 0, f"{viol} violations")

# claimed-false statement must fail: Pearson is NOT invariant to a<0 without the sign
neg_found = False
w = [F(1, 2)] * 2; S = [F(0), F(1)]; Y = [F(0), F(1)]
T = [-s for s in S]
neg_found = cov(w, T, Y) / var(w, T) != cov(w, S, Y) / var(w, S)
record("claimed-false: r(-S,Y) = r(S,Y) fails (sign flip)", neg_found, "2-point law")

bad = 0
for name, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  [{detail}]"); bad += (not ok)
print(f"seed={SEED}; {len(results) - bad}/{len(results)} PASS")
sys.exit(1 if bad else 0)
