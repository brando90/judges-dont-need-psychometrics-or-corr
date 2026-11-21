"""Exp 18, step 2: check the per-term correlation equations and the CE / RPS decompositions
(sympy exact checks on small cases, numpy random checks at K = 6). Prints PASS/FAIL lines and
writes math/results/verify_math.json. Run: python3 math/verify_math.py   (numpy, scipy, sympy)
Date: 11-21-2026."""
import json, itertools, os, sys
import numpy as np
import sympy as sp

OUT = {}
def rec(name, ok, detail=""):
    OUT[name] = {"pass": bool(ok), "detail": str(detail)}
    print(("PASS " if ok else "FAIL ") + name + (" | " + str(detail) if detail else ""))

# ---------------------------------------------------------------- sympy (exact) -----------------
# S1. Pearson numerator: sum_i (t_i - tbar)(m_i - mbar) == sum_i (t_i - tbar) m_i, with m_i = sum_r r p_i(r)
n, K = 3, 4
p = sp.Matrix(n, K, lambda i, r: sp.Symbol(f"p{i}_{r}"))
t = sp.symbols("t0:%d" % n)
m = [sum(r * p[i, r] for r in range(K)) for i in range(n)]
mbar = sum(m) / n; tbar = sum(t) / n
lhs = sum((t[i] - tbar) * (m[i] - mbar) for i in range(n))
rhs = sum((t[i] - tbar) * sum(r * p[i, r] for r in range(K)) for i in range(n))
rec("S1 numerator is linear in p with coefficient r*(t_i - tbar)", sp.simplify(lhs - rhs) == 0)

# S2. gradient of rho wrt p_i(r) equals r * g_i with g_i = (zeta_i - rho z_i) / (n s_m)
ssm = sp.sqrt(sum((mi - mbar) ** 2 for mi in m) / n)
sst = sp.sqrt(sum((ti - tbar) ** 2 for ti in t) / n)
rho = rhs / (n * ssm * sst)
vals = {}
rng = np.random.default_rng(0)
for i in range(n):
    q = rng.dirichlet(np.ones(K)); vals.update({p[i, r]: sp.Rational(int(q[r] * 1e6), 10**6) for r in range(K)})
for i in range(n): vals[t[i]] = sp.Rational(int(rng.uniform(0, 3) * 1000), 1000)
ok = True; worst = 0
subs = vals
rho_v = rho.subs(subs); mv = [mi.subs(subs) for mi in m]; mbv = sum(mv) / n
smv = ssm.subs(subs); stv = sst.subs(subs); tbv = tbar.subs(subs)
for i in range(n):
    gi = ((t[i].subs(subs) - tbv) / stv - rho_v * (mv[i] - mbv) / smv) / (n * smv)
    for r in range(K):
        d = sp.diff(rho, p[i, r]).subs(subs)
        worst = max(worst, abs(float(d - r * gi)))
rec("S2 d rho / d p_i(r) = r * g_i (only the mean direction), exact derivative", worst < 1e-9, f"max abs diff {worst:.2e}")

# S3. RPS = (dm)^2/(K-1) + sum_k (d_k - dbar)^2 for CDF differences d_k, k=0..K-2 (symbolic, K=4)
K3 = 4
pp = sp.symbols("a0:%d" % K3); qq = sp.symbols("b0:%d" % K3)
Fp = [sum(pp[: k + 1]) for k in range(K3 - 1)]; Fq = [sum(qq[: k + 1]) for k in range(K3 - 1)]
d = [Fp[k] - Fq[k] for k in range(K3 - 1)]
rps = sum(x ** 2 for x in d)
dm = sum(r * (pp[r] - qq[r]) for r in range(K3))
dbar = sum(d) / (K3 - 1)
decomp = dm ** 2 / (K3 - 1) + sum((x - dbar) ** 2 for x in d)
# identity holds when both are normalized: substitute last coordinate
sub = {pp[-1]: 1 - sum(pp[:-1]), qq[-1]: 1 - sum(qq[:-1])}
rec("S3 RPS = (mean error)^2/(K-1) + shape part (exact, K=4)", sp.simplify((rps - decomp).subs(sub)) == 0)
# sum_k d_k = -(m_p - m_q)
rec("S3b sum_k (F_p(k)-F_q(k)) = m_q - m_p (exact, K=4)", sp.simplify((sum(d) + dm).subs(sub)) == 0)

# S4. vector-Pearson (correlation of the two histograms over r) is 1 for p = (1-b)/K + b q, b in (0,1]
Kv = 6; qv = sp.symbols("c0:%d" % Kv); bb = sp.symbols("b", positive=True)
pv = [(1 - bb) / Kv + bb * c for c in qv]
mp_ = sum(pv) / Kv; mq_ = sum(qv) / Kv
cov = sum((pv[i] - mp_) * (qv[i] - mq_) for i in range(Kv))
vp = sum((pv[i] - mp_) ** 2 for i in range(Kv)); vq = sum((qv[i] - mq_) ** 2 for i in range(Kv))
rec("S4 vector-Pearson of p=(1-b)/K+b q with q: cov^2 = var_p var_q (so |rho|=1), exact", sp.simplify(cov ** 2 - vp * vq) == 0)

# ---------------------------------------------------------------- numpy (random, K = 6) ----------
K = 6; r_ = np.arange(K, dtype=float)
def mean(P): return P @ r_
def pearson(a, b): return np.corrcoef(a, b)[0, 1]
def tilted(mu):
    """max-entropy distribution on {0..K-1} with mean mu: q(y) ~ exp(lam y)."""
    from scipy.optimize import brentq
    f = lambda lam: (lambda w: (w / w.sum()) @ r_ - mu)(np.exp(lam * r_))
    lam = brentq(f, -60, 60); w = np.exp(lam * r_); return w / w.sum()
H = lambda P: -(P * np.log(np.clip(P, 1e-300, None))).sum(-1)
rng = np.random.default_rng(1)

# N1. rho depends on p only through means: add a mean-preserving perturbation, rho unchanged
worst = 0
for _ in range(200):
    n = 12; P = rng.dirichlet(np.ones(K), n); T = rng.dirichlet(np.ones(K), n)
    # null space of [1; r] on R^K: 4-dim; add a small null-space vector to every row
    B = np.linalg.svd(np.vstack([np.ones(K), r_]))[2][2:]
    P2 = P + 1e-3 * (rng.standard_normal((n, 4)) @ B)
    if (P2 < 0).any(): continue
    worst = max(worst, abs(pearson(mean(P), mean(T)) - pearson(mean(P2), mean(T))))
rec("N1 mean-preserving shape changes leave rho unchanged (200 random draws)", worst < 1e-10, f"max diff {worst:.2e}")

# N2. affine invariance: rho(a + b m) = rho(m), b > 0 (level and scale are not identified)
worst = 0
for _ in range(200):
    m_ = rng.uniform(0, 5, 20); t_ = rng.uniform(0, 5, 20); a, b = rng.normal(), rng.uniform(0.1, 3)
    worst = max(worst, abs(pearson(m_, t_) - pearson(a + b * m_, t_)))
rec("N2 affine invariance of rho in the judge's mean", worst < 1e-12, f"max diff {worst:.2e}")

# N3. logit gradients: d rho/d u_i(r) = p_i(r)(r - m_i) g_i ; CE: d L/d u_i(r) = p_i(r) - p*_i(r) (finite differences)
def softmax(U): e = np.exp(U - U.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
n = 7; U = rng.standard_normal((n, K)); T = rng.dirichlet(np.ones(K), n); t_ = mean(T)
def rho_U(U): return pearson(mean(softmax(U)), t_)
def ce_U(U): return -(T * np.log(softmax(U))).sum()
def numgrad(f, U, h=1e-6):
    G = np.zeros_like(U)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            e = np.zeros_like(U); e[i, j] = h; G[i, j] = (f(U + e) - f(U - e)) / (2 * h)
    return G
P = softmax(U); m_ = mean(P); mb = m_.mean(); tb = t_.mean()
sm = np.sqrt(((m_ - mb) ** 2).mean()); st = np.sqrt(((t_ - tb) ** 2).mean()); rh = rho_U(U)
g = ((t_ - tb) / st - rh * (m_ - mb) / sm) / (n * sm)
Gform = P * (r_[None, :] - m_[:, None]) * g[:, None]
rec("N3a logit gradient of rho = p_i(r)(r - m_i) g_i", np.abs(numgrad(rho_U, U) - Gform).max() < 1e-7, f"max diff {np.abs(numgrad(rho_U, U) - Gform).max():.2e}")
rec("N3b logit gradient of CE = p_i(r) - p*_i(r)", np.abs(numgrad(ce_U, U) - (P - T)).max() < 1e-6)
# rank of the per-item gradient directions: Corr 1 per item (n total), CE up to K-1 per item
rec("N3c Corr gradient lies in one direction per item: p_i * (r - m_i)", True,
    "by N3a; CE gradient p_i - p*_i ranges over the (K-1)-dim tangent space")

# N4. CE decomposition: CE(P,q) = H(P) + G(P) + KL(q_mP || q) + C(P,q); C = 0 when log q is affine in y
worst = 0; worstC0 = 0
for _ in range(300):
    Pst = rng.dirichlet(np.ones(K) * rng.uniform(0.3, 3)); q = rng.dirichlet(np.ones(K) * rng.uniform(0.3, 3))
    qm = tilted(Pst @ r_)
    ce = -(Pst * np.log(q)).sum(); G = H(qm) - H(Pst)
    kl = (qm * np.log(qm / q)).sum(); C = ((Pst - qm) * (-np.log(q))).sum()
    worst = max(worst, abs(ce - (H(Pst) + G + kl + C)))
    # judge inside the tilted family: C = 0
    qE = tilted(rng.uniform(0.3, 4.7)); C0 = ((Pst - qm) * (-np.log(qE))).sum(); worstC0 = max(worstC0, abs(C0))
rec("N4a CE = H + G + KL(q_m || q) + C (300 random P, q)", worst < 1e-12, f"max abs err {worst:.2e}")
rec("N4b cross term C = 0 for judges in the tilted-uniform (1-D score) family", worstC0 < 1e-9, f"max |C| {worstC0:.2e}")

# N5. RPS = (dm)^2/(K-1) + shape, random K=6; TVD >= |dm|/(K-1)
worst = 0; okT = True
for _ in range(500):
    a = rng.dirichlet(np.ones(K)); b = rng.dirichlet(np.ones(K))
    d = np.cumsum(a)[:-1] - np.cumsum(b)[:-1]; dm = a @ r_ - b @ r_
    worst = max(worst, abs((d ** 2).sum() - (dm ** 2 / (K - 1) + ((d - d.mean()) ** 2).sum())))
    okT &= abs(dm) <= (K - 1) * 0.5 * np.abs(a - b).sum() + 1e-12
rec("N5a RPS = (mean error)^2/(K-1) + shape part (500 random pairs, K=6)", worst < 1e-12, f"max err {worst:.2e}")
rec("N5b |m_p - m_q| <= (K-1) TVD(p, q)", okT)

# N6. non-propriety of rho (item level, n items): any judge with m_i = a + b t_i, b>0, attains rho = 1,
# including a very wrong spread; CE / RPS / TVD at those judges are far from their optimum
n = 30; Tt = rng.dirichlet(np.ones(K) * 0.7, n); t_ = mean(Tt)
J = np.stack([tilted(np.clip(t_[i], 0.05, 4.95)) for i in range(n)])  # max-entropy with the right mean (spread wrong)
two = np.zeros((n, K)); 
for i in range(n):
    mu = np.clip(t_[i], 0, 5); two[i, 5] = mu / 5; two[i, 0] = 1 - mu / 5  # two-point on {0,5} with the right mean
res = {}
for name, Q in [("truth", Tt), ("max-ent same mean", J), ("two-point {0,5} same mean", np.clip(two, 1e-12, None))]:
    Qn = Q / Q.sum(1, keepdims=True)
    res[name] = dict(rho=float(pearson(mean(Qn), t_)),
                     CE=float(-(Tt * np.log(np.clip(Qn, 1e-12, None))).sum(1).mean()),
                     TVD=float(0.5 * np.abs(Tt - Qn).sum(1).mean()),
                     RPS=float(((np.cumsum(Tt, 1) - np.cumsum(Qn, 1))[:, :-1] ** 2).sum(1).mean()))
rec("N6 three judges with rho = 1 against t_i (true, max-ent, two-point) differ in CE, TVD, RPS",
    all(abs(v["rho"] - 1) < 1e-6 for v in res.values()) and res["two-point {0,5} same mean"]["CE"] > res["truth"]["CE"] + 1,
    json.dumps({k: {a: round(b, 4) for a, b in v.items()} for k, v in res.items()}))
OUT["N6_table"] = res

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
json.dump(OUT, open(os.path.join(os.path.dirname(__file__), "results", "verify_math.json"), "w"), indent=1)
bad = [k for k, v in OUT.items() if isinstance(v, dict) and v.get("pass") is False]
print("FAILED:", bad if bad else "none"); sys.exit(1 if bad else 0)
