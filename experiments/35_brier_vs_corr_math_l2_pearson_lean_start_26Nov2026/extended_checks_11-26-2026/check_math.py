"""Numerical and exact-rational checks for experiment 35.

Brier score, cross-entropy (CE), ranked probability score (RPS), squared error (L2)
and Pearson correlation: which are proper, what each identifies, and exact
counterexamples. Synthetic data only; no human ratings are used anywhere.

Run:  python check_math.py   (from this folder or the repo root)
Prints one PASS/FAIL line per check, writes results/*.csv and figures/*.{pdf,png}.
Notation follows the memo appendix B: p = true rating distribution, q = forecast.
"""
from __future__ import annotations

import csv
import os
from fractions import Fraction as Fr

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

K = 6
R = np.arange(K)
TOL = 1e-12
results = []  # (id, name, pass, detail)


def report(cid, name, ok, detail):
    results.append((cid, name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'} {cid} {name}: {detail}")


def rand_simplex(rng, k=K, n=None):
    return rng.dirichlet(np.ones(k), size=n)


# ---------------------------------------------------------------- scores
def brier_expected(q, p):
    """E_{Y~p} sum_k (q_k - 1[Y=k])^2, computed by brute force over Y."""
    k = len(p)
    return sum(p[y] * np.sum((q - np.eye(k)[y]) ** 2) for y in range(k))


def ce_expected(q, p):
    return -np.sum(p * np.log(q))


def rps_expected(q, p):
    k = len(p)
    Q = np.cumsum(q)[:-1]
    tot = 0.0
    for y in range(k):
        o = (np.arange(k - 1) >= y).astype(float)
        tot += p[y] * np.sum((Q - o) ** 2) / (k - 1)
    return tot


def tvd(a, b, axis=-1):
    return 0.5 * np.sum(np.abs(a - b), axis=axis)


def corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return np.mean((a - a.mean()) * (b - b.mean())) / (a.std() * b.std())


# =============================================================== Q1
def q1_checks():
    rng = np.random.default_rng(1)
    # C1: Brier decomposition E_p S(q,Y) = sum (q-p)^2 + 1 - sum p^2
    err = 0.0
    for _ in range(2000):
        p, q = rand_simplex(rng), rand_simplex(rng)
        lhs = brier_expected(q, p)
        rhs = np.sum((q - p) ** 2) + 1 - np.sum(p ** 2)
        err = max(err, abs(lhs - rhs))
    report("N1", "Brier decomposition (2000 random K=6 pairs)", err < TOL, f"max abs err {err:.2e}")

    # C2: CE = H(p) + KL(p||q)
    err = 0.0
    for _ in range(2000):
        p, q = rand_simplex(rng), rand_simplex(rng)
        H = -np.sum(p * np.log(p))
        KL = np.sum(p * np.log(p / q))
        err = max(err, abs(ce_expected(q, p) - (H + KL)))
    report("N2", "CE = H + KL (2000 random pairs)", err < TOL, f"max abs err {err:.2e}")

    # C3: RPS decomposition and strict propriety
    err, minexcess = 0.0, np.inf
    for _ in range(2000):
        p, q = rand_simplex(rng), rand_simplex(rng)
        P, Q = np.cumsum(p)[:-1], np.cumsum(q)[:-1]
        rhs = (np.sum((Q - P) ** 2) + np.sum(P * (1 - P))) / (K - 1)
        err = max(err, abs(rps_expected(q, p) - rhs))
        minexcess = min(minexcess, rps_expected(q, p) - rps_expected(p, p))
    report("N3", "RPS decomposition; excess>0 for q!=p (2000 pairs)", err < TOL and minexcess > 0,
           f"max abs err {err:.2e}; smallest excess {minexcess:.3e}")

    # N4: strict propriety by perturbation: for 500 truths p, 200 random q each, no q beats p
    worst = {"Brier": np.inf, "CE": np.inf, "RPS": np.inf}
    for _ in range(500):
        p = rand_simplex(rng)
        for q in rand_simplex(rng, n=200):
            worst["Brier"] = min(worst["Brier"], brier_expected(q, p) - brier_expected(p, p))
            worst["CE"] = min(worst["CE"], ce_expected(q, p) - ce_expected(p, p))
            worst["RPS"] = min(worst["RPS"], rps_expected(q, p) - rps_expected(p, p))
    report("N4", "no random forecast beats the truth (100000 draws)", all(v > 0 for v in worst.values()),
           "smallest excess " + ", ".join(f"{k} {v:.2e}" for k, v in worst.items()))

    # N5: empirical frequencies minimize empirical Brier over the simplex (numerical optimizer agrees)
    maxgap = 0.0
    for s in range(50):
        r2 = np.random.default_rng(100 + s)
        y = r2.integers(0, K, size=int(r2.integers(5, 60)))
        E = np.eye(K)[y]
        phat = E.mean(0)

        def emp(qz):
            q = np.exp(qz - qz.max()); q /= q.sum()
            return np.mean(np.sum((q - E) ** 2, 1))
        # closed form: emp(q) = sum (q - phat)^2 + const, so phat is the unique minimizer
        const = np.mean(np.sum((phat - E) ** 2, 1))
        for q in rand_simplex(r2, n=100):
            lhs = np.mean(np.sum((q - E) ** 2, 1))
            maxgap = max(maxgap, abs(lhs - (np.sum((q - phat) ** 2) + const)))
    report("N5", "empirical Brier = ||q - phat||^2 + const (50 samples x 100 q)", maxgap < TOL,
           f"max abs err {maxgap:.2e}")

    # N6: consistency of the empirical frequency (finite-context nonparametric case)
    p = np.array([0.05, 0.10, 0.25, 0.30, 0.20, 0.10])
    Ns = [10, 30, 100, 300, 1000, 3000, 10000, 100000]
    rows = []
    r3 = np.random.default_rng(6)
    for N in Ns:
        tv = [tvd(np.bincount(r3.choice(K, N, p=p), minlength=K) / N, p) for _ in range(200)]
        rows.append((N, float(np.mean(tv))))
    ok = rows[-1][1] < 0.01 and all(rows[i + 1][1] < rows[i][1] for i in range(len(rows) - 1))
    report("N6", "mean TVD(phat_N, p) over 200 reps decreases in N", ok,
           "; ".join(f"N={n}: {v:.4f}" for n, v in rows))

    # N7: misspecified class -> different proper scores pick different best-in-class forecasts
    pt = np.array([0.5, 0.1, 0.4])  # K=3 truth; class: Binomial(2, theta)

    def fam(th):
        return np.array([(1 - th) ** 2, 2 * th * (1 - th), th ** 2])
    thB = minimize_scalar(lambda t: brier_expected(fam(t), pt), bounds=(1e-6, 1 - 1e-6), method="bounded",
                          options={"xatol": 1e-12}).x
    thC = minimize_scalar(lambda t: ce_expected(fam(t), pt), bounds=(1e-6, 1 - 1e-6), method="bounded",
                          options={"xatol": 1e-12}).x
    thR = minimize_scalar(lambda t: rps_expected(fam(t), pt), bounds=(1e-6, 1 - 1e-6), method="bounded",
                          options={"xatol": 1e-12}).x
    # CE in this family is maximum likelihood: theta = mean/2 = 0.45 exactly
    report("N7", "misspecified Binomial(2,theta) class, truth (0.5,0.1,0.4): best theta differs by score",
           abs(thC - 0.45) < 1e-8 and abs(thB - thC) > 1e-3,
           f"theta_CE={thC:.6f} (exact 0.45), theta_Brier={thB:.6f}, theta_RPS={thR:.6f}")

    # propriety curves (figure 1), K=2, true P(Y=1)=0.3
    pr = np.linspace(0.005, 0.995, 397)
    p1 = 0.3
    eb = 2 * (pr - p1) ** 2 + 2 * p1 * (1 - p1)
    ec = -(p1 * np.log(pr) + (1 - p1) * np.log(1 - pr))
    el2 = None
    report("N8", "K=2 propriety curves: grid argmin of expected Brier and CE is the truth 0.3",
           abs(pr[np.argmin(eb)] - p1) < 0.003 and abs(pr[np.argmin(ec)] - p1) < 0.003,
           f"argmin Brier {pr[np.argmin(eb)]:.4f}, argmin CE {pr[np.argmin(ec)]:.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(pr, eb, color="C0")
    ax[0].axvline(p1, color="k", ls="--", lw=1)
    ax[0].set_xlabel("Reported probability of rating 1 (synthetic, K=2)")
    ax[0].set_ylabel("Expected Brier score under truth 0.3\n(lower is better)")
    ax[0].set_title("Brier: minimum at the truth")
    ax[1].plot(pr, ec, color="C1")
    ax[1].axvline(p1, color="k", ls="--", lw=1)
    ax[1].set_xlabel("Reported probability of rating 1 (synthetic, K=2)")
    ax[1].set_ylabel("Expected cross-entropy (CE), nats\n(lower is better)")
    ax[1].set_title("CE: minimum at the truth")
    for a in ax:
        a.grid(True)
    fig.suptitle("Exact expected scores, no sampling, no human ratings; dashed line = true probability 0.3", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig1_propriety_curves.{ext}"), dpi=150)
    plt.close(fig)


# =============================================================== Q2
def q2_checks():
    rng = np.random.default_rng(2)
    # N9: L2 decomposition on random finite distributions over arbitrary real support
    err = 0.0
    for _ in range(2000):
        v = rng.normal(size=7) * 3
        w = rand_simplex(rng, 7)
        f = rng.normal() * 3
        mu = np.sum(w * v)
        var = np.sum(w * (v - mu) ** 2)
        err = max(err, abs(np.sum(w * (f - v) ** 2) - ((f - mu) ** 2 + var)))
    report("N9", "E(f-Y)^2 = (f-mu)^2 + Var Y (2000 random finite laws)", err < 1e-10 * 1 or err < TOL,
           f"max abs err {err:.2e}")

    # N10: L2 empirical minimizer (sample mean) converges to mu under non-Gaussian noise
    laws = {
        "skewed exponential(1)-1": (lambda g, n: g.exponential(1, n) - 1.0, 0.0),
        "bimodal 0.5N(-2,0.25)+0.5N(2,0.25)": (lambda g, n: np.where(g.random(n) < 0.5, -2, 2) + 0.5 * g.normal(size=n), 0.0),
        "heavy-tailed Student t(3)": (lambda g, n: g.standard_t(3, n), 0.0),
    }
    lines, ok = [], True
    for name, (draw, mu) in laws.items():
        g = np.random.default_rng(10)
        errs = []
        for N in [100, 10000, 1000000]:
            y = draw(g, N)
            fhat = minimize_scalar(lambda f: np.mean((f - y) ** 2)).x  # numeric L2 ERM
            errs.append(abs(fhat - mu))
        ok &= errs[-1] < 0.01
        lines.append(f"{name}: |fhat-mu| at N=1e2,1e4,1e6 = " + ",".join(f"{e:.4f}" for e in errs))
    report("N10", "L2 ERM recovers the mean without Gaussian noise", ok, " | ".join(lines))

    # N11: L2 is not proper for the distribution: two forecasts with equal mean, equal expected L2
    # truth bimodal G = 0.5N(-2,0.5^2)+0.5N(2,0.5^2); forecasts F1 = G, F2 = N(0,1). Score S(F,y)=(mean_F - y)^2.
    g = np.random.default_rng(11)
    y = np.where(g.random(10 ** 6) < 0.5, -2, 2) + 0.5 * g.normal(size=10 ** 6)
    sF1 = np.mean((0.0 - y) ** 2)
    sF2 = np.mean((0.0 - y) ** 2)
    # log score distinguishes them
    ls1 = -np.mean(np.log(0.5 * stats.norm.pdf(y, -2, 0.5) + 0.5 * stats.norm.pdf(y, 2, 0.5)))
    ls2 = -np.mean(stats.norm.logpdf(y, 0, 1))
    report("N11", "unimodal N(0,1) and the true bimodal law get identical L2 score; log score separates them",
           sF1 == sF2 and ls2 - ls1 > 1.0,
           f"L2 both {sF1:.4f} (exact 4.25); log score truth {ls1:.4f} vs N(0,1) {ls2:.4f} nats")

    # N12: one-hot L2 == Brier, exactly
    err = 0.0
    for _ in range(1000):
        q = rand_simplex(rng)
        y = rng.integers(K)
        err = max(err, abs(np.sum((q - np.eye(K)[y]) ** 2) - (np.sum(q ** 2) - 2 * q[y] + 1)))
    report("N12", "L2 on one-hot vector equals Brier (1000 draws)", err < TOL, f"max abs err {err:.2e}")

    # N13: 6-point example, exact rationals. truth p, forecast b with the same mean 5/2
    p = [Fr(x, 20) for x in (1, 2, 7, 7, 2, 1)]
    b = [Fr(x, 20) for x in (7, 2, 1, 1, 2, 7)]
    mean = lambda d: sum(Fr(r) * d[r] for r in range(K))
    l2 = lambda f, d: sum(d[r] * (mean(f) - r) ** 2 for r in range(K))  # E_{Y~d}(mean_f - Y)^2
    bri = lambda f, d: sum(d[y] * sum((f[k] - (1 if k == y else 0)) ** 2 for k in range(K)) for y in range(K))

    def rpsx(f, d):
        F = [sum(f[: j + 1]) for j in range(K - 1)]
        return sum(d[y] * sum((F[j] - (1 if y <= j else 0)) ** 2 for j in range(K - 1)) for y in range(K)) / (K - 1)
    ceb = ce_expected(np.array([float(x) for x in b]), np.array([float(x) for x in p]))
    cep = ce_expected(np.array([float(x) for x in p]), np.array([float(x) for x in p]))
    ok = mean(p) == mean(b) == Fr(5, 2) and l2(p, p) == l2(b, p) and bri(b, p) > bri(p, p) and rpsx(b, p) > rpsx(p, p) and ceb > cep
    report("N13", "6-point equal-mean example: same L2 risk, different Brier/RPS/CE (exact)", ok,
           f"means {mean(p)},{mean(b)}; L2 risk {l2(p,p)} both; Brier {bri(p,p)} vs {bri(b,p)}; "
           f"RPS {rpsx(p,p)} vs {rpsx(b,p)}; CE {cep:.4f} vs {ceb:.4f} nats")
    return p, b


# =============================================================== Q3
def q3_checks():
    rng = np.random.default_rng(3)
    # N14: MSE = (mx-my)^2 + (sx-sy)^2 + 2 sx sy (1-rho); standardized: 2(1-rho)
    e1 = e2 = 0.0
    for _ in range(2000):
        n = int(rng.integers(3, 50))
        x = rng.normal(size=n) * rng.uniform(0.1, 5) + rng.normal() * 3
        y = 0.5 * x + rng.normal(size=n) * rng.uniform(0.1, 5) + rng.normal()
        rho = corr(x, y)
        lhs = np.mean((x - y) ** 2)
        rhs = (x.mean() - y.mean()) ** 2 + (x.std() - y.std()) ** 2 + 2 * x.std() * y.std() * (1 - rho)
        e1 = max(e1, abs(lhs - rhs) / max(1, abs(lhs)))
        zx, zy = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
        e2 = max(e2, abs(np.mean((zx - zy) ** 2) - 2 * (1 - rho)))
    report("N14", "MSE = bias^2 + scale^2 + 2 sx sy (1-rho), and 2(1-rho) when standardized (2000 samples)",
           e1 < TOL and e2 < TOL, f"max rel err {e1:.2e}, {e2:.2e}")

    # N15: affine invariance
    err = 0.0
    for _ in range(1000):
        x, y = rng.normal(size=20), rng.normal(size=20)
        a, b = rng.uniform(0.01, 100), rng.normal() * 10
        err = max(err, abs(corr(a * x + b, y) - corr(x, y)))
    report("N15", "corr(aX+b, Y) = corr(X, Y) for a>0 (1000 draws)", err < TOL, f"max abs err {err:.2e}")

    # N16: correlation ratio: max_f corr(f(X), Y) = eta, attained by affine images of E[Y|X]
    best_gap, att = np.inf, 0.0
    for _ in range(200):
        nx = 5
        px = rand_simplex(rng, nx)
        pyx = rand_simplex(rng, K, n=nx)  # rows: p(y|x)
        mu = pyx @ R
        EY = px @ mu
        VarY = px @ (pyx @ R ** 2) - EY ** 2
        eta = np.sqrt((px @ (mu - EY) ** 2) / VarY)

        def pcorr(f):
            Ef = px @ f
            cov = px @ (f * mu) - Ef * EY
            return cov / np.sqrt((px @ f ** 2 - Ef ** 2) * VarY)
        att = max(att, abs(pcorr(3 * mu - 7) - eta))
        for f in rng.normal(size=(200, nx)):
            best_gap = min(best_gap, eta - pcorr(f))
    report("N16", "max_f corr(f(X),Y) = correlation ratio eta, attained by 3*E[Y|X]-7 (200 laws x 200 f)",
           att < TOL and best_gap >= -TOL, f"attainment err {att:.2e}; smallest eta - corr(f) {best_gap:.2e}")

    # N17: min_{a,b} mean (y - (a x + b))^2 = Var(y)(1 - rho^2)
    err = 0.0
    for _ in range(1000):
        x = rng.normal(size=30)
        y = 2 * x + rng.normal(size=30) * 2
        A = np.c_[x, np.ones_like(x)]
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        mse = np.mean((y - A @ coef) ** 2)
        err = max(err, abs(mse - y.var() * (1 - corr(x, y) ** 2)))
    report("N17", "oracle affine recalibration: min MSE = Var(Y)(1-rho^2) (1000 samples)", err < TOL,
           f"max abs err {err:.2e}")

    # N18: correlation is not an average of per-item scores: the effect of changing item 1's forecast
    # depends on the other items, which is impossible for (1/n) sum_i S(f_i, y_i).
    y = np.array([0.0, 1.0, 0.0, 1.0])
    ctxA, ctxB = np.array([0.2, 0.8, 0.3, 0.7]), np.array([0.2, 0.6, 0.5, 0.9])
    dA = corr(np.r_[0.9, ctxA[1:]], y) - corr(ctxA, y)
    dB = corr(np.r_[0.9, ctxB[1:]], y) - corr(ctxB, y)
    report("N18", "same change to item 1 (forecast 0.2->0.9, rating 0) shifts corr by context-dependent amounts",
           abs(dA - dB) > 1e-3, f"delta corr in context A {dA:.4f}, in context B {dB:.4f}")

    # N19 exact counterexample (i): binary, equal correlation, different Brier and CE
    t = [Fr(1, 10), Fr(3, 10), Fr(1, 2), Fr(7, 10), Fr(9, 10)]
    A = list(t)
    B = [Fr(1, 2) + Fr(1, 10) * (ti - Fr(1, 2)) for ti in t]

    def ebrier2(m, ti):  # two-category Brier: 2 (m - y)^2 expected under y~Bern(ti)
        return 2 * ((m - ti) ** 2 + ti * (1 - ti))

    def cov(u, v):
        n = len(u); mu, mv = sum(u) / n, sum(v) / n
        return sum((a - mu) * (b - mv) for a, b in zip(u, v)) / n
    rho2A = cov(A, t) ** 2 / (cov(A, A) * cov(t, t))
    rho2B = cov(B, t) ** 2 / (cov(B, B) * cov(t, t))
    bA = sum(ebrier2(m, ti) for m, ti in zip(A, t)) / 5
    bB = sum(ebrier2(m, ti) for m, ti in zip(B, t)) / 5
    ceA = np.mean([-(float(ti) * np.log(float(m)) + (1 - float(ti)) * np.log(1 - float(m))) for m, ti in zip(A, t)])
    ceB = np.mean([-(float(ti) * np.log(float(m)) + (1 - float(ti)) * np.log(1 - float(m))) for m, ti in zip(B, t)])
    ok = rho2A == rho2B == 1 and cov(A, t) > 0 and cov(B, t) > 0 and bB > bA and ceB > ceA
    report("N19", "calibrated vs squashed judge 0.5+0.1(t-0.5): both corr 1, Brier and CE differ (exact)", ok,
           f"corr^2 {rho2A},{rho2B}; mean expected Brier {bA}={float(bA):.4f} vs {bB}={float(bB):.4f}; "
           f"CE {ceA:.4f} vs {ceB:.4f} nats")

    # N20 exact counterexample (ii): ordering reversal, binary
    t = [Fr(1, 5), Fr(2, 5), Fr(3, 5), Fr(4, 5)]
    C = [Fr(1, 2) + Fr(1, 10) * (ti - Fr(1, 2)) for ti in t]  # (47,49,51,53)/100
    D = [Fr(1, 4), Fr(7, 20), Fr(13, 20), Fr(3, 4)]
    rC2 = cov(C, t) ** 2 / (cov(C, C) * cov(t, t))
    rD2 = cov(D, t) ** 2 / (cov(D, D) * cov(t, t))
    bC = sum(ebrier2(m, ti) for m, ti in zip(C, t)) / 4
    bD = sum(ebrier2(m, ti) for m, ti in zip(D, t)) / 4
    ok = rC2 == 1 and rD2 < 1 and cov(D, t) > 0 and bC > bD
    report("N20", "judge C corr 1 > judge D corr <1, yet C has worse Brier (exact)", ok,
           f"corr C=1, corr D^2={rD2} (corr {float(rD2)**0.5:.6f}); expected Brier C {bC}={float(bC):.4f} "
           f"> D {bD}={float(bD):.4f}")

    # N21 exact counterexample (iii): 6-point ordinal, RPS reversal
    P = [[Fr(x, 10) for x in row] for row in ((4, 3, 2, 1, 0, 0), (0, 0, 1, 2, 3, 4), (1, 2, 2, 2, 2, 1))]
    U = [Fr(1, 6)] * 6
    Cq = [[Fr(1, 10) * a + Fr(9, 10) * u for a, u in zip(row, U)] for row in P]
    Dq = [P[0], P[1], [Fr(x, 10) for x in (1, 2, 2, 2, 1, 2)]]
    mean = lambda d: sum(Fr(r) * d[r] for r in range(K))

    def rpsx(f, d):
        F = [sum(f[: j + 1]) for j in range(K - 1)]
        return sum(d[y] * sum((F[j] - (1 if y <= j else 0)) ** 2 for j in range(K - 1)) for y in range(K)) / (K - 1)
    tm = [mean(d) for d in P]
    cm, dm = [mean(d) for d in Cq], [mean(d) for d in Dq]
    rC2 = cov(cm, tm) ** 2 / (cov(cm, cm) * cov(tm, tm))
    rD2 = cov(dm, tm) ** 2 / (cov(dm, dm) * cov(tm, tm))
    rpsC = sum(rpsx(f, d) for f, d in zip(Cq, P)) / 3
    rpsD = sum(rpsx(f, d) for f, d in zip(Dq, P)) / 3
    rpsT = sum(rpsx(d, d) for d in P) / 3
    ok = rC2 == 1 and cov(cm, tm) > 0 and rD2 < 1 and rpsC > rpsD
    report("N21", "6-point ordinal: judge C (corr 1 of means) has worse mean expected RPS than judge D (corr<1)", ok,
           f"true means {[str(x) for x in tm]}; C means {[str(x) for x in cm]}; D means {[str(x) for x in dm]}; "
           f"corr D={float(rD2)**0.5:.6f}; RPS truth {rpsT}={float(rpsT):.5f}, C {rpsC}={float(rpsC):.5f}, D {rpsD}={float(rpsD):.5f}")

    # N25: the Pearson gradient with respect to the forecast means is orthogonal to the
    # constant vector (location) and to the centered forecast (scale): no first-order signal on level or scale.
    g25 = np.random.default_rng(25)
    worst = 0.0
    for _ in range(1000):
        n = int(g25.integers(3, 40))
        m, tt = g25.normal(size=n), g25.normal(size=n)
        z, zeta = (m - m.mean()) / m.std(), (tt - tt.mean()) / tt.std()
        rho = np.mean(z * zeta)
        grad = (zeta - rho * z) / (n * m.std())
        worst = max(worst, abs(grad.sum()), abs(grad @ (m - m.mean())))
    report("N25", "Pearson gradient in mean space is orthogonal to level (1) and scale (m - mean m) (1000 draws)",
           worst < TOL, f"max |<grad,1>|, |<grad,m-mbar>| = {worst:.2e}")

    # figure 3: counterexample (ii)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].bar(["judge C\n(squashed)", "judge D\n(calibrated, noisy)"], [1.0, float(rD2) ** 0.5], color=["C3", "C0"])
    ax[0].set_ylabel("Pearson correlation of forecast with\ntrue P(rating=1) (higher is better)")
    ax[0].set_ylim(0.9, 1.005)
    ax[1].bar(["judge C\n(squashed)", "judge D\n(calibrated, noisy)"], [float(bC), float(bD)], color=["C3", "C0"])
    ax[1].axhline(float(sum(2 * ti * (1 - ti) for ti in t) / 4), color="k", ls="--", lw=1, label="truth (floor)")
    ax[1].set_ylabel("Expected Brier score, mean over 4 items\n(lower is better; range 0 to 2)")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(True)
    fig.suptitle("Exact counterexample, synthetic, no human ratings: higher correlation, worse Brier", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig3_corr_vs_brier_counterexample.{ext}"), dpi=150)
    plt.close(fig)


# =============================================================== Q4: training
def softmax(U):
    U = U - U.max(1, keepdims=True)
    E = np.exp(U)
    return E / E.sum(1, keepdims=True)


def feats(x):
    return np.c_[np.ones_like(x), x]  # logits u_r(x) = a_r + b_r x, r=0..5 (r=0 fixed to 0)


def unpack(w):
    W = np.zeros((2, K))
    W[:, 1:] = w.reshape(2, K - 1)
    return W


def objective(w, X, y, kind):
    W = unpack(w)
    Q = softmax(X @ W)
    n = len(y)
    E = np.eye(K)[y]
    m = Q @ R
    if kind == "ce":
        f = -np.mean(np.log(Q[np.arange(n), y] + 1e-300))
        G = (Q - E) / n
    elif kind == "brier":
        D = Q - E
        f = np.mean(np.sum(D ** 2, 1))
        G = 2 * Q * (D - np.sum(Q * D, 1, keepdims=True)) / n
    elif kind == "mse":
        f = np.mean((m - y) ** 2)
        G = (2 * (m - y) / n)[:, None] * Q * (R[None, :] - m[:, None])
    elif kind == "pearson":
        sm, sy = m.std(), y.std()
        if sm < 1e-12:
            return 0.0, np.zeros_like(w)
        z, zeta = (m - m.mean()) / sm, (y - y.mean()) / sy
        rho = np.mean(z * zeta)
        f = -rho
        g = -(zeta - rho * z) / (n * sm)
        G = g[:, None] * Q * (R[None, :] - m[:, None])
    grad = (X.T @ G)[:, 1:].ravel()
    return f, grad


def truth(x, spec):
    s = 1 / (1 + np.exp(-(1.5 * x + 0.2)))
    B = stats.binom.pmf(R[None, :], 5, s[:, None])
    if spec == "well":
        return B
    s2 = 1 / (1 + np.exp(-(-1.0 + 0.5 * x)))
    return 0.6 * B + 0.4 * stats.binom.pmf(R[None, :], 5, s2[:, None])


def tilt_to_mean(Q, target):
    target = np.clip(target, 0.02, K - 1 - 0.02)
    lo, hi = np.full(len(Q), -30.0), np.full(len(Q), 30.0)
    L = np.log(Q + 1e-300)
    for _ in range(80):
        g = (lo + hi) / 2
        m = softmax(L + g[:, None] * R[None, :]) @ R
        lo, hi = np.where(m < target, g, lo), np.where(m < target, hi, g)
    return softmax(L + ((lo + hi) / 2)[:, None] * R[None, :])


def q4_training():
    Ns = [10, 30, 100, 300, 1000, 3000, 10000, 100000]
    inits = [0, 1, 2]
    data_seeds = [0, 1]
    xt = np.random.default_rng(999).uniform(-2, 2, 4000)
    Xt = feats(xt)
    rows = []
    for spec in ["well", "mis"]:
        Pt = truth(xt, spec)
        mut = Pt @ R
        for ds in data_seeds:
            g = np.random.default_rng(1000 * ds + (0 if spec == "well" else 7))
            xall = g.uniform(-2, 2, max(Ns))
            Pall = truth(xall, spec)
            yall = (g.random(len(xall))[:, None] > np.cumsum(Pall, 1)).sum(1)
            for N in Ns:
                x, y = xall[:N], yall[:N]
                X = feats(x)
                for kind in ["ce", "brier", "mse", "pearson"]:
                    for ini in inits:
                        w0 = np.random.default_rng(50 + ini).normal(size=2 * (K - 1))
                        if kind == "pearson":
                            ntr = int(round(0.7 * N))
                            res = minimize(objective, w0, args=(X[:ntr], y[:ntr], kind), jac=True, method="L-BFGS-B",
                                           options={"maxiter": 500})
                            # same Pearson fit on all N for the uncalibrated arm
                            resall = minimize(objective, w0, args=(X, y, kind), jac=True, method="L-BFGS-B",
                                              options={"maxiter": 500})
                            Qall = softmax(Xt @ unpack(resall.x))
                            rows.append((spec, ds, N, kind, ini, float(np.mean(tvd(Qall, Pt))),
                                         float(np.sqrt(np.mean((Qall @ R - mut) ** 2))), -resall.fun))
                            # recalibration on held-out 30%: least squares of y on m
                            Qtr = softmax(X[ntr:] @ unpack(res.x)) if N - ntr >= 2 else None
                            Qt = softmax(Xt @ unpack(res.x))
                            if Qtr is not None and np.std(Qtr @ R) > 1e-12:
                                mh = Qtr @ R
                                b, a = np.polyfit(mh, y[ntr:], 1)
                                Qc = tilt_to_mean(Qt, a + b * (Qt @ R))
                            else:
                                Qc = Qt
                            rows.append((spec, ds, N, "pearson+recal", ini, float(np.mean(tvd(Qc, Pt))),
                                         float(np.sqrt(np.mean((Qc @ R - mut) ** 2))), np.nan))
                        else:
                            res = minimize(objective, w0, args=(X, y, kind), jac=True, method="L-BFGS-B",
                                           options={"maxiter": 2000})
                            Q = softmax(Xt @ unpack(res.x))
                            rows.append((spec, ds, N, kind, ini, float(np.mean(tvd(Q, Pt))),
                                         float(np.sqrt(np.mean((Q @ R - mut) ** 2))), res.fun))
    with open(os.path.join(RES, "q4_training_tvd_vs_n.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["truth_spec", "data_seed", "n_train_ratings", "objective", "init_seed",
                    "test_mean_tvd_to_true_dist", "test_rmse_mean_vs_true_mean", "final_train_objective"])
        w.writerows(rows)
    arr = rows
    def stat(spec, kind, N):
        v = [r[5] for r in arr if r[0] == spec and r[3] == kind and r[2] == N]
        return np.median(v), np.min(v), np.max(v)
    def statm(spec, kind, N):
        v = [r[6] for r in arr if r[0] == spec and r[3] == kind and r[2] == N]
        return np.median(v)
    kinds = ["ce", "brier", "mse", "pearson", "pearson+recal"]
    labels = {"ce": "CE-trained", "brier": "Brier-trained", "mse": "MSE-trained (mean only)",
              "pearson": "Pearson-trained", "pearson+recal": "Pearson-trained + held-out affine mean recal."}
    for spec in ["well", "mis"]:
        for kind in kinds:
            s = ", ".join(f"N={N}:{stat(spec, kind, N)[0]:.3f}[{stat(spec, kind, N)[1]:.3f},{stat(spec, kind, N)[2]:.3f}]"
                          for N in Ns)
            print(f"  INFO Q4 {spec}-specified {labels[kind]} test mean TVD median[min,max] over 6 runs: {s}")
            print(f"  INFO Q4 {spec}-specified {labels[kind]} test RMSE of mean rating vs true mean (median): " +
                  ", ".join(f"N={N}:{statm(spec, kind, N):.3f}" for N in Ns))
    big = Ns[-1]
    ce_w, br_w = stat("well", "ce", big)[0], stat("well", "brier", big)[0]
    pe_w_lo, pe_w_hi = stat("well", "pearson", big)[1], stat("well", "pearson", big)[2]
    report("N22", "well-specified: CE- and Brier-ERM TVD to truth < 0.02 at N=1e5; Pearson-trained TVD stays large",
           ce_w < 0.02 and br_w < 0.02 and pe_w_lo > 0.05,
           f"N=1e5 median TVD CE {ce_w:.4f}, Brier {br_w:.4f}; Pearson range over runs [{pe_w_lo:.3f},{pe_w_hi:.3f}]")
    ce_m, br_m = stat("mis", "ce", big)[0], stat("mis", "brier", big)[0]
    report("N23", "misspecified: CE- and Brier-ERM plateau at a positive TVD (best-in-class != truth)",
           ce_m > 0.01 and br_m > 0.01,
           f"N=1e5 median TVD CE {ce_m:.4f}, Brier {br_m:.4f}, RPS not trained")
    rc = statm("well", "pearson+recal", big); pc = statm("well", "pearson", big)
    report("N24", "Pearson + held-out affine recalibration fixes the mean (RMSE small) but not the distribution",
           rc < 0.1 and stat("well", "pearson+recal", big)[0] > 2 * ce_w,
           f"N=1e5 RMSE of mean: Pearson {pc:.3f}, Pearson+recal {rc:.3f}; TVD Pearson+recal "
           f"{stat('well','pearson+recal',big)[0]:.3f} vs CE {ce_w:.4f}")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, spec, title in zip(axs, ["well", "mis"], ["Well-specified model class", "Misspecified model class"]):
        for i, kind in enumerate(kinds):
            med = [stat(spec, kind, N)[0] for N in Ns]
            lo = [stat(spec, kind, N)[1] for N in Ns]
            hi = [stat(spec, kind, N)[2] for N in Ns]
            ax.plot(Ns, med, marker="o", ms=3, label=labels[kind], color=f"C{i}")
            ax.fill_between(Ns, lo, hi, alpha=0.15, color=f"C{i}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, which="major"); ax.grid(True, which="minor", alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("Number of training ratings N (synthetic, one rating per item)")
    axs[0].set_ylabel("Held-out test mean total variation distance (TVD)\nto the true rating distribution (lower is better, 0-1)")
    axs[1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Synthetic data, no human ratings. Linear-logit softmax judge, L-BFGS-B from 3 random starts x 2 data seeds; "
                 "line = median, band = min-max over the 6 runs", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig2_tvd_vs_n.{ext}"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    q1_checks()
    q2_checks()
    q3_checks()
    q4_training()
    with open(os.path.join(RES, "checks.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["check_id", "name", "pass", "detail"])
        w.writerows(results)
    npass = sum(r[2] for r in results)
    print(f"SUMMARY {npass} PASS, {len(results) - npass} FAIL out of {len(results)} checks")
