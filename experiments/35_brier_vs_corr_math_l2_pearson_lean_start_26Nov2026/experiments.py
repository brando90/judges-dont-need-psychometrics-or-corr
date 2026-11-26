"""Numerical checks: Brier properness, scalar L2, and Pearson correlation.

Run from the repository root:

    python3 experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/experiments.py

Seed 0. Writes results.txt and figures/brier_vs_corr.png next to this file.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG_PATH = ROOT / "figures" / "brier_vs_corr.png"
RESULTS_PATH = ROOT / "results.txt"
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            if getattr(stream, "closed", False):
                continue
            stream.flush()


_results_file = open(RESULTS_PATH, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _results_file)

rng = np.random.default_rng(0)
K = 6
ys = np.arange(K)
E = np.eye(K)

def exp_brier(p, q):            # E_{Y~p} ||q - e_Y||^2
    return sum(p[y] * np.sum((q - E[y]) ** 2) for y in range(K))
def exp_rps(p, q):              # E_{Y~p} sum_k (F_q(k) - 1[Y<=k])^2, k=0..K-2
    Fq = np.cumsum(q)[:-1]
    return sum(p[y] * np.sum((Fq - (ys[:-1] >= y)) ** 2) for y in range(K))
def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))
def two_point(m):               # minimal-variance distribution on {0..5} with mean m
    m = float(np.clip(m, 0, K - 1)); lo = int(np.floor(m)); hi = min(lo + 1, K - 1)
    q = np.zeros(K); w = m - lo
    q[lo] += 1 - w; q[hi] += w
    return q

hr = lambda s: print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)

# ---------------------------------------------------------------- Part 1
hr("PART 1  Brier score is strictly proper and converges as N -> inf")
pstar = np.array([0.30, 0.05, 0.05, 0.10, 0.15, 0.35])     # bimodal, mean 2.85
bayes = 1 - np.sum(pstar ** 2)
print("p* =", pstar, " mean =", pstar @ ys, " Bayes Brier 1-sum p^2 =", round(bayes, 6))
cands = {
    "q = p* (truth)": pstar,
    "uniform": np.full(K, 1 / K),
    "two-point at mean 2.8": two_point(pstar @ ys),
    "point mass at 5": E[5],
    "slightly off p*": np.array([0.28, 0.07, 0.05, 0.10, 0.15, 0.35]),
    "random Dirichlet": rng.dirichlet(np.ones(K)),
}
print(f"{'candidate q':28s} {'E[BS(q)]':>10s} {'E[BS(p*)]+||q-p*||^2':>22s} {'abs diff':>10s}")
for name, q in cands.items():
    lhs = exp_brier(pstar, q); rhs = exp_brier(pstar, pstar) + np.sum((q - pstar) ** 2)
    print(f"{name:28s} {lhs:10.6f} {rhs:22.6f} {abs(lhs-rhs):10.1e}")
# numeric minimisation over the simplex
res = minimize(lambda z: exp_brier(pstar, np.exp(z) / np.exp(z).sum()), np.zeros(K), method="BFGS", options={"gtol": 1e-12})
qopt = np.exp(res.x) / np.exp(res.x).sum()
print("argmin_q E[BS(q)] (numerical, softmax-param) =", np.round(qopt, 5), " max|q-p*| =", f"{np.abs(qopt-pstar).max():.1e}")

print("\nMonte Carlo: q_hat_N = argmin empirical Brier = empirical frequencies p_hat_N")
Ns = [10, 100, 1000, 10_000, 100_000]; R = 200
conv = []
print(f"{'N':>7s} {'mean ||p_hat-p*||_2':>20s} {'sqrt(sum p(1-p)/N)':>19s} {'emp Brier(p_hat)':>17s} {'true E[BS(p_hat)]':>18s} {'Bayes':>8s}")
for N in Ns:
    errs, embs, tbs = [], [], []
    for _ in range(R):
        Y = rng.choice(K, size=N, p=pstar)
        ph = np.bincount(Y, minlength=K) / N
        errs.append(np.linalg.norm(ph - pstar))
        embs.append(np.mean(np.sum((ph[None, :] - E[Y]) ** 2, axis=1)))   # in-sample
        tbs.append(exp_brier(pstar, ph))                                    # population risk
    theo = np.sqrt(np.sum(pstar * (1 - pstar)) / N)
    conv.append((N, np.mean(errs), theo, np.mean(embs), np.mean(tbs)))
    print(f"{N:7d} {np.mean(errs):20.5f} {theo:19.5f} {np.mean(embs):17.5f} {np.mean(tbs):18.5f} {bayes:8.5f}")
# check empirical minimiser == frequency vector for one sample
Y = rng.choice(K, size=500, p=pstar); ph = np.bincount(Y, minlength=K) / 500
r = minimize(lambda z: np.mean(np.sum((np.exp(z)/np.exp(z).sum() - E[Y]) ** 2, axis=1)), np.zeros(K), method="BFGS", options={"gtol": 1e-12})
print("N=500: numerical argmin of empirical Brier vs p_hat: max abs diff =",
      f"{np.abs(np.exp(r.x)/np.exp(r.x).sum() - ph).max():.1e}")

# ---------------------------------------------------------------- Part 2
hr("PART 2  L2 loss vs 'Gaussian noise'")
pb = np.array([0.5, 0, 0, 0, 0, 0.5])
r = minimize_scalar(lambda c: pb @ (ys - c) ** 2, bounds=(-10, 10), method="bounded", options={"xatol": 1e-12})
print(f"P(Y=0)=P(Y=5)=0.5 (no mass at 2.5).  argmin_c E[(Y-c)^2] = {r.x:.6f}  (E[Y] = {pb@ys})")
print(f"  min risk = {r.fun:.4f} = Var(Y) = {pb@(ys-2.5)**2:.4f}")
Yb = rng.choice(K, size=100_000, p=pb)
print(f"  from 100k samples: argmin of empirical L2 = sample mean = {Yb.mean():.4f}")
res = minimize(lambda z: exp_brier(pb, np.exp(z)/np.exp(z).sum()), np.zeros(K), method="BFGS", options={"gtol": 1e-12})
qb = np.exp(res.x)/np.exp(res.x).sum()
print("  argmin one-hot L2 (Brier) =", np.round(qb, 4), " -> recovers the full bimodal distribution")
print("  scalar L2 answer 2.5 read as a distribution (two-point) =", two_point(2.5), " (all mass where p*=0)")
print(f"  Brier of p* = {exp_brier(pb,pb):.4f}; Brier of 'everything at 2.5'-style two-point = {exp_brier(pb,two_point(2.5)):.4f}")
# Gaussian MLE with free sigma: also mean 2.5, sd 2.5 -> mode at 2.5 where p* has zero mass
print(f"  Gaussian MLE fit: mu = {Yb.mean():.3f}, sigma = {Yb.std():.3f}  -> density peaks at 2.5 (where truth has 0 mass)")
# binary
print("\nBinary Y in {0,1}: scalar L2 (q-y)^2 vs 2-class Brier (q-y)^2+((1-q)-(1-y))^2")
for q in [0.1, 0.37, 0.8]:
    for y in [0, 1]:
        l2 = (q - y) ** 2; bs = (q - y) ** 2 + ((1 - q) - (1 - y)) ** 2
        print(f"  q={q:.2f} y={y}: L2={l2:.4f}  Brier={bs:.4f}  ratio={bs/l2:.4f}")
pbin = 0.3
r = minimize_scalar(lambda c: pbin*(1-c)**2 + (1-pbin)*c**2, bounds=(0, 1), method="bounded")
print(f"  P(Y=1)=0.3: argmin scalar L2 = {r.x:.6f} = P(Y=1) -> for binary, the mean IS the distribution")

# ---------------------------------------------------------------- Part 3
hr("PART 3a  Standardized identity and MSE decomposition")
n = 10_000
X = rng.normal(1.3, 2.0, n); Yc = 0.6 * X + rng.gamma(2.0, 1.5, n) - 4
mx, my, sx, sy = X.mean(), Yc.mean(), X.std(), Yc.std()   # population (ddof=0)
rho = pearson(X, Yc)
Xt, Yt = (X - mx) / sx, (Yc - my) / sy
lhs, rhs = np.mean((Xt - Yt) ** 2), 2 * (1 - rho)
print(f"rho = {rho:.6f}; E[(X~-Y~)^2] = {lhs:.10f}; 2(1-rho) = {rhs:.10f}; diff = {abs(lhs-rhs):.1e}")
mse = np.mean((X - Yc) ** 2)
dec = (mx - my) ** 2 + (sx - sy) ** 2 + 2 * sx * sy * (1 - rho)
print(f"MSE = {mse:.10f};  (mx-my)^2 + (sx-sy)^2 + 2 sx sy (1-rho) = {dec:.10f};  diff = {abs(mse-dec):.1e}")
print(f"   terms: bias^2 = {(mx-my)**2:.4f}, scale^2 = {(sx-sy)**2:.4f}, 'correlation' = {2*sx*sy*(1-rho):.4f}")

hr("PART 3b  Affine invariance: Pearson is blind to shift + scale miscalibration")
I = 200
P = np.array([rng.dirichlet(rng.uniform(0.3, 3.0, K)) for _ in range(I)])   # varied p*(y|x)
m = P @ ys
Ysamp = np.array([rng.choice(K, p=p) for p in P])
Ymany = np.array([rng.choice(K, size=20, p=p) for p in P])   # 20 ratings/item
judges = {"Bayes  y*=E[Y|x]": m, "bad1   0.1*y*+4": 0.1 * m + 4.0, "bad2   5*y*-10": 5 * m - 10}
rows3b = {}
print(f"{'judge':18s} {'rho(.,E[Y|x])':>14s} {'rho(.,Y 1 sample)':>18s} {'rho(.,Y 20 samp)':>17s} {'E[MSE] vs Y':>12s} {'E[Brier]':>9s} {'E[RPS]':>8s}")
for name, f in judges.items():
    r1, r2 = pearson(f, m), pearson(f, Ysamp)
    r3 = pearson(np.repeat(f, 20), Ymany.ravel())
    emse = np.mean([P[i] @ (ys - f[i]) ** 2 for i in range(I)])   # exact expected MSE
    Q = np.array([two_point(v) for v in f])                         # same scalar->dist map for all
    eb = np.mean([exp_brier(P[i], Q[i]) for i in range(I)])
    er = np.mean([exp_rps(P[i], Q[i]) for i in range(I)])
    rows3b[name] = (r1, r2, emse, eb, er)
    print(f"{name:18s} {r1:14.10f} {r2:18.10f} {r3:17.10f} {emse:12.4f} {eb:9.4f} {er:8.4f}")
print("  (Brier/RPS for scalar judges use the same scalar->two-point distribution map, clipped to [0,5])")
ebA = np.mean([exp_brier(P[i], P[i]) for i in range(I)]); erA = np.mean([exp_rps(P[i], P[i]) for i in range(I)])
print(f"  reference: judge that outputs full p*(y|x): E[Brier] = {ebA:.4f}, E[RPS] = {erA:.4f}")
# Pearson's population maximiser with noisy Y: rho(f,Y) = rho(f, E[Y|x]) * sd(E[Y|x]) / sd(Y)
Ypop_var = np.mean([P[i] @ (ys - m.mean()) ** 2 for i in range(I)])
print(f"  ceiling: max_f rho(f,Y) = sd(E[Y|x])/sd(Y) = {m.std()/np.sqrt(Ypop_var):.4f}, attained by EVERY a*E[Y|x]+b, a>0")
fpert = m + rng.normal(0, 0.5, I)
print(f"  perturbed judge E[Y|x]+N(0,.5^2): rho vs E[Y|x] = {pearson(fpert, m):.4f} (<1)")

hr("PART 3c  Same mean, different spread: Pearson & scalar MSE are blind, Brier/RPS are not")
# Judge A outputs p* (spread); Judge B outputs the collapsed two-point distribution with the SAME mean.
PA = []
for i in range(I):
    if i < I // 2:              # high-uncertainty items: 50/50 on two far-apart ratings
        a, b = sorted(rng.choice(K, size=2, replace=False))
        while b - a < 3:
            a, b = sorted(rng.choice(K, size=2, replace=False))
        p = np.zeros(K); p[a] += 0.5; p[b] += 0.5
    else:                       # moderate items: random Dirichlet
        p = rng.dirichlet(np.ones(K))
    PA.append(p)
PA = np.array(PA); mA = PA @ ys
PB = np.array([two_point(v) for v in mA]); mB = PB @ ys
assert np.allclose(mA, mB)
Yc_s = np.array([rng.choice(K, p=p) for p in PA])
out = {}
for name, Q, mq in [("Judge A (true p*)", PA, mA), ("Judge B (collapsed, same mean)", PB, mB)]:
    out[name] = (pearson(mq, Yc_s), pearson(mq, mA),
                 np.mean([PA[i] @ (ys - mq[i]) ** 2 for i in range(I)]),
                 np.mean([exp_brier(PA[i], Q[i]) for i in range(I)]),
                 np.mean([exp_rps(PA[i], Q[i]) for i in range(I)]),
                 np.mean(np.sum((Q - E[Yc_s]) ** 2, axis=1)))
print(f"max |mean_A - mean_B| over items = {np.abs(mA-mB).max():.1e}")
print(f"{'judge':32s} {'rho(mean,Y)':>12s} {'rho(mean,E[Y|x])':>17s} {'E[MSE]':>8s} {'E[Brier]':>9s} {'E[RPS]':>8s} {'sampled Brier':>14s}")
for k, v in out.items():
    print(f"{k:32s} {v[0]:12.6f} {v[1]:17.6f} {v[2]:8.4f} {v[3]:9.4f} {v[4]:8.4f} {v[5]:14.4f}")
# the simple 2-group version from the brief
pHi = np.array([0.5, 0, 0, 0, 0.5, 0]); pLo = E[2]
print(f"\nToy: item with p*=50/50 on {{0,4}} (mean 2) vs judge saying 100% at 2 (mean 2):")
print(f"  Brier: truth {exp_brier(pHi,pHi):.3f} vs collapsed {exp_brier(pHi,pLo):.3f};  RPS: truth {exp_rps(pHi,pHi):.3f} vs collapsed {exp_rps(pHi,pLo):.3f}; means identical -> rho, MSE identical")

def _grid(axis, log=False):
    axis.grid(True)
    if log:
        axis.minorticks_on()
        axis.grid(True, which="minor", alpha=0.3)


# ---------------------------------------------------------------- Figure
fig, ax = plt.subplots(2, 2, figsize=(13.2, 10.2))
a = ax[0, 0]
N_, e_, t_, _, tb_ = map(np.array, zip(*conv))
a.loglog(N_, e_, "o-", label=r"mean $\|\hat p_N - p^*\|_2$ (200 replications)")
a.loglog(N_, t_, "k--", label=r"$\sqrt{\sum_k p_k(1-p_k)/N}$")
a.loglog(N_, tb_ - bayes, "s-", label=r"mean excess population Brier $E[BS(\hat p_N)] - BS^*$")
a.set_xlabel("Number of i.i.d. ratings N from one fixed synthetic p* (K = 6)")
a.set_ylabel("Mean L2 distance or excess expected Brier (lower is closer to p*)")
a.set_title("(1) Empirical-frequency Brier minimizer approaches p*")
a.legend(fontsize=8)
_grid(a, log=True)
a = ax[0, 1]
w = 0.38
a.bar(ys, pb, w, label="true p* (Brier minimizer)")
a.axvline(2.5, color="r", lw=2, ls="--", label="scalar L2 minimizer = 2.5 (the mean)")
a.set_xticks(ys)
a.set_xlabel("Ordinal rating y in {0, 1, 2, 3, 4, 5}")
a.set_ylabel("Probability under the synthetic bimodal law")
a.set_title("(2) Bimodal law: scalar L2 returns the mean")
a.legend(fontsize=8)
_grid(a)
a = ax[1, 0]
names = list(rows3b); x = np.arange(len(names))
a.bar(x - 0.2, [rows3b[k][0] for k in names], 0.4, label=r"Pearson correlation vs $E[Y|x]$")
a.bar(x + 0.2, [rows3b[k][1] for k in names], 0.4, label="Pearson correlation vs one sampled Y")
a2 = a.twinx()
a2.plot(x, [rows3b[k][2] for k in names], "ro-", label="expected MSE versus Y (right axis)")
a2.set_ylim(0, 10.5)
a2.set_ylabel("Expected MSE of the scalar forecast versus Y (lower is better)")
a.set_xticks(x)
a.set_xticklabels(["Bayes E[Y|x]", "0.1 E[Y|x] + 4", "5 E[Y|x] - 10"])
a.set_ylim(0, 1.15)
a.set_xlabel("Scalar forecast on 200 synthetic items (one seeded draw)")
a.set_ylabel("Pearson correlation (higher is closer to 1)")
a.set_title("(3b) Positive affine maps keep correlation and change MSE")
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(
    h1 + h2, l1 + l2, fontsize=8, loc="upper center",
    bbox_to_anchor=(0.5, -0.34), ncol=2, frameon=True,
)
_grid(a)
_grid(a2)
a = ax[1, 1]
mets = ["Pearson rho\n(mean, Y)", "E[MSE]", "E[Brier]", "E[RPS]\n(unnormalized)"]
A = out["Judge A (true p*)"]; B = out["Judge B (collapsed, same mean)"]
va = [A[0], A[2], A[3], A[4]]; vb = [B[0], B[2], B[3], B[4]]
x = np.arange(4)
a.bar(x - 0.2, va, 0.4, label="Judge A: outputs p*")
a.bar(x + 0.2, vb, 0.4, label="Judge B: two-point collapse, same mean")
for i in range(4):
    a.text(x[i]-0.2, va[i], f"{va[i]:.3f}", ha="center", va="bottom", fontsize=8)
    a.text(x[i]+0.2, vb[i], f"{vb[i]:.3f}", ha="center", va="bottom", fontsize=8)
a.set_xticks(x)
a.set_xticklabels(mets)
a.set_xlabel("Metric on the same 200 items (rho: higher better; MSE, Brier, RPS: lower better)")
a.set_ylabel("Value of the metric named on the horizontal axis")
a.set_title("(3c) Same conditional mean: rho and MSE tie")
a.legend(fontsize=8)
_grid(a)
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved figure -> {FIG_PATH.relative_to(ROOT).as_posix()}")
sys.stdout = sys.__stdout__
_results_file.close()
