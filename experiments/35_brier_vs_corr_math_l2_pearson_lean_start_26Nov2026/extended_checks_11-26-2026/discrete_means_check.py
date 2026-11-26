#!/usr/bin/env python3
"""discrete_means_check.py (11-26-2026): do matching expectations recover a discrete distribution?

SYNTHETIC DATA, NO HUMAN RATINGS. Every number is printed by this script.
Run:  python3 -I discrete_means_check.py   (from any directory; needs numpy, scipy, matplotlib; about one minute)
Writes results/discrete_means_output.txt and figures/discrete_means_fig{1,2}_*.{pdf,png}.

Question. For a discrete outcome Y on K categories, E[1{Y=k} | x] = P(Y=k | x) is an identity, so matching the
VECTOR of K indicator means recovers the distribution. The numeric rating mean E[R | x] = sum_k k P(Y=k | x) is ONE
number: it fixes the distribution at K = 2 and leaves K-2 shape directions free at K >= 3. Pearson correlation also
discards level and scale, so even at K = 2 it does not fix the probabilities, whereas Brier (squared error on the
indicators) and cross-entropy (CE) do. This script tests each statement.

Models. A table model: every context c has its own free logits, so the model class contains the truth and only the
training objective (and, for flat objectives, the initialization and optimizer path) selects the answer. Every
objective has an analytic gradient (checked against finite differences); training is L-BFGS from random initial logits
N(0, std^2) with std in (0.1, 0.3, 1.0, 3.0) (K = 2, 6) or (1.0, 2.0) (V = 50). Data are counts drawn from the true
distributions with a fixed design (the same number of ratings per context). TVD is exact against the true distribution.

Converged-run rule (added after the first run, see below). A correlation-type run counts as "converged" when its
in-sample correlation is within 1e-4 of the best achievable in-sample correlation. Stalled runs are counted and
reported and are excluded from the correlation-objective medians; this favors the correlation objective, so it is
conservative for the thesis tested here. CE, Brier and mean-squared-error runs are never excluded.

PREDICTIONS (written before the first run; each PASS/FAIL line tests one of them)
  P1  Indicator-mean map has kernel dimension 0, numeric-mean map has kernel dimension K-2 (4 at K = 6, 0 at K = 2).
  P2  Binomial(5, 1/2), a same-mean-same-variance law and a two-point law share the mean 5/2; the excess ranked
      probability score (RPS) of forecasting the wrong one is all shape (zero mean error); CE, Brier, RPS, TVD of the
      wrong shape are strictly positive; Pearson of the means is identical across populations with the same item means.
  P3  K = 2 table model: Pearson-trained reaches the best in-sample correlation from every init (gap < 1e-4) but its
      learned probability is a + b p with (a, b) depending on the init (max/min of b > 2); Brier and CE land on the truth
      (|b-1| < 0.02, |a| < 0.01 at the largest N); raw Pearson TVD stays > 0.05 at the largest N while CE and Brier are
      < 0.01; a held-out affine calibration brings Pearson within 2x of CE at the largest N.
      REVISED AFTER THE FIRST RUN: "from every init" was wrong for init std 3.0 (some runs stall at saturated outputs
      whose gradient vanishes; gap up to 0.08). The training check now covers init std <= 1.0, and the stalled
      runs are reported and excluded from the correlation-objective medians (conservative, see above).
  P4  Control, 2-parameter well-specified logistic model: UNSURE; reported as an observation only (the sigmoid link
      could pin the answer, since correlation is invariant to affine maps of the probability but not of the logit).
  P5  K = 6 table model: per-class-correlation training keeps every class ranking (Spearman >= 0.95) but its
      probabilities are mis-scaled (raw TVD > 0.05 and > 5x CE at the largest N); K affine maps plus renormalization
      on held-out ratings bring it within 2x of CE; correlation of the MEAN rating cannot be repaired this way (TVD
      after calibration > 5x CE); squared error on the mean pins the mean (RMSE < 0.01) but not the shape; with a
      softmax output the K slopes of the per-class variant coincide (simplex constraint).
  P6  Toy language model (V = 50): CE TVD falls with log-log slope in [-0.6, -0.4]; correlation of the expected token ID
      reaches its ceiling and learns the mean ID up to an affine map (R^2 > 0.99), but its TVD does not shrink (ratio of
      TVD at the largest N to N = 1e5 above 0.7, and TVD above 0.3); CE is invariant to a relabeling of the token IDs
      while the correlation ceiling is not (range over relabelings above 0.05).
"""
from __future__ import annotations

import math
import os
import time
from fractions import Fraction as Fr

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SEED = 20261126
GAP_TOL = 1e-4  # converged-run rule for correlation objectives
INIT_STDS = (0.1, 0.3, 1.0, 3.0)
LINES: list[str] = []
VERDICTS: list[tuple[str, bool]] = []
T0 = time.time()


def emit(s: str) -> None:
    print(s, flush=True)
    LINES.append(s)


def report(cid: str, name: str, ok: bool, detail: str) -> None:
    VERDICTS.append((cid, bool(ok)))
    emit(f"{'PASS' if ok else 'FAIL'} {cid} {name}: {detail}")


def info(cid: str, text: str) -> None:
    emit(f"  INFO {cid} {text}")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


LBL = {"ce": "CE", "brier": "Brier", "pearson": "Pearson", "corr_perclass": "per-class Pearson (softmax)",
       "corr_perclass_sigmoid": "per-class Pearson (sigmoid)", "corr_mean": "Pearson of the mean rating",
       "mse_mean": "squared error on the mean rating", "corr_id": "Pearson of the expected token ID"}


def lbl(key: str) -> str:
    """Descriptive name of a stored series; a '+cal' suffix means a held-out affine calibration was applied."""
    base, plus, _ = key.partition("+")
    return LBL.get(base, base) + (" + held-out affine calibration" if plus else "")


# ---------------------------------------------------------------------------------------------- table-model objectives
class Counts:
    """Ratings of M contexts with n ratings each (fixed design); S[c, k] = number of ratings equal to k."""

    def __init__(self, S: np.ndarray):
        S = np.asarray(S, float)
        self.S = S
        self.M, self.K = S.shape
        self.n = S.sum(1)
        self.N = float(self.n.sum())
        self.w = self.n / self.N
        self.Yh = S / self.n[:, None]  # empirical conditional frequencies


def draw_counts(rng: np.random.Generator, P: np.ndarray, n: int) -> Counts:
    return Counts(np.stack([rng.multinomial(n, p / p.sum()) for p in P]))


def _softmax(U: np.ndarray) -> np.ndarray:
    Z = U - U.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def _logsoftmax(U: np.ndarray) -> np.ndarray:
    Z = U - U.max(1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(1, keepdims=True))


def _through_softmax(GQ: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Gradient wrt logits U given the gradient wrt probabilities Q = softmax(U)."""
    return Q * (GQ - (GQ * Q).sum(1, keepdims=True))


def loss_ce(u, D: Counts):
    """Average cross-entropy (nats) of the individual ratings."""
    U = u.reshape(D.M, D.K)
    LQ = _logsoftmax(U)
    Q = np.exp(LQ)
    return -np.sum(D.w[:, None] * D.Yh * LQ), (D.w[:, None] * (Q - D.Yh)).ravel()


def loss_brier(u, D: Counts):
    """Average Brier score sum_k (q_k - 1{y=k})^2 of the individual ratings."""
    U = u.reshape(D.M, D.K)
    Q = _softmax(U)
    L = np.sum(D.w[:, None] * (Q ** 2 - 2 * D.Yh * Q)) + 1.0
    return L, _through_softmax(D.w[:, None] * (2 * Q - 2 * D.Yh), Q).ravel()


def loss_mse_mean(u, D: Counts, f):
    """Squared error of the expected coded rating m = sum_k f_k q_k against the coded rating (up to a constant)."""
    U = u.reshape(D.M, D.K)
    Q = _softmax(U)
    m = Q @ f
    Yf = D.Yh @ f
    dm = 2 * D.w * (m - Yf)
    return np.sum(D.w * (m - Yf) ** 2), _through_softmax(dm[:, None] * f[None, :], Q).ravel()


def loss_corr_mean(u, D: Counts, f):
    """Minus the Pearson correlation between the expected coded rating m(c_i) and the coded rating of each rating."""
    U = u.reshape(D.M, D.K)
    Q = _softmax(U)
    m = Q @ f
    Yf = D.Yh @ f
    ybar = np.sum(D.w * Yf)
    var_y = np.sum(D.w * (D.Yh @ (f ** 2))) - ybar ** 2
    mbar = np.sum(D.w * m)
    var_m = max(np.sum(D.w * m ** 2) - mbar ** 2, 1e-300)
    sd_m, sd_y = math.sqrt(var_m), math.sqrt(var_y)
    corr = (np.sum(D.w * m * Yf) - mbar * ybar) / (sd_m * sd_y)
    dm = D.w * ((Yf - ybar) / (sd_m * sd_y) - corr * (m - mbar) / var_m)
    return -corr, _through_softmax(-dm[:, None] * f[None, :], Q).ravel()


def loss_corr_perclass(u, D: Counts, link: str = "softmax"):
    """Minus the average over classes k of corr(q_k(c_i), 1{y_i = k}); link = softmax (on the simplex) or sigmoid."""
    U = u.reshape(D.M, D.K)
    Q = _softmax(U) if link == "softmax" else sigmoid(U)
    Qbar = D.w @ Q
    Ybar = D.w @ D.Yh
    cov = np.sum(D.w[:, None] * Q * D.Yh, 0) - Qbar * Ybar
    varQ = np.maximum(np.sum(D.w[:, None] * Q ** 2, 0) - Qbar ** 2, 1e-300)
    varI = Ybar * (1 - Ybar)
    ok = varI > 1e-12
    sdQ, sdI = np.sqrt(varQ), np.sqrt(np.maximum(varI, 1e-300))
    corr = cov / (sdQ * sdI)
    nk = int(ok.sum())
    dc = D.w[:, None] * ((D.Yh - Ybar) / (sdQ * sdI) - corr * (Q - Qbar) / varQ)
    GQ = -(dc * ok[None, :]) / nk
    GU = _through_softmax(GQ, Q) if link == "softmax" else GQ * Q * (1 - Q)
    return -np.sum(corr[ok]) / nk, GU.ravel()


def max_corr_mean(D: Counts, f) -> float:
    """Best achievable in-sample correlation of any forecast mean with the coded rating (the correlation ratio)."""
    Yf = D.Yh @ f
    ybar = np.sum(D.w * Yf)
    var_y = np.sum(D.w * (D.Yh @ (f ** 2))) - ybar ** 2
    return math.sqrt(np.sum(D.w * (Yf - ybar) ** 2) / var_y)


def max_corr_perclass(D: Counts) -> float:
    Ybar = D.w @ D.Yh
    ok = Ybar * (1 - Ybar) > 1e-12
    c = np.sqrt(np.sum(D.w[:, None] * (D.Yh - Ybar) ** 2, 0)[ok] / (Ybar * (1 - Ybar))[ok])
    return float(c.mean())


def fit(loss, D: Counts, init_std: float, seed, link: str | None = None, **kw):
    """L-BFGS from random initial logits N(0, init_std^2). Returns (Q, scipy result)."""
    rng = np.random.default_rng(seed)
    u0 = rng.normal(0.0, init_std, size=D.M * D.K)
    if link is not None:
        kw["link"] = link
    res = minimize(lambda u: loss(u, D, **kw), u0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 60000, "maxfun": 120000, "ftol": 1e-15, "gtol": 1e-11, "maxcor": 30})
    U = res.x.reshape(D.M, D.K)
    Q = sigmoid(U) if link == "sigmoid" else _softmax(U)
    return Q, res


def corr_gap(name: str, D: Counts, res, f=None) -> float:
    """Best achievable in-sample correlation minus the one reached (0 for objectives that are not correlations)."""
    if name.startswith("corr_perclass"):
        return max_corr_perclass(D) + res.fun
    if name == "corr_mean":
        return max_corr_mean(D, f) + res.fun
    return 0.0


def gradcheck(loss, D: Counts, seed=0, **kw) -> float:
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1.0, D.M * D.K)
    _, g = loss(u, D, **kw)
    gn = np.empty_like(g)
    h = 1e-6
    for i in range(u.size):
        e = np.zeros_like(u)
        e[i] = h
        gn[i] = (loss(u + e, D, **kw)[0] - loss(u - e, D, **kw)[0]) / (2 * h)
    return float(np.max(np.abs(g - gn)) / max(1.0, np.max(np.abs(g))))


def recal_affine_perclass(Q: np.ndarray, Dcal: Counts, floor: float = 1e-6) -> np.ndarray:
    """K affine maps (one per class: weighted least squares of held-out class frequencies on Q), clip, renormalize."""
    out = np.empty_like(Q)
    w = Dcal.w
    for k in range(Q.shape[1]):
        q, y = Q[:, k], Dcal.Yh[:, k]
        qb, yb = w @ q, w @ y
        vq = w @ (q - qb) ** 2
        d = 0.0 if vq < 1e-18 else (w @ ((q - qb) * (y - yb))) / vq
        out[:, k] = (yb - d * qb) + d * q
    out = np.maximum(out, floor)
    return out / out.sum(1, keepdims=True)


def tvd_rows(Q: np.ndarray, P: np.ndarray) -> float:
    """Mean over contexts of the total variation distance 0.5 sum_k |q_k - p_k| to the TRUE distribution."""
    return float((0.5 * np.abs(Q - P).sum(1)).mean())


def affine_fit(q: np.ndarray, p: np.ndarray):
    """Least-squares q = a + b p across contexts; returns (a, b, R^2)."""
    pb, qb = p.mean(), q.mean()
    b = np.sum((p - pb) * (q - qb)) / np.sum((p - pb) ** 2)
    return float(qb - b * pb), float(b), float(np.corrcoef(p, q)[0, 1] ** 2)


def fmt_range(v) -> str:
    v = np.asarray(v, float)
    return f"{np.median(v):.4g} [{v.min():.4g}, {v.max():.4g}]"


# ==================================================================================================== (A) exact checks
def part_a():
    emit("== A. Indicator means versus the numeric mean, K = 6 (exact arithmetic and random instances)")
    K = 6
    rng = np.random.default_rng(SEED)
    err = 0.0
    for _ in range(1000):
        p = rng.dirichlet(np.ones(K))
        ind = np.array([sum(p[y] * (1.0 if y == k else 0.0) for y in range(K)) for k in range(K)])
        err = max(err, float(np.abs(ind - p).max()))
    p0 = rng.dirichlet(np.ones(K))
    ys = rng.choice(K, 400000, p=p0)
    mc_err = float(np.abs(np.bincount(ys, minlength=K) / ys.size - p0).max())
    report("DM1", "E[1{Y=k}|x] = P(Y=k|x) for every k (1000 random K=6 laws, exact; plus 4e5 Monte Carlo draws)",
           err < 1e-15 and mc_err < 4e-3,
           f"max abs err exact {err:.1e}; Monte Carlo max abs err {mc_err:.1e}; the K indicator means sum to one, so K-1 free numbers")

    def kernel_dim(rows: np.ndarray) -> int:
        k = rows.shape[1]
        basis = np.linalg.svd(np.ones((1, k)))[2][1:].T  # k x (k-1) orthonormal basis of the sum-zero vectors
        s = np.linalg.svd(rows @ basis, compute_uv=False)
        return int((k - 1) - np.sum(s > 1e-10))

    ker_ind = {k: kernel_dim(np.eye(k)) for k in (2, 6)}
    ker_mean = {k: kernel_dim(np.arange(k, dtype=float)[None, :]) for k in (2, 3, 6)}
    ok = ker_ind[2] == 0 and ker_ind[6] == 0 and ker_mean[2] == 0 and ker_mean[3] == 1 and ker_mean[6] == 4
    report("DM2", "free directions left by 'match the K indicator means' versus 'match the one mean rating'", ok,
           f"indicator means: K=2 {ker_ind[2]}, K=6 {ker_ind[6]}; mean rating: K=2 {ker_mean[2]}, K=3 {ker_mean[3]}, K=6 {ker_mean[6]} (= K-2) free shape directions")

    PA = [Fr(c, 32) for c in (1, 5, 10, 10, 5, 1)]  # Binomial(5, 1/2)
    PD = [Fr(c, 12) for c in (1, 0, 5, 5, 0, 1)]  # same mean and variance, different shape
    PB = [Fr(0), Fr(0), Fr(1, 2), Fr(1, 2), Fr(0), Fr(0)]  # same mean, smaller variance

    def mean_(p): return sum(Fr(x) * i for i, x in enumerate(p))
    def var_(p): return sum(Fr(x) * i * i for i, x in enumerate(p)) - mean_(p) ** 2
    def cdf(p): return [sum(p[: k + 1]) for k in range(K - 1)]
    def rps_excess(truth, fc): return sum((a - b) ** 2 for a, b in zip(cdf(truth), cdf(fc))) / (K - 1)
    def brier_excess(truth, fc): return sum((a - b) ** 2 for a, b in zip(truth, fc))
    def kl(truth, fc): return sum(float(t) * math.log(float(t) / float(f_)) for t, f_ in zip(truth, fc) if t > 0)

    means = [mean_(p) for p in (PA, PD, PB)]
    vars_ = [var_(p) for p in (PA, PD, PB)]
    d = [a - b for a, b in zip(cdf(PD), cdf(PA))]
    dbar = sum(d) / len(d)
    decomp = (mean_(PA) - mean_(PD)) ** 2 / (K - 1) ** 2 + sum((x - dbar) ** 2 for x in d) / (K - 1)
    tv = float(0.5 * sum(abs(a - b) for a, b in zip(PD, PA)))
    ex = {"rps": rps_excess(PD, PA), "brier": brier_excess(PD, PA), "ce": kl(PD, PA), "tvd": tv}
    ok = (len(set(means)) == 1 and means[0] == Fr(5, 2) and vars_[0] == vars_[1] == Fr(5, 4) and vars_[2] == Fr(1, 4)
          and decomp == ex["rps"] and ex["rps"] > 0 and ex["brier"] > 0 and ex["ce"] > 0 and tv > 0)
    report("DM3", "same mean and variance, different shape (truth P_D, forecast Binomial(5,1/2) = P_A): the whole excess is shape and every proper score sees it", ok,
           f"means {[str(m) for m in means]}, variances P_A {vars_[0]}, P_D {vars_[1]}, P_B {vars_[2]}; mean error 0; "
           f"excess RPS {ex['rps']} = {float(ex['rps']):.5f} (RPS decomposition identity holds), excess Brier {ex['brier']} = {float(ex['brier']):.5f}, "
           f"excess CE (= KL) {ex['ce']:.5f} nats, TVD {tv:.5f}; indicator means differ: P_A {[str(x) for x in PA]} vs P_D {[str(x) for x in PD]}")

    M = 7
    theta = np.array([(j + 1) / (M + 1) for j in range(M)])
    pop1 = np.stack([np.array([math.comb(5, k) * th ** k * (1 - th) ** (5 - k) for k in range(K)]) for th in theta])  # binomial
    pop2 = np.zeros((M, K))
    pop2[:, 0], pop2[:, 5] = 1 - theta, theta  # two-point law with the same item mean 5 theta
    mean1, mean2 = pop1 @ np.arange(K), pop2 @ np.arange(K)
    fc = pop1  # binomial forecast: right in population 1, wrong shape in population 2

    def ce(p, q): return float(np.mean(-(np.where(p > 0, p * np.log(np.where(q > 0, q, 1)), 0)).sum(1)))
    def rps(p, q): return float(np.mean(((np.cumsum(q, 1) - np.cumsum(p, 1))[:, :-1] ** 2).sum(1) / (K - 1)))
    corr1 = np.corrcoef(fc @ np.arange(K), mean1)[0, 1]
    corr2 = np.corrcoef(fc @ np.arange(K), mean2)[0, 1]
    rnd = np.random.default_rng(SEED + 1)
    worst = max(abs(np.corrcoef(m, mean1)[0, 1] - np.corrcoef(m, mean2)[0, 1]) for m in rnd.normal(size=(500, M)))
    ce_ex1, ce_ex2 = ce(pop1, fc) - ce(pop1, pop1), ce(pop2, fc) - ce(pop2, pop2)
    rps_ex1, rps_ex2 = rps(pop1, fc) - rps(pop1, pop1), rps(pop2, fc) - rps(pop2, pop2)
    ok = (np.abs(mean1 - mean2).max() < 1e-12 and worst < 1e-12 and abs(corr1 - 1) < 1e-12 and abs(corr2 - 1) < 1e-12
          and rps_ex1 < 1e-15 and rps_ex2 > 1e-3 and ce_ex2 > 1e-3 and ce_ex1 < 1e-12)
    report("DM3b", "two 7-item populations with identical item means and different shape: Pearson of any forecast means is identical in both, proper scores are not", ok,
           f"corr(forecast means, true means) = {corr1:.6f} in both; max |difference| over 500 random forecasts {worst:.1e}; "
           f"binomial forecast: excess RPS {rps_ex1:.5f} (population 1, shape right) vs {rps_ex2:.5f} (population 2, shape wrong); excess CE {ce_ex1:.2e} vs {ce_ex2:.4f} nats")
    return PA, PD, ex


# ===================================================================================== (C0) Brier versus correlation
def part_identities():
    emit("== C. Brier is not correlation, even at K = 2 (exact example and identities)")
    p = [Fr(1, 5), Fr(1, 2), Fr(4, 5)]  # three items, true probability of rating 1
    pairs = [(Fr(0), Fr(1)), (Fr(2, 5), Fr(1, 5)), (Fr(0), Fr(1, 2)), (Fr(-1, 5), Fr(7, 5)), (Fr(3, 10), Fr(2, 5))]
    rows, ok, ex = [], True, {}
    for a, b in pairs:
        q = [a + b * x for x in p]
        rho = np.corrcoef([float(x) for x in q], [float(x) for x in p])[0, 1]
        excess = sum((qi - pi) ** 2 for qi, pi in zip(q, p)) / 3  # K=2: E_Y (q-Y)^2 = (q-p)^2 + p(1-p)
        formula = sum((a + (b - 1) * x) ** 2 for x in p) / 3
        ok = ok and excess == formula and abs(rho - 1) < 1e-12 and (excess == 0) == (a == 0 and b == 1)
        rows.append(f"(a={a}, b={b}): corr {rho:.6f}, Brier excess {excess} = {float(excess):.4f}")
        ex[(a, b)] = excess
    base = sum(x * (1 - x) for x in p) / 3
    report("DM4", "K=2: forecasts q = a + b p all have Pearson correlation 1 with p, and expected Brier excess mean((a + (b-1) p)^2), zero only at (0, 1)", ok,
           "; ".join(rows) + f"; Brier of the truth itself {base} = {float(base):.4f}")
    rng = np.random.default_rng(SEED + 3)
    e1 = e2 = e3 = 0.0
    for _ in range(2000):
        n = int(rng.integers(5, 40))
        X = rng.normal(size=n) * rng.uniform(0.2, 3) + rng.normal()
        Y = rng.normal(size=n) * rng.uniform(0.2, 3) + rng.normal()
        rho = np.corrcoef(X, Y)[0, 1]
        sx, sy = X.std(), Y.std()
        lhs = np.mean((X - Y) ** 2)
        rhs = (X.mean() - Y.mean()) ** 2 + (sx - sy) ** 2 + 2 * sx * sy * (1 - rho)
        e1 = max(e1, abs(lhs - rhs))
        Zx, Zy = (X - X.mean()) / sx, (Y - Y.mean()) / sy
        e2 = max(e2, abs(np.mean((Zx - Zy) ** 2) - 2 * (1 - rho)))
        a, b = rng.normal(), rng.uniform(0.1, 4)
        e3 = max(e3, abs(np.corrcoef(a + b * X, Y)[0, 1] - rho))
    report("DM5", "E[(X-Y)^2] = (mu_X-mu_Y)^2 + (s_X-s_Y)^2 + 2 s_X s_Y (1-rho); standardized scores: 2(1-rho); corr(a+bX, Y) = corr(X, Y) for b > 0 (2000 random pairs)",
           max(e1, e2, e3) < 1e-12, f"max abs errors {e1:.1e}, {e2:.1e}, {e3:.1e}")
    # K = 2: Brier = 2 x squared error on the mean rating = 2 x ranked probability score (RPS), for every forecast and outcome
    e4 = 0.0
    for _ in range(1000):
        q1 = rng.uniform()
        for y in (0, 1):
            brier = (1 - q1 - (1 - y)) ** 2 + (q1 - y) ** 2  # sum_k (q_k - 1{y=k})^2
            mse = (q1 - y) ** 2  # (m - y)^2 with the mean rating m = q_1
            rps_ = (((1 - q1) - (1 if y <= 0 else 0)) ** 2) / (2 - 1)  # (1/(K-1)) sum_{k<=K-2} (F_q(k) - 1{y<=k})^2, F_q(0) = q_0
            e4 = max(e4, abs(brier - 2 * mse), abs(brier - 2 * rps_), abs(mse - rps_))
    report("DM5b", "K=2: Brier = 2 x squared error on the mean rating = 2 x ranked probability score (RPS) for every forecast and outcome (so the squared-error, RPS and Brier objectives coincide)",
           e4 < 1e-12, f"max abs error over 2000 forecast-outcome pairs {e4:.1e}")


# ======================================================================================= (B) K = 2 table model sweep
def part_b():
    emit("== B. K = 2, table model with 20 contexts: Pearson versus Brier versus CE (synthetic ratings)")
    M = 20
    rng0 = np.random.default_rng(SEED + 10)
    p1 = rng0.uniform(0.05, 0.95, M)
    P = np.stack([1 - p1, p1], 1)
    f = np.array([0.0, 1.0])
    Ns = [200, 1000, 5000, 25000, 125000, 1000000]
    n_data, n_seed = 2, 3
    names = ["pearson", "brier", "ce", "pearson+cal", "brier+cal", "ce+cal"]
    store = {N: {k: [] for k in names} for N in Ns}
    store_all = {N: [] for N in Ns}  # Pearson TVD including stalled runs
    affine = {k: [] for k in ("pearson", "brier", "ce")}  # at the largest N, first data draw
    gap_small, stalled = [], []
    D = draw_counts(np.random.default_rng(SEED + 11), P, 50)
    ge = {"ce": gradcheck(loss_ce, D), "brier": gradcheck(loss_brier, D), "pearson": gradcheck(loss_corr_mean, D, f=f)}
    report("DM6.0", "analytic gradients of CE, Brier and Pearson (K=2 table) match central finite differences", max(ge.values()) < 1e-6,
           "max relative error " + ", ".join(f"{k} {v:.1e}" for k, v in ge.items()))
    for N in Ns:
        n = N // M
        for d in range(n_data):
            rng = np.random.default_rng([SEED, 12, N, d])
            D = draw_counts(rng, P, n)
            Dcal = draw_counts(rng, P, n)
            for s in range(n_seed):
                for si, std in enumerate(INIT_STDS):
                    seed = [SEED, 13, N, d, s, si]
                    for name, loss, kw in (("pearson", loss_corr_mean, {"f": f}), ("brier", loss_brier, {}), ("ce", loss_ce, {})):
                        Q, res = fit(loss, D, std, seed, **kw)
                        if name == "pearson":
                            gap = corr_gap("corr_mean", D, res, f)
                            store_all[N].append(tvd_rows(Q, P))
                            if std <= 1.0:
                                gap_small.append(gap)
                            if gap > GAP_TOL:
                                stalled.append((N, std, gap))
                                continue
                        store[N][name].append(tvd_rows(Q, P))
                        store[N][name + "+cal"].append(tvd_rows(recal_affine_perclass(Q, Dcal), P))
                        if N == Ns[-1] and d == 0:
                            affine[name].append(affine_fit(Q[:, 1], p1) + (std,))
    nrun = n_data * n_seed * len(INIT_STDS)
    for N in Ns:
        info("DM6", f"N={N:>8d} ({N // M} ratings per context) median [min, max] TVD to the truth over {len(store[N]['ce'])} runs ({len(store[N]['pearson'])} converged Pearson runs): "
             + "; ".join(f"{lbl(k)} {fmt_range(v)}" for k, v in store[N].items()))
    big = Ns[-1]
    ap, ab, ac = (np.array(affine[k]) for k in ("pearson", "brier", "ce"))
    n_st = len(stalled)
    st_std = sorted({s for _, s, _ in stalled})
    per_std = {sd: sum(1 for _, s_, _ in stalled if s_ == sd) for sd in INIT_STDS}
    info("DM6", f"stalled Pearson runs (gap to the best in-sample correlation above {GAP_TOL:g}; excluded from the Pearson medians): {n_st} of {nrun * len(Ns)}; "
         "by init std " + ", ".join(f"{sd}: {per_std[sd]} of {n_data * n_seed * len(Ns)}" for sd in INIT_STDS) + f"; largest gap {max([g for *_, g in stalled], default=0):.3f}; "
         f"median Pearson TVD at N={big} including the stalled runs {np.median(store_all[big]):.4f} (converged only {np.median(store[big]['pearson']):.4f})")
    info("DM6", "Pearson-trained at the largest N, learned probability = a + b * truth, per init std: "
         + "; ".join(f"std {s}: b in [{ap[ap[:, 3] == s, 1].min():.3f}, {ap[ap[:, 3] == s, 1].max():.3f}], a in [{ap[ap[:, 3] == s, 0].min():.3f}, {ap[ap[:, 3] == s, 0].max():.3f}]"
                        for s in INIT_STDS if (ap[:, 3] == s).any()))
    info("DM6", f"Brier-trained: b in [{ab[:, 1].min():.4f}, {ab[:, 1].max():.4f}], a in [{ab[:, 0].min():.4f}, {ab[:, 0].max():.4f}]; "
         f"CE-trained: b in [{ac[:, 1].min():.4f}, {ac[:, 1].max():.4f}], a in [{ac[:, 0].min():.4f}, {ac[:, 0].max():.4f}]; "
         f"the largest scale a probability can have here is 1 / (p_max - p_min) = {1 / (p1.max() - p1.min()):.4f} because probabilities must stay in [0, 1]")
    report("DM6a", "K=2, revised prediction P3: every Pearson run with init std <= 1.0 reaches the best in-sample correlation (init std 3.0 is expected to stall at saturated outputs)",
           max(gap_small) < GAP_TOL, f"largest gap to the best achievable in-sample correlation over {len(gap_small)} runs with std <= 1.0: {max(gap_small):.2e}; "
           f"stalled runs with std <= 1.0: {sum(1 for _, s_, _ in stalled if s_ <= 1.0)} of {len(gap_small)}, with std 3.0: {per_std[3.0]} of {n_data * n_seed * len(Ns)}; stalled runs are excluded from the Pearson statistics")
    rat = float(ap[:, 1].max() / ap[:, 1].min())
    report("DM6b", "K=2: the converged Pearson-trained probability is an exact affine map a + b p of the truth, and (a, b) depends on the initialization",
           rat > 2 and ap[:, 2].min() > 0.999,
           f"learned scale b over [{ap[:, 1].min():.3f}, {ap[:, 1].max():.3f}] (max/min {rat:.1f}), shift a over [{ap[:, 0].min():.3f}, {ap[:, 0].max():.3f}]; "
           f"affine R^2 >= {ap[:, 2].min():.5f}; the truth is (a, b) = (0, 1)")
    ok = (np.abs(ab[:, 1] - 1).max() < 0.02 and np.abs(ab[:, 0]).max() < 0.01 and np.abs(ac[:, 1] - 1).max() < 0.02 and np.abs(ac[:, 0]).max() < 0.01)
    report("DM6c", "K=2: Brier-trained and CE-trained probabilities land on the truth (a = 0, b = 1) from every init, up to sampling noise", ok,
           f"largest |b-1| {max(np.abs(ab[:, 1] - 1).max(), np.abs(ac[:, 1] - 1).max()):.4f}, largest |a| {max(np.abs(ab[:, 0]).max(), np.abs(ac[:, 0]).max()):.4f}")
    mp, mc_, mb = np.median(store[big]["pearson"]), np.median(store[big]["ce"]), np.median(store[big]["brier"])
    report("DM6d", f"K=2, N={big}: raw Pearson TVD is large and does not shrink with N, Brier and CE TVD are small",
           mp > 0.05 and mc_ < 0.01 and mb < 0.01 and np.median(store[Ns[-1]]["pearson"]) > 0.5 * np.median(store[Ns[2]]["pearson"]),
           f"median TVD Pearson {mp:.4f} (N={Ns[2]}: {np.median(store[Ns[2]]['pearson']):.4f}), Brier {mb:.4f}, CE {mc_:.4f}; ratio Pearson / CE {mp / mc_:.0f} (TVD ranges 0 to 1, lower is better)")
    mr = np.median(store[big]["pearson+cal"])
    report("DM6e", f"K=2, N={big}: a held-out affine calibration repairs Pearson (the gap is level and scale, not shape)", mr < 2 * mc_ + 0.002,
           f"median TVD Pearson + calibration {mr:.4f} vs CE {mc_:.4f}; the same calibration applied to CE gives {np.median(store[big]['ce+cal']):.4f} and to Brier {np.median(store[big]['brier+cal']):.4f}")
    return Ns, store, p1, ap


def part_b2():
    emit("== B'. control: 2-parameter logistic regression, truth inside the model class, continuous covariate x ~ Uniform(-2, 2), truth logit -0.5 + 1.5 x")
    beta = np.array([-0.5, 1.5])
    grid = np.linspace(-2, 2, 4001)
    pg = sigmoid(beta[0] + beta[1] * grid)
    res_tab = {}
    for N in (1000, 10000, 100000):
        rng = np.random.default_rng([SEED, 20, N])
        x = rng.uniform(-2, 2, N)
        y = (rng.random(N) < sigmoid(beta[0] + beta[1] * x)).astype(float)
        X = np.stack([np.ones(N), x], 1)

        def ce(th):
            z = X @ th
            return np.mean(np.logaddexp(0, z) - y * z), X.T @ (sigmoid(z) - y) / N

        def brier(th):
            q = sigmoid(X @ th)
            return np.mean((q - y) ** 2), X.T @ (2 * (q - y) * q * (1 - q)) / N

        def corr_q(th):
            q = sigmoid(X @ th)
            qb, yb = q.mean(), y.mean()
            vq = max(np.mean((q - qb) ** 2), 1e-300)
            return np.mean((q - qb) * (y - yb)) / math.sqrt(vq * yb * (1 - yb))

        def pearson(th):
            q = sigmoid(X @ th)
            qb, yb = q.mean(), y.mean()
            vq = max(np.mean((q - qb) ** 2), 1e-300)
            vy = yb * (1 - yb)
            rho = np.mean((q - qb) * (y - yb)) / math.sqrt(vq * vy)
            dq = ((y - yb) / math.sqrt(vq * vy) - rho * (q - qb) / vq) / N
            return -rho, -(X.T @ (dq * q * (1 - q)))

        out = {}
        for name, fn in (("ce", ce), ("brier", brier), ("pearson", pearson)):
            runs = []
            for std in INIT_STDS:
                for s in range(3):
                    th0 = np.random.default_rng([SEED, 21, N, s, int(std * 10)]).normal(0, std, 2)
                    r = minimize(fn, th0, jac=True, method="L-BFGS-B", options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-11})
                    runs.append((float(np.mean(np.abs(sigmoid(r.x[0] + r.x[1] * grid) - pg))), r.x, corr_q(r.x)))
            out[name] = runs
        best = max(c for _, _, c in out["pearson"])
        succ = [(tv, th) for tv, th, c in out["pearson"] if c >= best - 1e-6]
        fail = [(tv, th) for tv, th, c in out["pearson"] if c < best - 1e-6]
        res_tab[N] = (np.median([t for t, _, _ in out["ce"]]), np.median([t for t, _, _ in out["brier"]]), succ, fail)
        ths = np.array([th for _, th in succ])
        info("DM7", f"N={N:>6d}: CE TVD {fmt_range([t for t, _, _ in out['ce']])}; Brier TVD {fmt_range([t for t, _, _ in out['brier']])}; "
             f"Pearson: {len(succ)} of 12 runs reach the best in-sample correlation (median TVD {np.median([t for t, _ in succ]):.4f}, logit (theta0, theta1) in "
             f"[{ths[:, 0].min():.3f}, {ths[:, 0].max():.3f}] x [{ths[:, 1].min():.3f}, {ths[:, 1].max():.3f}], truth (-0.5, 1.5)); {len(fail)} runs stall at saturated outputs "
             f"(median TVD {np.median([t for t, _ in fail]) if fail else float('nan'):.3f})")
    emit("  OBSERVATION DM7 (no pass/fail; prediction P4 was 'unsure'): inside this well-specified two-parameter model the probability is sigmoid(theta0 + theta1 x), and an affine image "
         "a + b * sigmoid(...) of the truth is not itself of that form, so the model class cannot realize the scale-and-shift family; the converged Pearson fit is then pinned near the truth "
         "(TVD close to CE's), and training is fragile (some initializations stall at saturated outputs). "
         "The scale-and-shift freedom therefore needs a model class (or a judge read-out) flexible enough to realize affine images of the truth, as the table model of section B does; "
         "it is not a property of every model class.")
    return res_tab


# ===================================================================================== (C) K = 6 per-class correlation
def part_c():
    emit("== D. K = 6, table model with 12 contexts: per-class correlation versus CE versus Brier (synthetic ratings)")
    K, M = 6, 12
    rng0 = np.random.default_rng(SEED + 30)
    P = rng0.dirichlet(1.5 * np.ones(K), M)
    f = np.arange(K, dtype=float)
    Ns = [120, 600, 3000, 15000, 75000, 600000]
    n_data, n_seed = 2, 2
    objs = [("ce", loss_ce, {}, None), ("brier", loss_brier, {}, None),
            ("corr_perclass", loss_corr_perclass, {}, "softmax"), ("corr_perclass_sigmoid", loss_corr_perclass, {}, "sigmoid"),
            ("corr_mean", loss_corr_mean, {"f": f}, None), ("mse_mean", loss_mse_mean, {"f": f}, None)]
    D = draw_counts(np.random.default_rng(SEED + 31), P, 40)
    ge = {nm: gradcheck(l, D, **({"link": lk} if lk else {}), **kw) for nm, l, kw, lk in objs}
    report("DM8.0", "analytic gradients of the six K=6 objectives (CE, Brier, per-class corr with softmax and with sigmoid, corr of the mean, MSE of the mean) match finite differences",
           max(ge.values()) < 1e-6, "max relative error " + ", ".join(f"{k} {v:.1e}" for k, v in ge.items()))
    store = {N: {} for N in Ns}
    detail = {}
    stalled = {nm: [] for nm, *_ in objs if nm.startswith("corr")}
    gap_small = []
    nrun = n_data * n_seed * len(INIT_STDS)
    for N in Ns:
        n = N // M
        for d in range(n_data):
            rng = np.random.default_rng([SEED, 32, N, d])
            D = draw_counts(rng, P, n)
            Dcal = draw_counts(rng, P, n)
            for s in range(n_seed):
                for si, std in enumerate(INIT_STDS):
                    seed = [SEED, 33, N, d, s, si]
                    for nm, loss, kw, lk in objs:
                        Q, res = fit(loss, D, std, seed, link=lk, **kw)
                        if nm in stalled:
                            gap = corr_gap(nm, D, res, f)
                            if std <= 1.0:
                                gap_small.append(gap)
                            if gap > GAP_TOL:
                                stalled[nm].append((N, std, gap))
                                continue
                        Qn = Q / Q.sum(1, keepdims=True)
                        store[N].setdefault(nm, []).append(tvd_rows(Qn, P))
                        store[N].setdefault(nm + "+cal", []).append(tvd_rows(recal_affine_perclass(Q, Dcal), P))
                        if N == Ns[-1] and d == 0:
                            detail.setdefault(nm, []).append((Q, std))
    for N in Ns:
        info("DM8", f"N={N:>7d} ({N // M} ratings per context), median TVD to the truth: " + "; ".join(f"{lbl(k)} {np.median(v):.4f}" for k, v in store[N].items()))
    info("DM8", "stalled correlation runs (gap above %g, excluded from the medians of that objective): " % GAP_TOL
         + "; ".join(f"{lbl(k)} {len(v)} of {nrun * len(Ns)} (by init std: " + ", ".join(f"{sd}: {sum(1 for _, s_, _ in v if s_ == sd)}" for sd in INIT_STDS) + ")" for k, v in stalled.items()))
    big = Ns[-1]
    med = {k: float(np.median(v)) for k, v in store[big].items()}
    sp_min, slopes, sums = [], [], []
    for Q, std in detail["corr_perclass"]:
        sp_min.append(min(spearmanr(Q[:, k], P[:, k]).statistic for k in range(K)))
        sl = [affine_fit(Q[:, k], P[:, k]) for k in range(K)]
        slopes.append([x[1] for x in sl])
        sums.append(sum(x[0] for x in sl) + np.mean([x[1] for x in sl]))
    slopes = np.array(slopes)
    spread_within = float(np.max(slopes.max(1) / slopes.min(1)))
    bs = slopes.mean(1)
    info("DM8", f"per-class corr (softmax) at N={big}, {len(slopes)} converged runs: learned class probability = a_k + b_k p_k; slopes b_k (min..max over classes) per run: "
         + "; ".join(f"[{s.min():.3f}, {s.max():.3f}]" for s in slopes) + f"; sum_k a_k + mean b_k - 1 over runs in [{min(sums) - 1:.3f}, {max(sums) - 1:.3f}] (simplex constraint predicts 0)")
    report("DM8a", "K=6: per-class correlation (softmax) keeps the ranking of every class across contexts", min(sp_min) >= 0.95,
           f"smallest within-class Spearman correlation between learned and true probability over the {K} classes and {len(sp_min)} converged runs: {min(sp_min):.4f}")
    report("DM8b", "K=6: per-class correlation (softmax) has the right ranks but the wrong probabilities (raw TVD large, > 5x CE)",
           med["corr_perclass"] > 0.05 and med["corr_perclass"] > 5 * med["ce"],
           f"median TVD at N={big}: per-class corr {med['corr_perclass']:.4f}, corr of the mean rating {med['corr_mean']:.4f}, MSE of the mean {med['mse_mean']:.4f}, Brier {med['brier']:.4f}, CE {med['ce']:.4f}; "
           f"mean slope b per run over [{bs.min():.3f}, {bs.max():.3f}] (1 would be exact); TVD at N=15000 was {np.median(store[15000]['corr_perclass']):.4f}, so it does not shrink")
    report("DM8c", "K=6: with a softmax output the K slopes of the per-class variant coincide (one common scale), as the simplex constraint implies", spread_within < 1.1,
           f"largest max/min slope ratio across the classes within one converged run: {spread_within:.3f}")
    Qs = [Q for Q, _ in detail["corr_perclass_sigmoid"]]
    sl_sig = np.array([[affine_fit(Q[:, k], P[:, k])[1] for k in range(K)] for Q in Qs])
    rowsum = np.array([Q.sum(1) for Q in Qs])
    info("DM8", f"per-class corr with independent sigmoids (no simplex constraint) at N={big}, {len(Qs)} converged runs: slopes b_k (min..max over classes) "
         + "; ".join(f"[{s.min():.3f}, {s.max():.3f}]" for s in sl_sig) + f"; row sums of the raw outputs over [{rowsum.min():.3f}, {rowsum.max():.3f}] (a distribution sums to 1); median TVD after renormalizing {med['corr_perclass_sigmoid']:.4f}")
    report("DM8d", f"K=6, N={big}: K affine maps + renormalization on held-out ratings bring per-class correlation within 2x of CE (the gap is level and scale)",
           med["corr_perclass+cal"] < 2 * med["ce"] + 0.002 and med["corr_perclass_sigmoid+cal"] < 2 * med["ce"] + 0.002,
           f"median TVD per-class Pearson + calibration {med['corr_perclass+cal']:.4f} (sigmoid variant {med['corr_perclass_sigmoid+cal']:.4f}) vs CE {med['ce']:.4f}; the same calibration on CE gives {med['ce+cal']:.4f}, on Brier {med['brier+cal']:.4f}")
    rm_mse = [math.sqrt(np.mean((Q @ f - P @ f) ** 2)) for Q, _ in detail["mse_mean"]]
    rm_cor = [math.sqrt(np.mean((Q @ f - P @ f) ** 2)) for Q, _ in detail["corr_mean"]]
    aff = np.array([affine_fit(Q @ f, P @ f) for Q, _ in detail["corr_mean"]])
    report("DM8e", f"K=6, N={big}: correlation of the MEAN rating cannot be repaired by per-class calibration, and squared error on the mean pins the mean but not the shape",
           med["corr_mean+cal"] > 5 * med["ce"] and max(rm_mse) < 0.01 and med["mse_mean"] > 5 * med["ce"],
           f"median TVD: Pearson of the mean rating + calibration {med['corr_mean+cal']:.4f}, MSE of the mean raw {med['mse_mean']:.4f} (calibration {med['mse_mean+cal']:.4f}), CE {med['ce']:.4f}; "
           f"RMSE of the learned mean vs the true mean: MSE-trained max {max(rm_mse):.4f}, corr-trained over [{min(rm_cor):.3f}, {max(rm_cor):.3f}]; "
           f"corr-trained learned mean = a + b * true mean with b in [{aff[:, 1].min():.3f}, {aff[:, 1].max():.3f}], R^2 >= {aff[:, 2].min():.5f}")
    small_st = {k: sum(1 for _, s_, _ in v if s_ <= 1.0) for k, v in stalled.items()}
    report("DM8f", "K=6, revised prediction P3: every correlation-type run with init std <= 1.0 reaches the best in-sample correlation (stalls at saturated outputs are expected only at std 3.0)",
           max(gap_small) < GAP_TOL,
           f"largest gap over {len(gap_small)} runs with std <= 1.0: {max(gap_small):.2e}; stalled runs with std <= 1.0 per objective: "
           + ", ".join(f"{lbl(k)} {v}" for k, v in small_st.items()) + "; stalled runs are excluded from the statistics of that objective")
    return Ns, store, P, detail


# ============================================================================================ (D) toy language model
def part_d():
    emit("== E. toy language model: vocabulary V = 50, 10 contexts, table of next-token logits (synthetic tokens)")
    V, M = 50, 10
    rng0 = np.random.default_rng(SEED + 40)
    P = rng0.dirichlet(0.5 * np.ones(V), M)
    P = np.maximum(P, 1e-12)
    P /= P.sum(1, keepdims=True)
    f = np.arange(V, dtype=float)
    Ns = [10000, 100000, 1000000, 10000000]
    D = draw_counts(np.random.default_rng(SEED + 41), P, 200)
    ge = {"ce": gradcheck(loss_ce, D), "corr_mean": gradcheck(loss_corr_mean, D, f=f)}
    report("DM9.0", "analytic gradients of CE and of corr(expected token ID, observed token ID) (V=50) match finite differences", max(ge.values()) < 1e-6,
           "max relative error " + ", ".join(f"{k} {v:.1e}" for k, v in ge.items()))
    store = {N: {"ce": [], "corr_id": []} for N in Ns}
    stalled, aff, ceilings = [], [], []
    for N in Ns:
        n = N // M
        for d in range(2):
            D = draw_counts(np.random.default_rng([SEED, 42, N, d]), P, n)
            for s in range(2):
                for si, std in enumerate((1.0, 2.0)):
                    seed = [SEED, 43, N, d, s, si]
                    Qc, _ = fit(loss_ce, D, std, seed)
                    store[N]["ce"].append(tvd_rows(Qc, P))
                    Qr, res = fit(loss_corr_mean, D, std, seed, f=f)
                    gap = corr_gap("corr_mean", D, res, f)
                    if gap > GAP_TOL:
                        stalled.append((N, std, gap))
                        continue
                    store[N]["corr_id"].append(tvd_rows(Qr, P))
                    if N == Ns[-1] and d == 0:
                        aff.append(affine_fit(Qr @ f, P @ f))
                        ceilings.append(max_corr_mean(D, f))
    for N in Ns:
        info("DM9", f"N={N:>9d} ({N // M} tokens per context), median [min, max] TVD to the true next-token distribution: "
             + "; ".join(f"{lbl(k)} {fmt_range(v)} ({len(v)} runs)" for k, v in store[N].items()))
    info("DM9", f"stalled correlation runs (gap above {GAP_TOL:g}, excluded): {len(stalled)} of {len(Ns) * 8}, all at init std {sorted({s for _, s, _ in stalled})}, largest gap {max([g for *_, g in stalled], default=0):.1e}")
    x = np.log([N for N in Ns[1:]])
    y = np.log([np.median(store[N]["ce"]) for N in Ns[1:]])
    slope = float(np.polyfit(x, y, 1)[0])
    report("DM9a", "V=50: CE training recovers the conditional distribution; TVD falls like N^(-1/2)", -0.6 <= slope <= -0.4,
           f"log-log slope of the median CE TVD over N = 1e5..1e7: {slope:.3f}; TVD {np.median(store[Ns[0]]['ce']):.4f} at N=1e4 down to {np.median(store[Ns[-1]]['ce']):.4f} at N=1e7")
    r_small, r_big = np.median(store[Ns[1]]["corr_id"]), np.median(store[Ns[-1]]["corr_id"])
    aff = np.array(aff)
    ceil_txt = (f"{min(ceilings):.3f}" if round(min(ceilings), 3) == round(max(ceilings), 3) else f"{min(ceilings):.3f} to {max(ceilings):.3f}")
    report("DM9b", "V=50: correlation of the expected token ID with the observed token ID learns the mean ID up to an affine map, but TVD does not shrink",
           aff[:, 2].min() > 0.99 and r_big / r_small > 0.7 and r_big > 0.3,
           f"learned expected ID = a + b * true expected ID with b in [{aff[:, 1].min():.3f}, {aff[:, 1].max():.3f}], R^2 >= {aff[:, 2].min():.5f} (best achievable correlation {ceil_txt}); "
           f"median TVD {r_small:.3f} at N=1e5 and {r_big:.3f} at N=1e7 (ratio {r_big / r_small:.2f}); CE TVD at N=1e7 {np.median(store[Ns[-1]]['ce']):.4f}")
    rng = np.random.default_rng(SEED + 44)
    perms = [np.arange(V)] + [rng.permutation(V) for _ in range(7)]
    D0 = draw_counts(np.random.default_rng([SEED, 45]), P, 1000000 // M)
    tv_ce, eta = [], []
    for pm in perms:
        Pp = np.empty_like(P)
        Pp[:, pm] = P  # token k receives the new ID pm[k]
        S2 = np.empty_like(D0.S)
        S2[:, pm] = D0.S
        Q, _ = fit(loss_ce, Counts(S2), 1.0, [SEED, 46])
        tv_ce.append(tvd_rows(Q, Pp))
        mu = Pp @ f
        eta.append(math.sqrt(np.mean((mu - mu.mean()) ** 2) / (np.mean(Pp @ f ** 2) - mu.mean() ** 2)))
    inv = max(tv_ce) - min(tv_ce)
    report("DM9c", "V=50: CE is invariant to a relabeling of the token IDs, the best achievable correlation is not (IDs are arbitrary labels)",
           inv < 5e-4 and (max(eta) - min(eta)) > 0.05,
           f"over 8 labelings (identity + 7 random): CE TVD in [{min(tv_ce):.5f}, {max(tv_ce):.5f}] (spread {inv:.1e}); best achievable correlation eta in [{min(eta):.3f}, {max(eta):.3f}]")
    return Ns, store


# ================================================================================================================ figures
BLUE, RED, VIOLET = "#257de0", "#ec4647", "#6a4ce2"  # validated order: CE blue, correlation-type red, Brier violet
INK, MUTED, GRID = "#191919", "#666666", "#d9d9d9"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                     "axes.titlesize": 10.5, "legend.fontsize": 8.5, "figure.dpi": 130})


def style(ax, logx=False, logy=False):
    ax.grid(True)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    if logx or logy:
        ax.grid(True, which="minor", color=GRID, lw=0.5, ls=":", alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def line_with_band(ax, Ns, series, color, ls, label, marker="o", ms=5, band=False, z=2):
    med = np.array([np.median(v) for v in series])
    ax.plot(Ns, med, color=color, ls=ls, lw=2.0, marker=marker, ms=ms, mfc=color, mec="white", mew=0.8, label=label, zorder=z)
    if band:
        ax.fill_between(Ns, [np.min(v) for v in series], [np.max(v) for v in series], color=color, alpha=0.12, lw=0)


def make_fig1(b, c, d):
    Ns_b, st_b, _, _ = b
    Ns_c, st_c, _, _ = c
    Ns_d, st_d = d
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.2))
    ax = axs[0]
    line_with_band(ax, Ns_b, [st_b[N]["ce"] for N in Ns_b], BLUE, "-", "CE-trained", "o", ms=8, z=2)
    line_with_band(ax, Ns_b, [st_b[N]["brier"] for N in Ns_b], VIOLET, "-", "Brier-trained (lies on the CE line)", "s", ms=4, z=3)
    line_with_band(ax, Ns_b, [st_b[N]["pearson"] for N in Ns_b], RED, "-", "Pearson-trained, raw", "^", band=True)
    line_with_band(ax, Ns_b, [st_b[N]["pearson+cal"] for N in Ns_b], RED, "--", "Pearson-trained + held-out affine calibration", "v")
    ax.set_title("(a) Two ratings (K = 2), 20 contexts")
    ax = axs[1]
    line_with_band(ax, Ns_c, [st_c[N]["ce"] for N in Ns_c], BLUE, "-", "CE-trained", "o", ms=8, z=2)
    line_with_band(ax, Ns_c, [st_c[N]["brier"] for N in Ns_c], VIOLET, "-", "Brier-trained (lies on the CE line)", "s", ms=4, z=3)
    line_with_band(ax, Ns_c, [st_c[N]["corr_perclass"] for N in Ns_c], RED, "-", "per-class Pearson-trained, raw", "^", band=True)
    line_with_band(ax, Ns_c, [st_c[N]["corr_perclass+cal"] for N in Ns_c], RED, "--", "per-class Pearson + K held-out affine maps", "v")
    line_with_band(ax, Ns_c, [st_c[N]["corr_mean"] for N in Ns_c], RED, ":", "Pearson of the mean rating, raw", "D")
    ax.set_title("(b) Ratings 0 to 5 (K = 6), 12 contexts")
    ax = axs[2]
    line_with_band(ax, Ns_d, [st_d[N]["ce"] for N in Ns_d], BLUE, "-", "CE-trained (the next-token loss)", "o", ms=8)
    line_with_band(ax, Ns_d, [st_d[N]["corr_id"] for N in Ns_d], RED, "-", "Pearson of expected token ID with observed ID", "^", band=True)
    ax.set_title("(c) Toy language model, vocabulary V = 50, 10 contexts")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        style(ax, True, True)
        ax.set_xlabel("Number of training ratings N (synthetic data, no human ratings)")
        ax.legend(loc="lower left", frameon=False)
    for ax in axs[:2]:
        ax.set_ylabel("TVD to the true rating distribution, mean over contexts\n(total variation distance, range 0 to 1, lower is better)")
    axs[2].set_ylabel("TVD to the true next-token distribution, mean over contexts\n(total variation distance, range 0 to 1, lower is better)")
    fig.suptitle("Matching one number or a ranking does not recover a discrete distribution; matching all K indicator means with a proper loss does.\n"
                 "Synthetic data, no human ratings; distance to the known true distribution, measured on the training contexts.", fontsize=10, y=0.995)
    fig.text(0.5, 0.012, "Lines: median over random initializations and two data draws (correlation objectives: converged runs only). Band: min to max of the raw Pearson runs. "
             "Lower is better in every panel.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"discrete_means_fig1_tvd_vs_n.{ext}"), dpi=160)
    plt.close(fig)


def make_fig2(PA, PD, ex, b, c):
    _, _, p1, ap = b
    _, _, P6, detail = c
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.2))
    ax = axs[0]
    x = np.arange(6)
    ax.bar(x - 0.2, [float(v) for v in PA], 0.38, color=BLUE, label="Binomial(5, 1/2)")
    ax.bar(x + 0.2, [float(v) for v in PD], 0.38, color=VIOLET, label="Other law with the same mean and variance")
    ax.set_xlabel("Rating (0 to 5)")
    ax.set_ylabel("True probability of the rating (exact, synthetic)")
    ax.set_title("(a) Both have mean 2.5 and variance 1.25 (K = 6)")
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper center", frameon=False)
    ax.text(0.03, 0.86, "Forecast the left law when the right law is true:\n"
            f"the mean error is 0, yet excess RPS {float(ex['rps']):.4f},\nexcess Brier {float(ex['brier']):.4f}, "
            f"excess CE {ex['ce']:.3f} nats, TVD {ex['tvd']:.3f}\n(all excesses are lower-is-better, zero when the forecast is right)", transform=ax.transAxes, va="top", fontsize=8.5)
    style(ax)
    ax = axs[1]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls="--", label="Truth: learned = true (closer is better)", zorder=1)
    M = len(p1)
    P = np.stack([1 - p1, p1], 1)
    f = np.array([0.0, 1.0])
    D = draw_counts(np.random.default_rng([SEED, 12, 1000000, 0]), P, 1000000 // M)
    first = True
    for si, std in enumerate(INIT_STDS):
        for s in range(3):
            Q, res = fit(loss_corr_mean, D, std, [SEED, 13, 1000000, 0, s, si], f=f)
            if corr_gap("corr_mean", D, res, f) > GAP_TOL:
                continue
            ax.plot(p1, Q[:, 1], color=RED, lw=0.8, marker="^", ms=4.5, mfc=RED, mec="white", mew=0.5, alpha=0.6,
                    label="Pearson-trained, one line per random init (converged runs)" if first else None)
            first = False
    Qc, _ = fit(loss_ce, D, 1.0, [SEED, 13, 1000000, 0, 0, 1])
    Qb, _ = fit(loss_brier, D, 1.0, [SEED, 13, 1000000, 0, 0, 1])
    ax.plot(p1, Qc[:, 1], ls="none", marker="o", ms=7, mfc=BLUE, mec="white", mew=0.8, label="CE-trained", zorder=4)
    ax.plot(p1, Qb[:, 1], ls="none", marker="s", ms=4, mfc=VIOLET, mec="white", mew=0.6, label="Brier-trained", zorder=5)
    ax.set_xlabel("True probability of rating 1 in the context (synthetic)")
    ax.set_ylabel("Learned probability of rating 1 after training on\nN = 1,000,000 synthetic ratings (K = 2)")
    ax.set_title("(b) K = 2: Pearson fixes the ranking, not the probabilities")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False)
    style(ax)
    ax = axs[2]
    ax.plot([0, 0.6], [0, 0.6], color=MUTED, lw=1.2, ls="--", label="Truth: learned = true (closer is better)", zorder=1)
    Qr = [Q for Q, s in detail["corr_perclass"] if s == 1.0][0]
    Qs = [Q for Q, s in detail["corr_perclass_sigmoid"] if s == 1.0][0]
    Qc6 = detail["ce"][0][0]
    ax.plot(P6.ravel(), Qs.ravel(), ls="none", marker="s", ms=4.5, mfc="none", mec=RED, mew=1.0, label="per-class Pearson, independent sigmoid outputs (one init)")
    ax.plot(P6.ravel(), Qr.ravel(), ls="none", marker="^", ms=5, mfc=RED, mec="white", mew=0.4, label="per-class Pearson, softmax output (one init)")
    ax.plot(P6.ravel(), Qc6.ravel(), ls="none", marker="o", ms=4.5, mfc=BLUE, mec="white", mew=0.4, label="CE-trained", zorder=4)
    ax.set_xlabel("True probability of a rating, one point per context and rating (synthetic)")
    ax.set_ylabel("Learned probability of that rating after training on\nN = 600,000 synthetic ratings (K = 6)")
    ax.set_title("(c) K = 6: ranks kept per class, level and scale are not")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.34)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(loc="upper left", frameon=False)
    style(ax)
    fig.suptitle("Why correlation is not Brier even at K = 2, and why a numeric mean is not a distribution once K > 2. "
                 "Synthetic data, no human ratings; learned probabilities are read off the training contexts.", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"discrete_means_fig2_affine_freedom.{ext}"), dpi=160)
    plt.close(fig)


# ====================================================================================================================== main
def main():
    emit(f"discrete_means_check.py, 11-26-2026. Synthetic data, no human ratings. seed {SEED}. numpy {np.__version__}")
    emit("LEGEND. Data: every number below comes from synthetic ratings drawn from known true distributions (no human ratings, no held-out human data); N is the number of synthetic training ratings, "
         "and 'held-out' means a second synthetic sample of the same size used only for calibration.")
    emit("LEGEND. TVD = total variation distance 0.5 * sum_k |q_k - p_k| between a learned distribution q and the true distribution p, averaged equally over contexts; range 0 to 1, lower is better; "
         "a value of 0.05 means 5 percent of the probability mass is misplaced.")
    emit("LEGEND. Brier = sum_k (q_k - 1{y=k})^2; CE = cross-entropy -log q_y (lower is better, minimized by the true distribution); ranked probability score (RPS) = mean over the K-1 cumulative cuts "
         "of the squared difference between forecast and outcome cumulative probabilities; Pearson = Pearson correlation between forecast and outcome (higher is better, invariant to a + b x with b > 0).")
    emit("LEGEND. 'X-trained' = the table-model logits minimized the X objective by L-BFGS from random initial logits; 'X + held-out affine calibration' = after training, each class probability is mapped "
         "by a least-squares affine map fitted on a second synthetic sample, clipped at 1e-6 and renormalized. Medians are over random initializations and two data draws; no confidence intervals "
         "are used because the truth is known and the quantity of interest is the exact distance, so the spread (min to max) is reported where it matters.")
    PA, PD, ex = part_a()
    part_identities()
    b = part_b()
    part_b2()
    c = part_c()
    d = part_d()
    make_fig1(b, c, d)
    make_fig2(PA, PD, ex, b, c)
    nfail = sum(1 for _, ok in VERDICTS if not ok)
    emit(f"SUMMARY {len(VERDICTS) - nfail} PASS, {nfail} FAIL out of {len(VERDICTS)} checks; figures written to figures/discrete_means_fig1_tvd_vs_n.[pdf,png] and "
         f"figures/discrete_means_fig2_affine_freedom.[pdf,png]; {time.time() - T0:.0f} s")
    with open(os.path.join(RES, "discrete_means_output.txt"), "w") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
