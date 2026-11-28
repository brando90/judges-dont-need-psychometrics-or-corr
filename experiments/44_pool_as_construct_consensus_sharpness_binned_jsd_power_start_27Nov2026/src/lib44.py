"""Shared helpers for Experiment 44. No model calls; CPU only; deterministic given the seeds.

Scores are written for a forecast q (floored, sums to 1) and a rating y in {0..K-1}:
  brier: sum_k (q_k - 1{y=k})^2          nrps: (K-1)^-1 sum_{m<K-1} (Q_m - 1{y<=m})^2          ce: -log q_y
The unbiased per-sample risk of an item is the mean over its ratings of S(q, y); the set risk is the mean over items.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
EXP39 = REPO / "experiments" / "39_proper_score_vs_d_study_budget_allocation_start_26Nov2026"
EXP28 = next((REPO / "experiments").glob("28_public_rating_sets_*"))
FLOOR = 1e-6
SEED0 = 20261127
DATASETS = ["chaosnli_snli", "chaosnli_mnli", "chaosnli_alphanli", "dices350", "dices990"]
DICES = ("dices350", "dices990")
JUDGES = ["gpt-6-astra", "claude-opus-5-5", "kimi-k3", "glm-5.3", "deepseek-v4.1-flash"]
PRIMARY = {"chaosnli_snli": "brier", "chaosnli_mnli": "brier", "chaosnli_alphanli": "brier",
           "dices350": "nrps", "dices990": "nrps"}
DICES_CSV = {"dices350": "diverse_safety_adversarial_dialog_350.csv",
             "dices990": "diverse_safety_adversarial_dialog_990.csv"}
Z80 = 1.959963984540054 + 0.8416212335729143  # two-sided 5% level, 80% power


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- data
def load_bundle():
    path = EXP39 / "data" / "processed" / "study_a_bundle.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}: run experiments/39_*/src/fetch_raw_data.py, then prepare_data.py")
    z = np.load(path, allow_pickle=True)
    return z["bundle"].item(), sha256(path)


def forecast_array(d, drop_invalid=False):
    """[N, 5, K] floored forecasts; invalid forecasts -> uniform (Experiment 28 rule). keep mask for the sensitivity run."""
    K = int(d["K"])
    P = np.array(d["p_judges"], float)
    valid = np.asarray(d["judge_valid"], bool)
    P[~valid] = 1.0 / K
    P = (1.0 - K * FLOOR) * P + FLOOR
    keep = valid.all(1) if drop_invalid else np.ones(len(P), bool)
    return P, keep


def subset(d, keep):
    out = dict(d)
    for k in ("ids", "counts", "ratings", "n_avail", "rater_ids", "p_judges", "judge_valid", "p_mix"):
        if d.get(k) is not None:
            out[k] = d[k][keep]
    return out


# ---------------------------------------------------------------- scores
def score_matrix(q, kind):
    q = np.asarray(q, float)
    K = q.shape[-1]
    if kind == "brier":
        return (q ** 2).sum(-1, keepdims=True) - 2.0 * q + 1.0
    if kind == "ce":
        return -np.log(q)
    if kind == "nrps":
        Q = np.cumsum(q, -1)[..., : K - 1]
        ind = (np.arange(K)[:, None] <= np.arange(K - 1)[None, :]).astype(float)  # 1{k <= m}
        return ((Q[..., None, :] - ind) ** 2).sum(-1) / (K - 1)
    raise ValueError(kind)


def risk_items(q, counts, kind):
    """Per-item unbiased per-sample risk. q [N,K] or [N,J,K]; counts [N,K] -> [N] or [N,J]."""
    S = score_matrix(q, kind)
    c = np.asarray(counts, float)
    if S.ndim == c.ndim + 1:
        c = c[:, None, :]
    return (c * S).sum(-1) / c.sum(-1)


def temper(q, tau):
    K = q.shape[-1]
    z = np.log(q) / tau
    z = z - z.max(-1, keepdims=True)
    t = np.exp(z)
    t = t / t.sum(-1, keepdims=True)
    return (1.0 - K * FLOOR) * t + FLOOR


# ---------------------------------------------------------------- statistics helpers
def boot_weights(rng, N, B):
    idx = rng.integers(0, N, size=(B, N))
    return np.stack([np.bincount(r, minlength=N) for r in idx]).astype(float)


def boot_p(draws):
    draws = np.asarray(draws, float)
    B = len(draws)
    le = (1 + (draws <= 0).sum()) / (B + 1)
    ge = (1 + (draws >= 0).sum()) / (B + 1)
    return float(min(1.0, 2 * min(le, ge)))


def pct_ci(x):
    lo, hi = np.percentile(np.asarray(x, float), [2.5, 97.5])
    return float(lo), float(hi)


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def tau_b(a, b):
    r = kendalltau(a, b)
    return float(r.statistic)


# ---------------------------------------------------------------- binned JSD (Elangovan et al., ICLR 2025)
def human_bins(counts, ordinal):
    counts = np.asarray(counts)
    n = counts.sum(1)
    if ordinal:  # lower median sorted[(n-1)//2]
        cum = np.cumsum(counts, 1)
        return np.argmax(cum >= ((n - 1) // 2 + 1)[:, None], 1)
    return np.argmax(counts, 1)


def jsd2(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    m = 0.5 * (p + q)

    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(a > 0, a * (np.log2(a) - np.log2(b)), 0.0)
        return t.sum(-1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def jsb(counts, q, bins):
    """Binned JSD, log base 2, bins weighted by their share of items. Lower is better."""
    counts = np.asarray(counts, float)
    tot = 0.0
    for b in np.unique(bins):
        m = bins == b
        h = counts[m].sum(0)
        h = h / h.sum()
        mach = q[m].mean(0)
        mach = mach / mach.sum()
        tot += m.mean() * float(jsd2(h, mach))
    return tot


def jsb_boot(counts, q, bins, W):
    counts = np.asarray(counts, float)
    N, K = counts.shape
    nb = int(bins.max()) + 1
    Bm = np.eye(nb)[bins]
    Xh = (Bm[:, :, None] * counts[:, None, :]).reshape(N, nb * K)
    Xm = (Bm[:, :, None] * q[:, None, :]).reshape(N, nb * K)
    H = (W @ Xh).reshape(-1, nb, K)
    M = (W @ Xm).reshape(-1, nb, K)
    nbB = W @ Bm
    with np.errstate(divide="ignore", invalid="ignore"):
        hp = H / H.sum(-1, keepdims=True)
        mp = M / M.sum(-1, keepdims=True)
    j = np.where(nbB > 0, jsd2(np.nan_to_num(hp), np.nan_to_num(mp)), 0.0)
    return ((nbB / nbB.sum(1, keepdims=True)) * j).sum(1)


# ---------------------------------------------------------------- power / minimum detectable difference
def pair_components(counts, qa, qb, kind):
    counts = np.asarray(counts, float)
    n = counts.sum(1)
    delta = score_matrix(qa, kind) - score_matrix(qb, kind)
    h = counts / n[:, None]
    D = (h * delta).sum(1)
    s2 = np.maximum(((h * delta ** 2).sum(1) - D ** 2) * n / (n - 1), 0.0)
    return D, s2, n, delta


def variance_components(D, s2, n):
    return max(0.0, float(D.var(ddof=1) - (s2 / n).mean())), float(s2.mean())


def mdd(sb2, sw2, N, k):
    return float(Z80 * np.sqrt((sb2 + sw2 / k) / N))


def n80(sb2, sw2, observed):
    return float(Z80 ** 2 * (sb2 + sw2) / observed ** 2) if observed != 0 else float("inf")


def v_sub(D, s2, n, N, k):
    """Exact variance of the paired mean when N items are drawn without replacement from len(D) items and
    k ratings without replacement from each item's n_i ratings."""
    Nt = len(D)
    between = (1.0 - N / Nt) * D.var(ddof=1) / N
    within = float(((s2 / k) * (1.0 - k / n)).mean()) / N
    return between + within


# ---------------------------------------------------------------- Arm P helpers
def rater_attrs(dataset):
    """rater_id -> (gender, age group, race). An attribute that differs between a rater's rows becomes None
    (the rater is then excluded from the contrast that uses it; DICES-990: 1 rater for age, 1 for race)."""
    path = EXP39 / "data" / "raw" / DICES_CSV[dataset]
    seen = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sets = seen.setdefault(row["rater_id"], (set(), set(), set()))
            for s_, col in zip(sets, ("rater_gender", "rater_age", "rater_race")):
                s_.add(row[col])
    return {r: tuple(next(iter(s_)) if len(s_) == 1 else None for s_ in sets) for r, sets in seen.items()}


CONTRASTS = {  # name -> (label of pool A, label of pool B, function (gender, age, race) -> 0 / 1 / -1)
    "gender": ("Man", "Woman", lambda g, a, r: 0 if g == "Man" else (1 if g == "Woman" else -1)),
    "age": ("gen z", "millenial + gen x+", lambda g, a, r: 0 if a == "gen z" else (1 if a in ("millenial", "gen x+") else -1)),
    "race": ("White", "Black/African American",
             lambda g, a, r: 0 if r == "White" else (1 if r == "Black/African American" else -1)),
}


def rating_table(d):
    """One row per rating: item index, rater index, label; plus the rater id list."""
    items, raters, ys = [], [], []
    rmap = {}
    for i in range(len(d["ids"])):
        for t in range(int(d["n_avail"][i])):
            rid = d["rater_ids"][i, t]
            items.append(i)
            raters.append(rmap.setdefault(rid, len(rmap)))
            ys.append(int(d["ratings"][i, t]))
    return np.array(items), np.array(raters), np.array(ys), list(rmap)


def pool_item_counts(item, rater_lab, y, N, K):
    """[N, 2, K] label counts per item for pool 0 and pool 1 given a rater->pool vector (-1 = excluded)."""
    lab = rater_lab
    ok = lab >= 0
    idx = (item[ok] * 2 + lab[ok]) * K + y[ok]
    return np.bincount(idx, minlength=N * 2 * K).reshape(N, 2, K)


def mean_tvd(cnt, min_raters=5):
    nA = cnt[:, 0].sum(1)
    nB = cnt[:, 1].sum(1)
    elig = (nA >= min_raters) & (nB >= min_raters)
    if not elig.any():
        return float("nan"), elig
    pa = cnt[elig, 0] / nA[elig, None]
    pb = cnt[elig, 1] / nB[elig, None]
    return float(0.5 * np.abs(pa - pb).sum(1).mean()), elig


def pool_draw_risks(LA, MA, LB, MB, vA, vB, W):
    """Two-way bootstrap draws of pool risks. LA[j]: csr [Ne, nA] losses of judge j (0 where unrated); MA: csr presence.
    vA [nA, c], vB [nB, c]: rater multiplicities; W [Ne, c]: item multiplicities. Returns RA, RB each [c, J]."""
    denA = MA @ vA
    denB = MB @ vB
    valid = (denA > 0) & (denB > 0)
    Wv = W * valid
    tot = Wv.sum(0)
    RA, RB = [], []
    for La, Lb in zip(LA, LB):
        with np.errstate(divide="ignore", invalid="ignore"):
            ra = np.where(valid, (La @ vA) / np.where(denA > 0, denA, 1), 0.0)
            rb = np.where(valid, (Lb @ vB) / np.where(denB > 0, denB, 1), 0.0)
        RA.append((Wv * ra).sum(0) / tot)
        RB.append((Wv * rb).sum(0) / tot)
    return np.stack(RA, 1), np.stack(RB, 1)
