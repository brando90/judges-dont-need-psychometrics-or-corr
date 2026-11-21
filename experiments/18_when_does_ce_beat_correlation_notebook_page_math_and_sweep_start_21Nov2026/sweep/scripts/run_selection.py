"""Exp 18 sweep B: selecting ONE judge from a pool of M fixed candidate judges (prompt/harness-style coarse selection)
by Pearson, Spearman, CE or RPS on n training items with k ratings each; scored on test against the truth.
Pool types: 'prob' (judge outputs a 6-category probability vector), 'votes3' (a histogram of 3 sampled votes, floor 1e-6, as in the
cached scalar judges of experiments 12/14/16) and 'votes3_f5e-2' (floor 5e-2). Writes ../results/rawB.jsonl. Date 11-21-2026."""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_core as s_
toy, e11 = s_.toy, s_.e11
NS = [4, 8, 16, 32, 64, 128, 256]; KS = [1, 3, 10]; POOLS = ["prob", "votes3", "votes3_f5e-2"]; SEEDS = 100; NTEST = 500; M = 24
G = toy.GRID

def judge_params(rng):
    return dict(a=rng.uniform(0.2, 1.6, M), b=rng.uniform(-0.6, 0.6, M), sig=rng.uniform(0.0, 1.5, M))

def judge_probs(Ptrue, par, rng):
    """(M, m, 6) judge distributions: tempered truth, shifted, plus per-item logit noise."""
    L = np.log(np.clip(Ptrue, 1e-9, None))[None]
    Z = par["a"][:, None, None] * L + par["b"][:, None, None] * (G[None, None] - 2.5) \
        + par["sig"][:, None, None] * rng.standard_normal((M, Ptrue.shape[0], 6))
    Z = Z - Z.max(-1, keepdims=True); Q = np.exp(Z); return Q / Q.sum(-1, keepdims=True)

def to_votes(Q, rng, floor):
    """histogram of 3 votes sampled from each judge distribution (inverse-CDF sampling), then floored."""
    cdf = np.cumsum(Q, -1); u = rng.random((3,) + Q.shape[:-1] + (1,))
    cat = np.minimum((u > cdf[None]).sum(-1), 5)                       # (3, M, m)
    out = np.zeros_like(Q)
    for v in range(3):
        np.put_along_axis(out, cat[v][..., None], np.take_along_axis(out, cat[v][..., None], -1) + 1 / 3, -1)
    out = np.clip(out, floor, None); return out / out.sum(-1, keepdims=True)

def score_rows(Q, P):
    """criteria on train items for each of M judges: pearson, spearman, ce, rps."""
    h = P @ G; res = np.zeros((4, Q.shape[0]))
    pc = np.cumsum(P, 1)[:, :5]
    for j in range(Q.shape[0]):
        mu = Q[j] @ G
        res[0, j] = np.corrcoef(mu, h)[0, 1] if mu.std() > 1e-12 and h.std() > 1e-12 else -1
        res[1, j] = spearmanr(mu, h)[0] if mu.std() > 1e-12 and h.std() > 1e-12 else -1
        res[2, j] = -(P * np.log(np.clip(Q[j], 1e-6, None))).sum(1).mean()   # CE (lower better)
        qc = np.cumsum(Q[j], 1)[:, :5]; res[3, j] = ((qc - pc) ** 2).sum(1).mean()
    return res

def run_one(args):
    sc, n, k, pool, seed = args
    rng = np.random.default_rng([seed, n, k, ord(sc), {"prob": 0, "votes3": 1, "votes3_f5e-2": 2}[pool]])
    par = judge_params(rng)       # a fresh pool of M judges per seed
    x = rng.uniform(-1, 1, n); Ptr = e11.truth(x, sc)
    P = np.array([rng.multinomial(k, p) for p in Ptr]) / k
    xt = rng.uniform(-1, 1, NTEST); Pt = e11.truth(xt, sc)
    Qtr, Qte = judge_probs(Ptr, par, rng), judge_probs(Pt, par, rng)      # judges see the item, not the human histogram
    if pool != "prob":
        fl = 1e-6 if pool == "votes3" else 5e-2
        Qtr, Qte = to_votes(Qtr, rng, fl), to_votes(Qte, rng, fl)
    S = score_rows(Qtr, P)
    pick = {"corr_pearson": int(np.argmax(S[0])), "corr_spearman": int(np.argmax(S[1])), "ce": int(np.argmin(S[2])), "rps": int(np.argmin(S[3]))}
    out = {name: toy.metrics(Qte[j], Pt, xt) for name, j in pick.items()}
    # oracle: best judge on test by RPS, and the constant forecast (train histogram)
    allm = [toy.metrics(Qte[j], Pt, xt) for j in range(M)]
    out["oracle_rps"] = allm[int(np.argmin([a["rps"] for a in allm]))]
    out["oracle_corr"] = allm[int(np.argmax([a["pearson"] if np.isfinite(a["pearson"]) else -1 for a in allm]))]
    out["constant"] = toy.metrics(np.tile(P.mean(0), (NTEST, 1)), Pt, xt)
    return sc, n, k, pool, seed, out

if __name__ == "__main__":
    RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"); os.makedirs(RES, exist_ok=True)
    OUT = os.path.join(RES, "rawB.jsonl"); done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            if line.strip():
                r = json.loads(line); done.add((r["scenario"], r["n"], r["k"], r["pool"], r["seed"]))
    jobs = [(s, n, k, p, sd) for n in NS for k in KS for p in POOLS for s in "ABCD" for sd in range(SEEDS) if (s, n, k, p, sd) not in done]
    PROCS = int(sys.argv[1]) if len(sys.argv) > 1 else 4; t0 = time.time(); print(len(jobs), "jobs", flush=True)
    with Pool(PROCS) as pool_, open(OUT, "a") as f:
        for i, (s, n, k, p, sd, o) in enumerate(pool_.imap_unordered(run_one, jobs, chunksize=8), 1):
            f.write(json.dumps({"scenario": s, "n": n, "k": k, "pool": p, "seed": sd, "fits": o}, default=float) + "\n")
            if i % 500 == 0 or i == len(jobs):
                f.flush(); print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    open(os.path.join(RES, "DONE_B"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
