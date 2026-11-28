#!/usr/bin/env python3
"""Experiment 52: bias and variance of the plug-in skill-score estimator, one vs several ratings per item (PROTOCOL.md, frozen first).
Synthetic law: a check of an estimator, not an empirical claim.  Usage: python skill_estimator_sim.py OUT_DIR [--reps 5000]
"""
import argparse, json
from pathlib import Path
import numpy as np

K = 5; CUTS = np.array([0.5, 1.5, 2.5, 3.5])
NS = [25, 50, 100, 200, 500, 1000, 5000]


def sig(x): return 1 / (1 + np.exp(-x))


def disc_logistic(mu, s):
    c = sig((CUTS[None, :] - mu[:, None]) / s[:, None])
    F = np.column_stack([np.zeros(len(mu)), c, np.ones(len(mu))])
    p = np.diff(F, axis=1); p = np.clip(p, 1e-9, None); return p / p.sum(1, keepdims=True)


def population(rng, N=200_000):
    mu = rng.uniform(0.5, 4.5, N); s = rng.uniform(0.4, 1.5, N)
    P = disc_logistic(mu, s)
    J = {"J1_noisy_calibrated": disc_logistic(mu + rng.normal(0, 0.5, N), s), "J2_overconfident": disc_logistic(mu, np.full(N, 0.25)),
         "J3_shifted": disc_logistic(mu + 0.8, s), "J4_oracle": P}
    return P, J


def score_matrix(q, name):
    """S(q, y) for every class y: shape (..., K)."""
    if name == "nrps":
        F = np.cumsum(q, -1)[..., :-1]; out = []
        for y in range(K):
            G = (np.arange(K - 1) >= y).astype(float); out.append(((F - G) ** 2).mean(-1))
        return np.stack(out, -1)
    if name == "brier":
        sq = (q * q).sum(-1, keepdims=True); return sq - 2 * q + 1
    if name == "ce":
        return -np.log(np.clip(q, 1e-12, 1))
    raise ValueError(name)


def smooth(counts):  # base-rate forecast from counts: additive 0.5 per class
    return (counts + 0.5) / (counts.sum(-1, keepdims=True) + 0.5 * K)


def run(out, reps, seed=(20261127, 52)):
    rng = np.random.default_rng(np.random.SeedSequence(list(seed)))
    P, J = population(rng)
    b = P.mean(0)
    SC = ("nrps", "brier", "ce")
    Sj = {(j, s): score_matrix(q, s) for j, q in J.items() for s in SC}  # (N, K)
    Sb = {s: score_matrix(b[None, :], s)[0] for s in SC}  # (K,)
    truth = {(j, s): float(1 - (P * Sj[j, s]).sum(1).mean() / (P @ Sb[s]).mean()) for j in J for s in SC}
    cdf = np.cumsum(P, 1)
    res = []
    for n in NS:
        acc = {}
        chunk = max(1, min(reps, 2_000_000 // (n * 5)))
        done = 0
        while done < reps:
            r = min(chunk, reps - done); done += r
            idx = rng.integers(0, len(P), size=(r, n))
            u = rng.random((r, n, 5))
            Y = (u[..., None] > cdf[idx][:, :, None, :]).sum(-1)  # (r, n, 5) ratings in 0..K-1
            Y = np.minimum(Y, K - 1)
            y1 = Y[:, :, 0]
            oh1 = np.eye(K)[y1]; oh5 = np.eye(K)[Y]  # (r,n,K), (r,n,5,K)
            c1 = oh1.sum(1); c5 = oh5.sum((1, 2))
            n5 = max(1, n // 5)
            Y5s = Y[:, :n5, :]; oh5s = np.eye(K)[Y5s]; c5s = oh5s.sum((1, 2))
            for s in SC:
                b_in1 = smooth(c1); b_in5 = smooth(c5); b_in5s = smooth(c5s)
                Sb_known = Sb[s][y1].mean(1)
                Sb_in1 = np.take_along_axis(score_matrix(b_in1, s), y1, 1).mean(1)
                loo = smooth(c1[:, None, :] - oh1)  # (r, n, K)
                Sb_loo = np.take_along_axis(score_matrix(loo, s), y1[..., None], 2)[..., 0].mean(1)
                Sb_in5 = score_matrix(b_in5, s)[np.arange(r)[:, None, None], Y].mean((1, 2))
                Sb_in5s = score_matrix(b_in5s, s)[np.arange(r)[:, None, None], Y5s].mean((1, 2))
                for j in J:
                    M = Sj[j, s][idx]  # (r, n, K)
                    S1 = np.take_along_axis(M, y1[..., None], 2)[..., 0].mean(1)
                    S5 = np.take_along_axis(M, Y, 2).mean((1, 2))
                    S5s = np.take_along_axis(M[:, :n5], Y5s, 2).mean((1, 2))
                    est = {"E1_one_rating_known_base": 1 - S1 / Sb_known, "E2_one_rating_insample_base": 1 - S1 / Sb_in1,
                           "E3_one_rating_leave_one_out_base": 1 - S1 / Sb_loo, "E4_five_ratings_same_items": 1 - S5 / Sb_in5,
                           "E5_five_ratings_equal_total": 1 - S5s / Sb_in5s, "numerator_one_rating": S1, "numerator_five_ratings": S5}
                    for e, v in est.items():
                        acc.setdefault((s, j, e), []).append(v)
        for (s, j, e), v in acc.items():
            v = np.concatenate(v)
            tgt = float((P * Sj[j, s]).sum(1).mean()) if e.startswith("numerator") else truth[j, s]
            bias = float(v.mean() - tgt); sd = float(v.std(ddof=1))
            res.append({"n": n, "score": s, "judge": j, "estimator": e, "target": tgt, "mean": float(v.mean()), "bias": bias, "sd": sd,
                        "rmse": float(np.sqrt(bias ** 2 + sd ** 2)), "mcse_bias": sd / np.sqrt(len(v)), "reps": int(len(v))})
        print("n", n, "done", flush=True)
    payload = {"truth": {f"{j}|{s}": v for (j, s), v in truth.items()}, "rows": res, "reps": reps, "seed": list(seed)}
    if out:
        Path(out).mkdir(parents=True, exist_ok=True); (Path(out) / "skill_estimator_rows.json").write_text(json.dumps(payload, indent=1))
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--reps", type=int, default=5000)
    a = ap.parse_args(); run(a.out, a.reps)
