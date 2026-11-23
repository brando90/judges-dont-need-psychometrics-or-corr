"""Independent plain-loop check of lib28 (11-23-2026): fits and scores written again with explicit per-rating loops.

Shares only the training-draw sampler (draw_training) with lib28. For one judge and set it refits all 27 transforms for each
draw of fold 0 at one budget per condition, scores each held-out item by explicit expectation over its human labels, and
compares the per-item means over the 200 draws with lib28.run_set to 1e-9.
Usage: python3 verify.py FORECASTS.csv LABELS.json JUDGE DATASET
"""
import csv, json, math, sys
import numpy as np
from lib28 import (COEFFICIENTS, DATASETS, DRAWS, METHODS, BUDGETS, draw_training, folds_of, run_set, TOL)


def qfam(p, t, b, c):
    K = len(p)
    p = [(1 - K * 1e-6) * x + 1e-6 for x in p]
    u = [(k - (K - 1) / 2) / ((K - 1) / 2) for k in range(K)]
    l = [t * math.log(p[k]) + b * u[k] + c * u[k] ** 2 for k in range(K)]
    m = max(l); e = [math.exp(x - m) for x in l]; s = sum(e)
    return [x / s for x in e]


def rps_one(q, y):          # exactly one human label y
    K = len(q); F = np.cumsum(q)
    return sum((F[k] - (1.0 if y <= k else 0.0)) ** 2 for k in range(K - 1)) / (K - 1)


def main(fc_path, lab_path, judge, ds):
    labels = json.load(open(lab_path))
    ids = [i for i in labels if i.startswith(ds + "/")]
    counts = np.array([labels[i]["counts"] for i in ids], float)
    groups = [labels[i]["group"] for i in ids]
    K = counts.shape[1]
    fc = {r["id"]: [float(r[f"p{j}"]) for j in range(K)] for r in csv.DictReader(open(fc_path)) if r["judge"] == judge and r["id"].startswith(ds + "/")}
    p = np.array([fc[i] for i in ids])
    k = DATASETS[ds]; fold = folds_of(groups, k)
    test = np.flatnonzero(fold == 0); pool = np.flatnonzero(fold != 0)
    worst = 0.0
    for d, B in ((1, 16), (5, 40)):
        n = B // d
        out, _, _, _ = run_set(p, counts, groups, k, d, np.arange(K, dtype=float))
        bi = BUDGETS[d].index(B)
        order, c = draw_training(counts, pool, d, k, 0, DRAWS)
        acc = np.zeros((len(test), 6))                     # mean rps over draws, per method
        for r in range(DRAWS):
            items = order[r, :n]
            ratings = [[kk for kk in range(K) for _ in range(int(c[r, j, kk]))] for j in range(n)]
            fam = [[qfam(p[i], *co) for co in COEFFICIENTS] for i in items]
            ce_l, rp_l, rho = [], [], []
            for ti in range(27):
                ce_l.append(np.mean([-math.log(fam[j][ti][y]) for j in range(n) for y in ratings[j]]))
                rp_l.append(np.mean([rps_one(fam[j][ti], y) for j in range(n) for y in ratings[j]]))
                mt = [sum(fam[j][ti][kk] * kk for kk in range(K)) for j in range(n)]
                yt = [np.mean(ratings[j]) for j in range(n)]
                rho.append(np.corrcoef(mt, yt)[0, 1] if np.ptp(mt) > TOL and np.ptp(yt) > TOL else -np.inf)
            pick = {}
            for name, arr, fn in (("ce", ce_l, min), ("rps", rp_l, min), ("pearson", rho, max)):
                best = fn(arr)
                pick[name] = next(ti for ti in range(27) if abs(arr[ti] - best) <= TOL or arr[ti] == best)
            if not np.isfinite(max(rho)):
                pick["pearson"] = 0
            allr = [y for rr in ratings for y in rr]
            for ei, e in enumerate(test):
                he = counts[e]
                for mname in ("ce", "pearson", "rps"):
                    q = qfam(p[e], *COEFFICIENTS[pick[mname]])
                    acc[ei, METHODS.index(mname)] += sum(he[y] / he.sum() * rps_one(q, y) for y in range(K))
                qu = [1 / K] * K
                acc[ei, METHODS.index("uniform")] += sum(he[y] / he.sum() * rps_one(qu, y) for y in range(K))
                hs = [(allr.count(y) + 0.5) / (len(allr) + 0.5 * K) for y in range(K)]
                acc[ei, METHODS.index("hist")] += sum(he[y] / he.sum() * rps_one(hs, y) for y in range(K))
        acc /= DRAWS
        for mname in ("ce", "pearson", "rps", "uniform", "hist"):
            mi = METHODS.index(mname)
            diff = np.abs(acc[:, mi] - out[test, bi, mi, 0]).max()
            worst = max(worst, diff)
            print(f"{ds} {judge} d={d} B={B} {mname}: max |loop - vectorized| over {len(test)} held-out items = {diff:.2e}")
    print("WORST", f"{worst:.2e}")
    assert worst < 1e-9


if __name__ == "__main__":
    main(*sys.argv[1:5])
