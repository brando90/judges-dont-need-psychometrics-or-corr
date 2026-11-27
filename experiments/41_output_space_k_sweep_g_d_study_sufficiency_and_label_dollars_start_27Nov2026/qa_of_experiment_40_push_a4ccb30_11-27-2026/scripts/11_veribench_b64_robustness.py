"""Robustness of the one headline cell: VeriBench native_main, exp27_split, B = 64,
D-study k = 2 (N = 32) versus test-oracle k = 3 (N = 18).

(a) convergence of each fitted cell (L-BFGS-B success flag and iterations at the 35 cap);
(b) the same difference after refitting both cells with maxiter = 300 (converged optimizer);
(c) design variability the fixed-head interval ignores: 200 alternative seeds for which train
    artifacts enter the prefix and which ratings are kept (seeded permutation + soft-target RNG);
(d) a retraining bootstrap: resample train TASKS and test TASKS with replacement (10 + 10),
    refit both cells inside every replicate (200 replicates), percentile 95% interval.
Usage: python3 11_veribench_b64_robustness.py <exp40_dir>
"""
import sys
from pathlib import Path

import numpy as np

EXP40 = Path(sys.argv[1])
sys.path.insert(0, str(EXP40))
import run_all as ra  # noqa: E402
import fit_eval  # noqa: E402

data = ra.load_veribench_fsc_native_main()
sp = ra.exp27_split(data)
train, val, test = sp["train"], sp["val"], sp["test"]
feats = ra.hash_features(data.texts)
groups = np.asarray(data.groups)
print(f"train={len(train)} artifacts / {len(set(groups[train]))} tasks; val={len(val)}; test={len(test)} / {len(set(groups[test]))} tasks")
sizes = np.array([data.ratings[int(i)].size for i in train])
print("train ratings-per-artifact counts:", {int(s): int((sizes == s).sum()) for s in sorted(set(sizes))},
      "; total train ratings:", int(sizes.sum()))


def cell_ce(train_idx, test_idx, k, B, perm_seed=0, target_seed=0, maxiter=35):
    elig = np.asarray([int(i) for i in train_idx if data.ratings[int(i)].size >= k], dtype=int)
    order = np.random.default_rng(perm_seed).permutation(elig)
    n_take = min(order.size, B // k)
    if n_take < 4:
        return None, None
    chosen = order[:n_take]
    x_tr, projected, _ = ra._svd_project(feats[chosen], [feats[test_idx]])
    old = fit_eval.MAXITER
    fit_eval.MAXITER = maxiter
    try:
        pack = ra.fit_objective(x_tr, data.ratings, chosen, data.class_values, "ce", k, seed=target_seed)
    finally:
        fit_eval.MAXITER = old
    ce, _ = ra.item_losses(pack, projected[0], data.ratings, test_idx)
    return ce, pack


print("\n(a) convergence of the committed cells at maxiter=35, and (b) refit at maxiter=300:")
res = {}
for k, B in ((1, 64), (2, 64), (3, 64), (1, 32), (2, 32), (3, 32)):
    ce35, p35 = cell_ce(train, test, k, B)
    ce300, p300 = cell_ce(train, test, k, B, maxiter=300)
    res[(k, B)] = (ce35, ce300)
    print(f"  k={k} B={B}: N={min(sum(1 for i in train if data.ratings[int(i)].size >= k), B // k)} "
          f"test CE@35={ce35.mean():.4f} (success={p35['success']}, nit={p35['nit']}) | "
          f"test CE@300={ce300.mean():.4f} (success={p300['success']}, nit={p300['nit']})")
d35 = res[(2, 64)][0] - res[(3, 64)][0]
d300 = res[(2, 64)][1] - res[(3, 64)][1]
print(f"  B=64 difference k2-k3: maxiter35 {d35.mean():.4f}; maxiter300 {d300.mean():.4f}")

print("\n(c) 200 alternative design seeds (permutation of eligible train artifacts and rating subsample):")
diffs, wins, dmin = [], 0, []
for s in range(200):
    c2, _ = cell_ce(train, test, 2, 64, perm_seed=s, target_seed=s)
    c3, _ = cell_ce(train, test, 3, 64, perm_seed=s, target_seed=s)
    c1, _ = cell_ce(train, test, 1, 64, perm_seed=s, target_seed=s)
    diffs.append(c2.mean() - c3.mean())
    dmin.append(c2.mean() - min(c1.mean(), c2.mean(), c3.mean()))
diffs = np.asarray(diffs)
print(f"  CE(k=2,N=32) - CE(k=3,N=18): mean {diffs.mean():.4f}, sd {diffs.std(ddof=1):.4f}, "
      f"2.5/50/97.5% = {np.quantile(diffs, [0.025, 0.5, 0.975]).round(4).tolist()}, fraction > 0: {(diffs > 0).mean():.3f}")
print(f"  seed 0 (committed design): {diffs[0]:.4f}; D-study minus best-of-k1,2,3: mean {np.mean(dmin):.4f}, "
      f"fraction where k=2 is the best: {(np.asarray(dmin) == 0).mean():.3f}")

print("\n(d) retraining task-cluster bootstrap (200 replicates; both heads refit in every replicate):")
rng = np.random.default_rng(11)
tr_tasks = sorted(set(groups[train]))
te_tasks = sorted(set(groups[test]))
tr_members = {t: train[groups[train] == t] for t in tr_tasks}
te_members = {t: test[groups[test] == t] for t in te_tasks}
reps = []
for r in range(200):
    tr = np.concatenate([tr_members[tr_tasks[j]] for j in rng.integers(0, len(tr_tasks), len(tr_tasks))])
    te = np.concatenate([te_members[te_tasks[j]] for j in rng.integers(0, len(te_tasks), len(te_tasks))])
    c2, _ = cell_ce(tr, te, 2, 64, perm_seed=r)
    c3, _ = cell_ce(tr, te, 3, 64, perm_seed=r)
    if c2 is None or c3 is None:
        continue
    reps.append(c2.mean() - c3.mean())
reps = np.asarray(reps)
print(f"  replicates={reps.size}; difference percentile 95% = {np.quantile(reps, [0.025, 0.975]).round(4).tolist()}; "
      f"median {np.median(reps):.4f}; fraction > 0: {(reps > 0).mean():.3f}")
