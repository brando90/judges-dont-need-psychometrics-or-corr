"""Reviewer variant: proper-score selector restricted to the 80x16 pilot ratings only.
Same seeds as study_a.py for pilot items/ratings, folds and inner draws. Validation targets are the
16-rating pilot histograms; inner training ratings are subsampled from the item's 16 pilot ratings.
Deployment RPS of every candidate is read from the re-run study_a_replications.csv (same seeds)."""
import sys, json, time
import numpy as np, pandas as pd
WORK = sys.argv[1]; OUTP = sys.argv[2]
sys.path.insert(0, WORK + "/src")
import study_a as A
bundle = A.load_bundle()
df = pd.read_csv(WORK + "/results/study_a_replications.csv")
out = {}
t0 = time.time()
for name in A.DATASET_ORDER:
    data = bundle[name]; k = A.DATASET_K[name]; K = int(data["K"])
    ratings = data["ratings"]; n_avail = data["n_avail"]; dev_idx = np.asarray(data["dev_idx"], int)
    Q = A.transforms(data["p_mix"])
    sub = df[df.dataset == name].set_index("rep")
    rec = {str(B): {"ps_pilot": [], "u1_rps": [], "u2_rps": [], "ps_full_rps": [], "ps_pilot_rps": []} for B in A.BUDGETS}
    for rep in range(A.N_REPS):
        rng_items = A.rng_ss(A.MASTER, 2, k, rep); rng_rate = A.rng_ss(A.MASTER, 3, k, rep)
        pilot_items = rng_items.choice(dev_idx, size=A.N_PILOT_ITEMS, replace=False)
        X = A.build_pilot_table(ratings, n_avail, pilot_items, A.N_PILOT_RATERS, rng_rate).astype(int)
        pos = {int(it): j for j, it in enumerate(pilot_items)}
        hist = np.stack([np.bincount(X[j], minlength=K) for j in range(len(pilot_items))]) / A.N_PILOT_RATERS
        rng_fold = A.rng_ss(A.MASTER, 6, k, rep)
        fold = np.arange(A.N_PILOT_ITEMS) % A.N_FOLDS; rng_fold.shuffle(fold)
        val_rps = {d: [] for d in A.DESIGNS}
        for fold_id in range(A.N_FOLDS):
            val_items = pilot_items[fold == fold_id]; train_pool = pilot_items[fold != fold_id]
            val_h = hist[[pos[int(i)] for i in val_items]]
            for d_id, (n_i, n_r) in enumerate(A.DESIGNS):
                if n_i > len(train_pool): continue
                for m in range(A.N_INNER):
                    rng = A.rng_ss(A.MASTER, 4, k, rep, fold_id, d_id, m)
                    tr = rng.choice(train_pool, size=n_i, replace=False)
                    cc = np.zeros((n_i, K), int)
                    for t, it in enumerate(tr):
                        ch = rng.choice(A.N_PILOT_RATERS, size=n_r, replace=False)
                        cc[t] = np.bincount(X[pos[int(it)], ch], minlength=K)
                    rps, _, _, _ = A.ce_fit_and_score(Q, tr, cc, val_items, val_h)
                    val_rps[(n_i, n_r)].append(rps)
        mean_val = {d: (float(np.mean(v)) if v else np.inf) for d, v in val_rps.items()}
        row = sub.loc[rep]
        for B in A.BUDGETS:
            cands = A.designs_for_budget(B)
            best = None; bs = None
            for d in cands:  # same tie rule as study_a: lower; ties -> more items
                s = mean_val[d]
                if best is None or s < bs - A.TOL or (abs(s - bs) <= A.TOL and d[0] > best[0]):
                    best, bs = d, s
            r = rec[str(B)]
            r["ps_pilot"].append(f"{best[0]}x{best[1]}")
            r["ps_pilot_rps"].append(float(row[f"B{B}_cand_{best[0]}x{best[1]}_rps"]))
            r["u1_rps"].append(float(row[f"B{B}_u1_rps"])); r["u2_rps"].append(float(row[f"B{B}_u2_rps"]))
            r["ps_full_rps"].append(float(row[f"B{B}_ps_rps"]))
    out[name] = rec
    print(name, "done", round(time.time() - t0, 1), "s", flush=True)
json.dump(out, open(OUTP, "w"))
