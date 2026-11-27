"""Re-implement the original PS inner CV (full-pool, as in study_a.py) to (a) check it reproduces the stored PS
choices and (b) count how often the PS choice is decided by its 'more items' tie-break (exact validation ties)."""
import sys, json, time
import numpy as np, pandas as pd
WORK = sys.argv[1]; sys.path.insert(0, WORK + "/src")
import study_a as A
bundle = A.load_bundle(); df = pd.read_csv(WORK + "/results/study_a_replications.csv")
out = {}
for name in A.DATASET_ORDER:
    d = bundle[name]; k = A.DATASET_K[name]; K = int(d["K"]); ratings = d["ratings"]; n_avail = d["n_avail"]
    dev_idx = np.asarray(d["dev_idx"], int); Q = A.transforms(d["p_mix"]); h_full = d["counts"] / d["counts"].sum(1, keepdims=True)
    sub = df[df.dataset == name].set_index("rep")
    stats_ = {B: {"match": 0, "tie": 0, "n": 0} for B in A.BUDGETS if len(A.designs_for_budget(B)) > 1}
    for rep in range(A.N_REPS):
        pilot_items = A.rng_ss(A.MASTER, 2, k, rep).choice(dev_idx, size=A.N_PILOT_ITEMS, replace=False)
        rng_fold = A.rng_ss(A.MASTER, 6, k, rep); fold = np.arange(A.N_PILOT_ITEMS) % A.N_FOLDS; rng_fold.shuffle(fold)
        val = {dd: [] for dd in A.DESIGNS}
        for f in range(A.N_FOLDS):
            vi = pilot_items[fold == f]; tp = pilot_items[fold != f]
            for d_id, (ni, nr) in enumerate(A.DESIGNS):
                for m in range(A.N_INNER):
                    rng = A.rng_ss(A.MASTER, 4, k, rep, f, d_id, m)
                    tr = rng.choice(tp, size=ni, replace=False)
                    cc, _, _, ok = A.sample_counts(ratings, n_avail, tr, nr, rng, K)
                    val[(ni, nr)].append(A.ce_fit_and_score(Q, tr, cc, vi, h_full[vi])[0])
                    A.ce_mix_fit_and_score(Q, tr, cc, vi, h_full[vi])  # keep identical RNG consumption (none used) 
        mv = {dd: float(np.mean(v)) for dd, v in val.items()}
        for B, s in stats_.items():
            cands = A.designs_for_budget(B)
            vals = np.array([mv[c] for c in cands]); best = vals.min()
            tied = (vals <= best + A.TOL).sum() > 1
            pick = None; bs = None
            for c in cands:
                v = mv[c]
                if pick is None or v < bs - A.TOL or (abs(v - bs) <= A.TOL and c[0] > pick[0]): pick, bs = c, v
            s["n"] += 1; s["tie"] += int(tied); s["match"] += int(f"{pick[0]}x{pick[1]}" == sub.loc[rep, f"B{B}_ps"])
    out[name] = {str(B): s for B, s in stats_.items()}
    print(name, {B: (s["match"], s["tie"]) for B, s in stats_.items()}, flush=True)
json.dump(out, open("/tmp/opus_review_exp39/outputs/ps_val_ties.json", "w"))
