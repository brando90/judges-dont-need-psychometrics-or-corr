"""How often is the RPS oracle decided by the 'more items' tie-break (exact test-RPS ties)?"""
import sys, json
import numpy as np, pandas as pd
WORK = sys.argv[1]; sys.path.insert(0, WORK + "/src")
import study_a as A
df = pd.read_csv(WORK + "/results/study_a_replications.csv")
tot_tie = 0; tot = 0
print("dataset B | reps with an exact tie for best RPS | U1 agrees with RPS oracle (as reported) | U1 co-optimal (tied or best) | max-items design co-optimal | max-items is unique best")
for name in A.DATASET_ORDER:
    sub = df[df.dataset == name]
    for B in A.BUDGETS:
        cands = A.designs_for_budget(B)
        if len(cands) < 2: continue
        R = np.stack([sub[f"B{B}_cand_{a}x{b}_rps"].to_numpy() for a, b in cands], 1)
        best = R.min(1, keepdims=True); co = R <= best + 1e-12
        u1 = max(range(len(cands)), key=lambda j: cands[j][1]); mi = max(range(len(cands)), key=lambda j: cands[j][0])
        tie = (co.sum(1) > 1)
        agree = (sub[f"B{B}_u1"] == sub[f"B{B}_rps_oracle"]).mean()
        uniq_mi = (co[:, mi] & (co.sum(1) == 1)).mean()
        tot_tie += tie.sum(); tot += len(tie)
        print(f"{name} {B} | {tie.mean():.2f} | {agree:.2f} | {co[:, u1].mean():.2f} | {co[:, mi].mean():.2f} | {uniq_mi:.2f}")
print("overall fraction of comparable replications with an exact tie for best test RPS: %.3f" % (tot_tie / tot))
