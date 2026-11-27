import sys
import numpy as np, pandas as pd
WORK = sys.argv[1]; sys.path.insert(0, WORK + "/src")
import study_a as A
df = pd.read_csv(WORK + "/results/study_a_replications.csv")
print("dataset B | P(max-items RPS < U1 RPS) | P(equal) | P(U1 < max-items) | item-oracle exact ties | grand-oracle exact ties")
agg = np.zeros(3); n = 0
for name in A.DATASET_ORDER:
    sub = df[df.dataset == name]
    for B in A.BUDGETS:
        cands = A.designs_for_budget(B)
        if len(cands) < 2: continue
        u1 = max(cands, key=lambda c: c[1]); mi = max(cands, key=lambda c: c[0])
        a = sub[f"B{B}_cand_{mi[0]}x{mi[1]}_rps"].to_numpy(); b = sub[f"B{B}_cand_{u1[0]}x{u1[1]}_rps"].to_numpy()
        lt = (a < b - 1e-12).mean(); eq = (np.abs(a - b) <= 1e-12).mean(); gt = (a > b + 1e-12).mean()
        agg += [lt, eq, gt]; n += 1
        def ties(metric):
            R = np.stack([sub[f"B{B}_cand_{x}x{y}_{metric}"].to_numpy() for x, y in cands], 1)
            return ((R <= R.min(1, keepdims=True) + 1e-12).sum(1) > 1).mean()
        print(f"{name} {B} | {lt:.2f} | {eq:.2f} | {gt:.2f} | {ties('item_mse'):.2f} | {ties('grand_mse'):.2f}")
print("average over 20 cells: max-items better %.2f, exact tie %.2f, U1 better %.2f" % tuple(agg / n))
