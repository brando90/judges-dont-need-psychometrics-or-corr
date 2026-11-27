"""Analytic expectations of the Study A item-mean and grand-mean MSE (finite-population sampling without
replacement, as in study_a.sample_counts) versus the observed replication means."""
import sys, json
import numpy as np
WORK = sys.argv[1]; sys.path.insert(0, WORK + "/src")
import study_a as A
bundle = A.load_bundle(); S = json.load(open(WORK + "/results/study_a_summary.json"))
worst = 0; n_cells = 0; order_ok = 0
for name in A.DATASET_ORDER:
    d = bundle[name]; dev = np.asarray(d["dev_idx"], int)
    Ni = d["n_avail"][dev].astype(float)
    s2 = np.array([d["ratings"][i, :d["n_avail"][i]].var() for i in dev])        # population (divisor N) variance per item
    mu = np.array([d["ratings"][i, :d["n_avail"][i]].mean() for i in dev]); Nd = len(dev)
    sig_u = mu.var()                                                                 # divisor Nd
    def e_item(nr):   # E over random item and random n_r-subset of (mean_s - mean_rest)^2
        return float(np.mean(Ni**2 * s2 / (nr * (Ni - nr) * (Ni - 1))))
    def e_grand(ni, nr):
        within = np.mean(s2 * (Ni - nr) / ((Ni - 1) * nr))
        return float((sig_u * (Nd - ni) / (Nd - 1) + within) / ni)
    for B in A.BUDGETS:
        cands = A.designs_for_budget(B)
        if len(cands) < 2: continue
        u1 = max(cands, key=lambda c: c[1]); u2 = max(cands, key=lambda c: c[0])
        cell = S["datasets"][name]["budgets"][str(B)]["metrics"]
        obs = {k: cell[m][k]["mean"] for k in ("u1", "u2") for m in ("item_mse",)}
        og = {k: cell["grand_mse"][k]["mean"] for k in ("u1", "u2")}
        ei = {"u1": e_item(u1[1]), "u2": e_item(u2[1])}; eg = {"u1": e_grand(*u1), "u2": e_grand(*u2)}
        n_cells += 1
        order_ok += (ei["u1"] < ei["u2"]) and (eg["u2"] < eg["u1"])
        rel = max(abs(obs[k] / ei[k] - 1) for k in ei)
        worst = max(worst, rel)
        if B in (64, 512):
            print(f"{name} B={B}: item MSE U1 obs {obs['u1']:.5f} vs analytic {ei['u1']:.5f}; U2 obs {obs['u2']:.5f} vs {ei['u2']:.5f} | "
                  f"grand MSE U1 obs {og['u1']:.5f} vs {eg['u1']:.5f}; U2 obs {og['u2']:.5f} vs {eg['u2']:.5f}")
print(f"cells: {n_cells}; analytic ordering (U1<U2 on item MSE and U2<U1 on grand MSE) holds in {order_ok}/{n_cells}; "
      f"max relative gap observed vs analytic item-MSE mean: {100*worst:.1f}%")
