#!/usr/bin/env python3
"""Experiment 47: calibration label-budget sweep on Experiment 41's heads (PROTOCOL.md, frozen before this script existed).
Usage: python label_budget_sweep.py OUT_DIR [--draws 20]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heads41 import build_views, heads, item_nrps_ce, map_from, paired_ci, SIGNAL  # noqa: E402

NS = [25, 50, 100, 200, 500, 1000, "all"]
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--draws", type=int, default=20); ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rows = []; t0 = time.time()
for vname, base, k, data in build_views():
    for seed in a.seeds:
        H = heads(vname, data, seed)
        val, test = H["val"], H["test"]
        ce_raw, _ = item_nrps_ce(H["ce_pt"], data.ratings, test, k)
        const, _ = item_nrps_ce(np.repeat(H["const"].reshape(1, -1), len(test), 0), data.ratings, test, k)
        for n in NS:
            nn = len(val) if n == "all" else min(int(n), len(val))
            for design in ("R1", "RA"):
                per_draw = {"M": [], "G": [], "E": []}; item_acc = {"M": 0, "G": 0}
                ndraw = 1 if (n == "all" and design == "RA") else a.draws
                for dr in range(ndraw):
                    rng = np.random.default_rng(np.random.SeedSequence([20261127, 47, seed, nn, dr, 0 if design == "R1" else 1]))
                    pick = val[rng.permutation(len(val))[:nn]] if not (n == "all" and design == "RA") else val
                    pos = {int(i): j for j, i in enumerate(val)}
                    if design == "R1":
                        tg = np.zeros((nn, k))
                        for r_, i in enumerate(pick):
                            ys = np.asarray(data.ratings[i], int); tg[r_, ys[rng.integers(len(ys))]] = 1.0
                    else:
                        tg = np.stack([np.bincount(np.asarray(data.ratings[i], int), minlength=k) / len(data.ratings[i]) for i in pick])
                    sc = {}
                    for obj in ("ce", "pearson"):
                        zc = H[f"{obj}_mv"][[pos[int(i)] for i in pick]]
                        q = map_from(zc, tg, H[f"{obj}_mt"])
                        sc[obj], _ = item_nrps_ce(q, data.ratings, test, k)
                    M_i, G_i = sc["pearson"] - sc["ce"], sc["pearson"] - ce_raw
                    per_draw["M"].append(M_i.mean()); per_draw["G"].append(G_i.mean()); per_draw["E"].append((sc["pearson"] - const).mean())
                    item_acc["M"] = item_acc["M"] + M_i / ndraw; item_acc["G"] = item_acc["G"] + G_i / ndraw
                ciM = paired_ci(item_acc["M"], np.zeros_like(item_acc["M"])); ciG = paired_ci(item_acc["G"], np.zeros_like(item_acc["G"]))
                rows.append({"view": vname, "base": base, "K": k, "seed": seed, "n_requested": n, "n_used": nn, "design": design, "draws": ndraw,
                             "signal": SIGNAL.get(base, True), "n_val": int(len(val)), "n_test": int(len(test)),
                             "M_mean": float(np.mean(per_draw["M"])), "M_sd": float(np.std(per_draw["M"], ddof=1)) if ndraw > 1 else 0.0,
                             "G_mean": float(np.mean(per_draw["G"])), "G_sd": float(np.std(per_draw["G"], ddof=1)) if ndraw > 1 else 0.0,
                             "E_mean": float(np.mean(per_draw["E"])), "M_ci": ciM[1:], "G_ci": ciG[1:],
                             "G_draws": [float(x) for x in per_draw["G"]], "M_draws": [float(x) for x in per_draw["M"]]})
        print(f"{vname} seed {seed} done {time.time()-t0:.0f}s", flush=True)
(out / "label_budget_rows.json").write_text(json.dumps(rows))
print("wrote", out / "label_budget_rows.json")
