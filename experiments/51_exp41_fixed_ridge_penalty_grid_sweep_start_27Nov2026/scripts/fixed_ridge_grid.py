#!/usr/bin/env python3
"""Experiment 51: Experiment 41 raw and matched gaps at every penalty of a fixed grid, no selection (PROTOCOL.md, frozen first).
Usage: python fixed_ridge_grid.py OUT_DIR [--lams ...]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
E47S = Path(__file__).resolve().parents[2] / "47_calibration_label_budget_sweep_exp41_heads_start_27Nov2026" / "scripts"
sys.path.insert(0, str(E47S))
from heads41 import build_views, fe, d, item_nrps_ce, map_from, paired_ci, histograms, SIGNAL  # noqa: E402
from e40_l2 import l2_settings  # noqa: E402  (Experiment 41's common penalty form)

ap = argparse.ArgumentParser(); ap.add_argument("out")
ap.add_argument("--lams", type=float, nargs="*", default=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
fe.MAXITER = 300
rows = []; t0 = time.time()
views = build_views()
for lam in a.lams:
    with l2_settings(lam):
        for vname, base, k, data in views:
            values = np.asarray(data.class_values, float)
            for seed in range(5):
                train, val, test = d.split_by_group(data.groups, seed=seed)
                x_train, (x_val, x_test), _ = fe.design_matrices(data.texts, train, [val, test])
                sc = {}
                for obj in ("ce", "pearson", "mse"):
                    pack = fe.fit_objective(x_train, data.ratings, train, data.class_values, obj, None, seed)
                    _, mv = fe.predict_distributions(pack, x_val, obj); pt, mt = fe.predict_distributions(pack, x_test, obj)
                    raw, _ = item_nrps_ce(pt, data.ratings, test, k)
                    cal, _ = item_nrps_ce(map_from(mv, histograms(data.ratings, val, k), mt), data.ratings, test, k)
                    sc[obj] = (raw, cal, pack.get("nit"))
                cprob, _ = fe.constant_marginal(data.ratings, train, k, values)
                const, _ = item_nrps_ce(np.repeat(cprob.reshape(1, -1), len(test), 0), data.ratings, test, k)
                rows.append({"lam": lam, "view": vname, "base": base, "K": k, "seed": seed, "signal": SIGNAL.get(base, True),
                             "raw_gap": paired_ci(sc["pearson"][0], sc["ce"][0]), "matched_gap": paired_ci(sc["pearson"][1], sc["ce"][1]),
                             "matched_mse_minus_ce": paired_ci(sc["mse"][1], sc["ce"][1]), "ce_raw_minus_const": paired_ci(sc["ce"][0], const),
                             "ce_cal_minus_const": paired_ci(sc["ce"][1], const),
                             "nit": {o: sc[o][2] for o in sc}})
            print(f"lam {lam:g} {vname} done {time.time()-t0:.0f}s", flush=True)
    (out / "fixed_ridge_rows.json").write_text(json.dumps(rows))
print("wrote", out / "fixed_ridge_rows.json")
