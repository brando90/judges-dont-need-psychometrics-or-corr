#!/usr/bin/env python3
"""Experiment 41, NON-PREREGISTERED follow-up: does the K-sweep conclusion depend on the fixed L2 = 0.01?  (written 11-27-2026 after an independent review)

The preregistered K-sweep (k_sweep_matched_calibration.py) used Experiment 36's fixed L2 = 0.01 for every head.  The review showed that value is
arbitrary and that it shrinks the learned score toward the training-set marginal, so the raw gap between the Pearson-trained and the CE-trained head
partly measures how badly each head's own distribution is shrunk, and the MSE head's penalty is about 0.5 * N times weaker than the softmax heads'.
This script re-runs the same pipeline with a penalty grid and a validation-chosen penalty per head.  It is a SENSITIVITY analysis: the grid, the
selection rule and the reported statistics below were fixed in this file before it produced a number, but they are NOT in the preregistration
file, and the script was not committed beforehand.  The preregistered results stay as reported (labelled "L2 = 0.01, preregistered").

Pipeline per (view, split seed, objective in {ce, pearson, mse}, penalty lam in LAMBDA_GRID):
    fit the head on the TRAIN prompts (<= 300 L-BFGS iterations; the MSE head is closed form);
    score its OWN forecast on validation and on test ("raw");
    pass its scalar mean score through the validation-fitted multinomial-logit map (matched calibration, as in the preregistered run), score on both ("calibrated").
Selection (validation only, same rule for every head): lam_raw   = argmin over the grid of validation normalized RPS of the raw forecast,
                                                         lam_cal   = argmin over the grid of validation normalized RPS of the calibrated forecast
                                                         (the calibration map is fitted on the same validation items, so this score is slightly optimistic and
                                                          equally so for every head; it only picks lam, and every test number is out of sample).
Reported: test normalized RPS gaps at the selected penalties with paired item-clustered 95% bootstrap intervals, and the same gaps at every COMMON penalty.
Normalized RPS = RPS / (K - 1), 0 to 1, lower is better.  gap = test nRPS of the Pearson- (or MSE-) trained head minus test nRPS of the CE-trained head; positive means
the CE-trained head is better.

Usage: OMP_NUM_THREADS=1 python3 -I k_sweep_l2_sensitivity.py --shard 0 --of 2     (writes ../results/k_sweep_l2_sensitivity_shard0of2.json)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python3 -I does not add the script directory

import numpy as np

from e40_common import EXP40, d, fe, jsonable, paired_ci, per_item_on, three_level
from e40_l2 import LAMBDA_GRID, calibrate_both, l2_settings

ap = argparse.ArgumentParser()
ap.add_argument("--maxiter", type=int, default=300)
ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--of", type=int, default=1)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--only", nargs="*", default=None)
args = ap.parse_args()
fe.MAXITER = args.maxiter

OUT = EXP40 / "results"
OUT.mkdir(exist_ok=True)
OBJECTIVES = ("ce", "pearson", "mse")


def build_views():
    views = []
    pop_off = d.load_popquorn("offensiveness")
    pop_pol = d.load_popquorn("politeness")
    se_exp = d.load_summeval("expert")
    se_crd = d.load_summeval("crowd")
    d990 = d._load_dices("990")
    chaos3 = d.load_chaos_3way()
    chaosa = d.load_chaos_alpha()
    for base in (pop_off, pop_pol, se_exp, se_crd):
        views.append((base.name + "|K2", base.name, 2, d.binarize(base, d.SUMMEVAL_BINARY_CUT, base.name + "_K2")))
        views.append((base.name + "|K3", base.name, 3, three_level(base, base.name + "_K3")))
        views.append((base.name + "|K5", base.name, 5, base))
    views.append(("dices990|K2", "dices990", 2, d.binarize(d990, 2, "dices990_K2_yes")))
    views.append(("dices990|K3", "dices990", 3, d990))
    views.append(("chaosnli_snli_mnli|K3", "chaosnli_snli_mnli", 3, chaos3))
    views.append(("chaosnli_alphanli|K2", "chaosnli_alphanli", 2, chaosa))
    return views  # VeriBench is in a repository this machine cannot read (HTTP 404), as in the preregistered run


def entropy_strata(ratings, test_idx, k, seed=7, min_k=4):
    rng = np.random.default_rng(seed)
    pos, bsets, ent = [], [], []
    for p, i in enumerate(test_idx):
        y = np.asarray(ratings[i], int)
        if y.size < min_k:
            continue
        perm = rng.permutation(y.size)
        h = y.size // 2
        a, b = y[perm[:h]], y[perm[h:]]
        pa = np.bincount(a, minlength=k) / a.size
        nz = pa[pa > 0]
        pos.append(p)
        bsets.append(b)
        ent.append(float(-(nz * np.log(nz)).sum()))
    return np.asarray(pos, int), bsets, np.asarray(ent)


def run_view(vname, base, k, data, seeds):
    values = np.asarray(data.class_values, float)
    rows = []
    for seed in seeds:
        t0 = time.time()
        train, val, test = d.split_by_group(data.groups, seed=seed)
        d.assert_no_prompt_leakage(data.groups, data.rater_ids, train, val, test)
        x_train, (x_val, x_test), _rank = fe.design_matrices(data.texts, train, [val, test])
        cprob, cmu = fe.constant_marginal(data.ratings, train, k, values)
        crec = fe._item_records(np.repeat(cprob.reshape(1, -1), len(test), 0), np.full(len(test), cmu), data.ratings, test, values)
        const_test = crec["rps"] / (k - 1)
        grid = {}   # grid[obj][lam] = dict of means and (kept in memory only) per-item test vectors and probability matrices
        for obj in OBJECTIVES:
            grid[obj] = {}
            for lam in LAMBDA_GRID:
                with l2_settings(lam):
                    pack = fe.fit_objective(x_train, data.ratings, train, data.class_values, obj, None, seed)
                pv, mv = fe.predict_distributions(pack, x_val, obj)
                pt, mt = fe.predict_distributions(pack, x_test, obj)
                raw_v = fe._item_records(pv, mv, data.ratings, val, values)
                raw_t = fe._item_records(pt, mt, data.ratings, test, values)
                qv, qt = calibrate_both(mv, mt, data.ratings, val, k)
                cal_v = fe._item_records(qv, qv @ values, data.ratings, val, values)
                cal_t = fe._item_records(qt, qt @ values, data.ratings, test, values)
                grid[obj][lam] = {
                    "val_raw": float((raw_v["rps"] / (k - 1)).mean()), "val_cal": float((cal_v["rps"] / (k - 1)).mean()),
                    "test_raw": float((raw_t["rps"] / (k - 1)).mean()), "test_cal": float((cal_t["rps"] / (k - 1)).mean()),
                    "nit": pack.get("nit"), "success": pack.get("success"),
                    "hit_cap": bool(obj != "mse" and pack.get("nit") is not None and pack["nit"] >= args.maxiter),
                    "_v": {"raw": raw_t["rps"] / (k - 1), "cal": cal_t["rps"] / (k - 1), "raw_ce": raw_t["ce"], "cal_ce": cal_t["ce"]},
                    "_p": {"raw": pt, "cal": qt}}
        chosen = {"raw": {o: min(LAMBDA_GRID, key=lambda l: grid[o][l]["val_raw"]) for o in OBJECTIVES},
                  "cal": {o: min(LAMBDA_GRID, key=lambda l: grid[o][l]["val_cal"]) for o in OBJECTIVES}}

        def vec(obj, lam, kind):
            return grid[obj][lam]["_v"][kind]

        def gap(a, b):
            m, lo, hi = paired_ci(a, b, boot=args.boot)
            return {"mean": m, "lo": lo, "hi": hi}

        gaps = {
            "raw_pearson_minus_ce": gap(vec("pearson", chosen["raw"]["pearson"], "raw"), vec("ce", chosen["raw"]["ce"], "raw")),
            "matched_pearson_minus_ce": gap(vec("pearson", chosen["cal"]["pearson"], "cal"), vec("ce", chosen["cal"]["ce"], "cal")),
            "matched_mse_minus_ce": gap(vec("mse", chosen["cal"]["mse"], "cal"), vec("ce", chosen["cal"]["ce"], "cal")),
            "matched_mse_minus_pearson": gap(vec("mse", chosen["cal"]["mse"], "cal"), vec("pearson", chosen["cal"]["pearson"], "cal")),
            "ce_raw_minus_ce_calibrated": gap(vec("ce", chosen["raw"]["ce"], "raw"), vec("ce", chosen["cal"]["ce"], "cal")),
            "constant_minus_ce_raw": gap(const_test, vec("ce", chosen["raw"]["ce"], "raw")),
            "constant_minus_ce_calibrated": gap(const_test, vec("ce", chosen["cal"]["ce"], "cal"))}
        ce_r, ce_c = vec("ce", chosen["raw"]["ce"], "raw_ce"), vec("ce", chosen["cal"]["ce"], "cal_ce")
        gaps["ce_raw_minus_ce_calibrated"]["ce_nats"] = dict(zip(("mean", "lo", "hi"), paired_ci(ce_r, ce_c, boot=args.boot)))
        # the same gaps at every common penalty (means only; the intervals are reported at the selected penalties)
        fixed = {}
        for lam in LAMBDA_GRID:
            fixed[str(lam)] = {
                "raw_pearson_minus_ce": float((vec("pearson", lam, "raw") - vec("ce", lam, "raw")).mean()),
                "matched_pearson_minus_ce": float((vec("pearson", lam, "cal") - vec("ce", lam, "cal")).mean()),
                "matched_mse_minus_ce": float((vec("mse", lam, "cal") - vec("ce", lam, "cal")).mean()),
                "raw_mse_minus_ce": float((vec("mse", lam, "raw") - vec("ce", lam, "raw")).mean()),
                "ce_raw": float(vec("ce", lam, "raw").mean()), "ce_cal": float(vec("ce", lam, "cal").mean()),
                "pearson_cal": float(vec("pearson", lam, "cal").mean()), "mse_cal": float(vec("mse", lam, "cal").mean()),
                "constant": float(const_test.mean())}
        # entropy strata on held-out test items with >= 4 ratings, scored on half B, at the selected penalties
        strata = {}
        pos, bsets, ent = entropy_strata(data.ratings, test, k)
        if pos.size >= 40:
            med = float(np.median(ent))
            low = ent <= med
            if low.all() or (~low).all():
                low = ent < med if (ent < med).any() else low
            loss = {}
            for obj, kind, sel in (("ce", "raw", "raw"), ("pearson", "raw", "raw"), ("ce", "cal", "cal"), ("pearson", "cal", "cal")):
                lam = chosen[sel][obj]
                loss[(obj, kind)] = per_item_on(grid[obj][lam]["_p"][kind][pos], data.ratings, test[pos], bsets, k)[1]
            for sname, mask in (("consensus", low), ("disputed", ~low)):
                if mask.sum() < 20:
                    continue
                strata[sname] = {"n_items": int(mask.sum()),
                                 "raw_pearson_minus_ce": dict(zip(("mean", "lo", "hi"), paired_ci(loss[("pearson", "raw")][mask], loss[("ce", "raw")][mask], boot=args.boot))),
                                 "matched_pearson_minus_ce": dict(zip(("mean", "lo", "hi"), paired_ci(loss[("pearson", "cal")][mask], loss[("ce", "cal")][mask], boot=args.boot)))}
        for obj in OBJECTIVES:
            for lam in LAMBDA_GRID:
                grid[obj][lam].pop("_v")
                grid[obj][lam].pop("_p")
        rows.append({"view": vname, "base": base, "K": k, "seed": seed, "n_train": int(train.size), "n_val": int(val.size), "n_test": int(test.size),
                     "constant_test_nrps": float(const_test.mean()), "chosen_lambda": chosen, "grid": {o: {str(l): v for l, v in g.items()} for o, g in grid.items()},
                     "gaps_at_chosen": gaps, "gaps_at_common_lambda": fixed, "strata_at_chosen": strata, "seconds": time.time() - t0})
        g = gaps
        print(f"  {vname:30s} seed {seed}: lam_raw {chosen['raw']} lam_cal {chosen['cal']} | raw gap {g['raw_pearson_minus_ce']['mean']:+.4f}"
              f" matched gap {g['matched_pearson_minus_ce']['mean']:+.4f} mse-ce {g['matched_mse_minus_ce']['mean']:+.4f} | {time.time() - t0:.0f}s", flush=True)
    return rows


def main():
    t0 = time.time()
    views = build_views()
    if args.only:
        views = [v for v in views if any(o in v[0] for o in args.only)]
    views = [v for i, v in enumerate(views) if i % args.of == args.shard]
    rows, summaries = [], {}
    for vname, base, k, data in views:
        print(f"[{vname}] items {data.n_items}, K {data.n_classes}", flush=True)
        summaries[vname] = data.summary()
        rows.extend(run_view(vname, base, k, data, args.seeds))
    payload = {"preregistered": False, "maxiter": args.maxiter, "seeds": args.seeds, "boot": args.boot, "lambda_grid": LAMBDA_GRID,
               "selection": "lam chosen per head on VALIDATION normalized RPS of the deployed forecast (raw forecast for 'raw', calibrated forecast for 'cal')",
               "views": summaries, "rows": rows, "elapsed_seconds": time.time() - t0}
    out = OUT / f"k_sweep_l2_sensitivity_shard{args.shard}of{args.of}.json"
    out.write_text(json.dumps(jsonable(payload), indent=1))
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
