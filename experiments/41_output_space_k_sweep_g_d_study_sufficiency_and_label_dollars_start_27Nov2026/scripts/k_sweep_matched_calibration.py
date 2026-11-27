#!/usr/bin/env python3
"""Real-data K-sweep with matched calibration and human-entropy strata (Experiment 41).

Question: on real multi-annotator ratings, does the output-space size K change how a Pearson-trained head compares with a
cross-entropy-trained head, once both pass through the SAME validation-fitted calibration map?

Predictions and decision rules were fixed in ../PREREGISTERED_PREDICTIONS_11-27-2026.md before this script produced a number.

Reads: Experiment 36's loaders/heads (imported unchanged). Writes: ../results/k_sweep_matched_calibration_maxiter<M>.json and .md.
Usage: python3 k_sweep_matched_calibration.py [--maxiter 35] [--seeds 0 1 2 3 4] [--only name ...]
All numbers printed here are produced by this script (deterministic: fixed split seeds, fixed bootstrap seed, fixed entropy-split seed).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python3 -I does not add the script directory

import numpy as np

from e40_common import (EXP40, calibrate, d, fe, jsonable, paired_ci, per_item_on, rebin, three_level)
from metrics import softmax  # noqa: E402  (path set in e40_common)

ap = argparse.ArgumentParser()
ap.add_argument("--maxiter", type=int, default=35)
ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
ap.add_argument("--only", nargs="*", default=None)
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()
fe.MAXITER = args.maxiter  # _fit_softmax reads this module global at call time

OUT = EXP40 / "results"
OUT.mkdir(exist_ok=True)


UNAVAILABLE = {}


def build_views():
    """(view name, base name, K, JudgingSet) for every re-binned view that this machine can rebuild."""
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
    try:
        vb = d.load_veribench_fsc_rv1("all5")
    except Exception as exc:  # the pinned VeriBench file is in a repository this machine cannot read
        UNAVAILABLE["veribench_fsc_rv1"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        print(f"[unavailable] veribench_fsc_rv1: {UNAVAILABLE['veribench_fsc_rv1']}", flush=True)
    else:
        views.append(("veribench_fsc_rv1|K2", "veribench_fsc_rv1", 2, d.binarize(vb, d.VERIBENCH_BINARY_CUT, "vb_rv1_K2")))
        views.append(("veribench_fsc_rv1|K3", "veribench_fsc_rv1", 3, three_level(vb, "vb_rv1_K3")))
        views.append(("veribench_fsc_rv1|K6", "veribench_fsc_rv1", 6, vb))
    return views


def head_outputs(data, prep, obj, seed):
    pack = fe.fit_objective(prep["x_train"], data.ratings, prep["train"], data.class_values, obj, None, seed)
    pv, mv = fe.predict_distributions(pack, prep["x_val"], obj)
    pt, mt = fe.predict_distributions(pack, prep["x_test"], obj)
    return pack, (pt, mt, mv)


def entropy_strata(ratings, test_idx, k, seed=7, min_k=4):
    """Positions (in test_idx) kept, half-B rating arrays, and boolean 'disputed' flags from half-A entropy."""
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
    assert k == data.n_classes, (vname, k, data.n_classes)
    rows = []
    for seed in seeds:
        train, val, test = d.split_by_group(data.groups, seed=seed)
        d.assert_no_prompt_leakage(data.groups, data.rater_ids, train, val, test)
        x_train, (x_val, x_test), rank = fe.design_matrices(data.texts, train, [val, test])
        prep = {"train": train, "val": val, "test": test, "x_train": x_train, "x_val": x_val, "x_test": x_test}
        # constant (train marginal) reference
        cprob, cmu = fe.constant_marginal(data.ratings, train, k, values)
        crec = fe._item_records(np.repeat(cprob.reshape(1, -1), len(test), 0), np.full(len(test), cmu), data.ratings, test, values)
        per = {"constant": {"raw": crec["rps"] / (k - 1), "raw_ce": crec["ce"]}}
        meta = {}
        outs = {}
        for obj in ("ce", "pearson", "mse"):
            pack, (pt, mt, mv) = head_outputs(data, prep, obj, seed)
            outs[obj] = (pt, mt, mv)
            raw = fe._item_records(pt, mt, data.ratings, test, values)
            qc = calibrate(mv, mt, data.ratings, val, k)
            cal = fe._item_records(qc, qc @ values, data.ratings, test, values)
            per[obj] = {"raw": raw["rps"] / (k - 1), "cal": cal["rps"] / (k - 1), "raw_ce": raw["ce"], "cal_ce": cal["ce"]}
            outs[obj] = (pt, mt, mv, qc)
            meta[obj] = {"nit": pack.get("nit"), "success": pack.get("success"),
                         "hit_cap": bool(pack.get("nit") is not None and obj in ("ce", "pearson") and pack.get("nit") >= args.maxiter)}
        row = {"view": vname, "base": base, "K": k, "seed": seed, "n_train": int(train.size), "n_val": int(val.size),
               "n_test": int(test.size), "median_k_ratings": float(np.median([len(r) for r in data.ratings])),
               "optimizer": meta}
        row["mean"] = {key: {v: float(per[key][v].mean()) for v in per[key]} for key in per}
        row["gaps"] = {}
        for tag, a_key, b_key in (("raw_pearson_minus_ce", ("pearson", "raw"), ("ce", "raw")),
                                  ("matched_pearson_minus_ce", ("pearson", "cal"), ("ce", "cal")),
                                  ("ce_raw_minus_ce_calibrated", ("ce", "raw"), ("ce", "cal")),
                                  ("raw_mse_minus_ce", ("mse", "raw"), ("ce", "raw")),
                                  ("matched_mse_minus_ce", ("mse", "cal"), ("ce", "cal"))):
            m, lo, hi = paired_ci(per[a_key[0]][a_key[1]], per[b_key[0]][b_key[1]], boot=args.boot)
            row["gaps"][tag] = {"nrps": {"mean": m, "lo": lo, "hi": hi}}
            ce_a = per[a_key[0]][a_key[1] + "_ce"]
            ce_b = per[b_key[0]][b_key[1] + "_ce"]
            if np.isfinite(ce_a).all() and np.isfinite(ce_b).all():
                m2, lo2, hi2 = paired_ci(ce_a, ce_b, boot=args.boot)
                row["gaps"][tag]["ce_nats"] = {"mean": m2, "lo": lo2, "hi": hi2}
            else:
                # A two-point forecast (the MSE head before calibration) gives probability 0 to some observed ratings.
                row["gaps"][tag]["ce_nats"] = {"mean": None, "lo": None, "hi": None, "note": "infinite cross-entropy on at least one item"}
        m, lo, hi = paired_ci(per["constant"]["raw"], per["ce"]["raw"], boot=args.boot)
        row["gaps"]["constant_minus_ce_raw"] = {"nrps": {"mean": m, "lo": lo, "hi": hi}}
        # entropy strata, scored on half B only
        pos, bsets, ent = entropy_strata(data.ratings, test, k)
        if pos.size >= 40:
            med = float(np.median(ent))
            low = ent <= med
            if low.all() or (~low).all():
                low = ent < med if (ent < med).any() else low
            strata = {}
            heads_probs = {}
            for obj in ("ce", "pearson"):
                pt, mt, mv, qc = outs[obj]
                heads_probs[(obj, "raw")] = pt[pos]
                heads_probs[(obj, "cal")] = qc[pos]
            loss = {}
            for key, probs in heads_probs.items():
                loss[key] = per_item_on(probs, data.ratings, test[pos], bsets, k)  # (ce, nrps)
            for sname, mask in (("consensus", low), ("disputed", ~low)):
                if mask.sum() < 20:
                    continue
                s = {"n_items": int(mask.sum()), "median_entropy_cut": med}
                for tag, a_key, b_key in (("raw_pearson_minus_ce", ("pearson", "raw"), ("ce", "raw")),
                                          ("matched_pearson_minus_ce", ("pearson", "cal"), ("ce", "cal"))):
                    m, lo, hi = paired_ci(loss[a_key][1][mask], loss[b_key][1][mask], boot=args.boot)
                    s[tag] = {"mean": m, "lo": lo, "hi": hi}
                strata[sname] = s
            row["strata"] = strata
        rows.append(row)
        g = row["gaps"]
        print(f"  {vname:34s} seed {seed}: constant {row['mean']['constant']['raw']:.4f} | CE raw {row['mean']['ce']['raw']:.4f} cal {row['mean']['ce']['cal']:.4f}"
              f" | Pearson raw {row['mean']['pearson']['raw']:.4f} cal {row['mean']['pearson']['cal']:.4f}"
              f" | raw gap {g['raw_pearson_minus_ce']['nrps']['mean']:+.4f} matched gap {g['matched_pearson_minus_ce']['nrps']['mean']:+.4f}"
              f" | Pearson hit cap {meta['pearson']['hit_cap']}", flush=True)
    return rows


def main():
    t0 = time.time()
    views = build_views()
    if args.only:
        views = [v for v in views if any(o in v[0] for o in args.only)]
    all_rows = []
    summaries = {}
    for vname, base, k, data in views:
        print(f"[{vname}] items {data.n_items}, K {data.n_classes}", flush=True)
        summaries[vname] = data.summary()
        all_rows.extend(run_view(vname, base, k, data, args.seeds))
    payload = {"maxiter": args.maxiter, "seeds": args.seeds, "boot": args.boot, "views": summaries, "unavailable": UNAVAILABLE, "rows": all_rows,
               "elapsed_seconds": time.time() - t0}
    out = OUT / f"k_sweep_matched_calibration_maxiter{args.maxiter}.json"
    out.write_text(json.dumps(jsonable(payload), indent=1))
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
