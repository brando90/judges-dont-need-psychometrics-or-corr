#!/usr/bin/env python3
"""Experiment 41, NON-PREREGISTERED follow-up (11-27-2026): learning curves, equal-budget allocation and test-loss components with a VALIDATION-CHOSEN L2.

Why.  gd_study_and_allocation.py trained every head with Experiment 36's fixed L2 = 0.01.  An independent review (11-27-2026) showed that a fixed penalty does not
shrink as the training set grows and that the 24 principal-component features are small (standard deviation 0.077 to 0.167), so the L2 = 0.01 learning curves are
flattened by the penalty (POPQUORN politeness, K = 5: test normalized RPS 0.1738 / 0.1718 / 0.1716 at N = 128 / 512 / 2230 at L2 = 0.01, but
0.1854 / 0.1680 / 0.1637 at L2 = 1e-4).  "Saturates at 128 to 512" therefore described the penalty, not the data.  This script redoes the L2-dependent parts
with the penalty chosen on VALIDATION prompts, separately for every (training-set size, ratings-per-item) cell, so every point of every curve is a tuned head.

Same data, same split seed (0), same subsample seeds, same head (CE-trained, 24-component linear softmax) as gd_study_and_allocation.py.  What is new:
    penalty grid        LAMBDA_GRID (6 values, 1e-5 to 1, softmax penalty 0.5 * lam * sum(W^2) on the mean loss, e40_l2.py)
    selection           per cell, lam = argmin over the grid of the MEAN over the subsample seeds of VALIDATION normalized RPS; test prompts never choose lam
    reported            test normalized RPS (RPS / (K - 1), 0 to 1, lower is better) of the head at the chosen lam, averaged over the subsample seeds; the test curve at every
                        fixed lam is stored too, so a reader can see how much of the shape is the penalty.
Parts that do not depend on the penalty (G-study variance components of the ratings, the ceiling on real ratings) are copied from gd_study_and_allocation.json.
Writes ../results/gd_study_l2_validation_chosen.json (after --merge of the shards).

Usage:  OMP_NUM_THREADS=1 python3 -I gd_study_l2_validation_chosen.py --only summeval_crowd dices990 chaosnli_alpha popquorn_offensiveness --tag A ;  ... --only summeval_expert popquorn_politeness chaosnli_snli --tag B ;
        python3 -I gd_study_l2_validation_chosen.py --merge
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python3 -I does not add the script directory

import numpy as np

from e40_common import EXP40, d, fe, jsonable, paired_ci, rps_one
from e40_l2 import LAMBDA_GRID, l2_settings

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--maxiter", type=int, default=300)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--of", type=int, default=1)
ap.add_argument("--only", nargs="*", default=None)
ap.add_argument("--tag", default=None, help="output file tag (use with --only to balance shards by hand)")
ap.add_argument("--merge", action="store_true")
args = ap.parse_args()
fe.MAXITER = args.maxiter
OUT = EXP40 / "results"
OUT.mkdir(exist_ok=True)
Z95, Z80 = 1.959963984540054, 0.8416212335729143


def mean_ci(a, boot=1000, seed=1):
    """Mean of a per-test-prompt vector with a 95% percentile interval from resampling test PROMPTS (the clustering unit)."""
    a = np.asarray(a, float)
    n = a.shape[0]
    idx = np.random.default_rng(seed).integers(0, n, size=(boot, n))
    m = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def one_way_components(groups):
    """Copied unchanged from gd_study_and_allocation.py: one-way random-effects variance components, unbalanced ANOVA estimator."""
    sizes = np.array([g.size for g in groups], float)
    n_items = len(groups)
    big_n = sizes.sum()
    means = np.array([g.mean() for g in groups])
    grand = np.concatenate(groups).mean()
    ssb = float((sizes * (means - grand) ** 2).sum())
    ssw = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    msb = ssb / (n_items - 1)
    msw = ssw / (big_n - n_items)
    n0 = (big_n - (sizes ** 2).sum() / big_n) / (n_items - 1)
    sc2 = max(0.0, (msb - msw) / n0)
    sw2 = msw
    icc = sc2 / (sc2 + sw2) if (sc2 + sw2) > 0 else float("nan")
    return {"sigma_c2": sc2, "sigma_w2": sw2, "icc1": icc, "n0": float(n0), "n_items": n_items, "n_ratings": int(big_n)}


def eval_grid(data, idx, k_keep, seed, val, test, keep_probs=False):
    """Fit the CE head on training prompts `idx` (k_keep ratings per prompt, or all) at every lam; score validation and test."""
    values = np.asarray(data.class_values, float)
    kk = data.n_classes
    x_tr, (x_va, x_te), _rank = fe.design_matrices(data.texts, idx, [val, test])
    out = {}
    for lam in LAMBDA_GRID:
        with l2_settings(lam):
            pack = fe.fit_objective(x_tr, data.ratings, idx, values, "ce", k_keep, seed)
        pv, mv = fe.predict_distributions(pack, x_va, "ce")
        pt, mt = fe.predict_distributions(pack, x_te, "ce")
        rv = fe._item_records(pv, mv, data.ratings, val, values)["rps"] / (kk - 1)
        rt = fe._item_records(pt, mt, data.ratings, test, values)
        out[lam] = {"val": float(rv.mean()), "test_vec": rt["rps"] / (kk - 1), "nit": pack.get("nit"), "probs": pt if keep_probs else None}
    return out


def choose(per_seed):
    """per_seed: list over subsample seeds of eval_grid outputs.  Returns (lam*, mean test vector at lam*, per-lam mean validation, per-lam mean test)."""
    val_mean = {lam: float(np.mean([s[lam]["val"] for s in per_seed])) for lam in LAMBDA_GRID}
    lam_star = min(LAMBDA_GRID, key=lambda lam: val_mean[lam])
    test_mean = {lam: float(np.mean([s[lam]["test_vec"].mean() for s in per_seed])) for lam in LAMBDA_GRID}
    vec = np.mean([s[lam_star]["test_vec"] for s in per_seed], axis=0)
    return lam_star, vec, val_mean, test_mean


def run_dataset(data, seeds, boot):
    t0 = time.time()
    kk = data.n_classes
    values = np.asarray(data.class_values, float)
    train, val, test = d.split_by_group(data.groups, seed=0)
    d.assert_no_prompt_leakage(data.groups, data.rater_ids, train, val, test)
    n_train = int(train.size)
    out = {"name": data.name, "n_items": data.n_items, "K": kk, "n_train": n_train, "n_val": int(val.size), "n_test": int(test.size),
           "lambda_grid": LAMBDA_GRID, "selection": "lam chosen per cell on validation normalized RPS (mean over subsample seeds)"}
    cap = int(np.percentile([r.size for r in data.ratings], 20))
    cprob, cmu = fe.constant_marginal(data.ratings, train, kk, values)
    crec = fe._item_records(np.repeat(cprob.reshape(1, -1), len(test), 0), np.full(len(test), cmu), data.ratings, test, values)
    out["constant_test_nrps"] = float(crec["rps"].mean() / (kk - 1))

    # ---- learning curve, k = 1 (one rating per training item) ----
    ns = [n for n in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096) if n <= n_train]
    if n_train not in ns:
        ns.append(n_train)
    curve, chosen, by_lam, val_by_lam, seed_means = {}, {}, {}, {}, {}
    for n in ns:
        per_seed = []
        for s in range(seeds):
            order = np.random.default_rng(100 + s).permutation(train)
            per_seed.append(eval_grid(data, np.sort(order[:n]), 1, 7 + s, val, test))
        lam_star, vec, vmean, tmean = choose(per_seed)
        curve[n], chosen[n], by_lam[n], val_by_lam[n] = vec, lam_star, tmean, vmean
        seed_means[n] = [float(sd[lam_star]["test_vec"].mean()) for sd in per_seed]
    out["learning_curve_k1"] = {str(n): float(v.mean()) for n, v in curve.items()}
    out["learning_curve_k1_chosen_lambda"] = {str(n): chosen[n] for n in ns}
    out["learning_curve_k1_test_by_lambda"] = {str(n): {str(l): v for l, v in by_lam[n].items()} for n in ns}
    out["learning_curve_k1_validation_by_lambda"] = {str(n): {str(l): v for l, v in val_by_lam[n].items()} for n in ns}
    const_vec = crec["rps"] / (kk - 1)
    out["learning_curve_k1_seed_means"] = {str(n): seed_means[n] for n in ns}   # one test mean per training-subsample seed, at the chosen penalty
    out["learning_curve_k1_mean_ci95"] = {str(n): dict(zip(("mean", "lo", "hi"), mean_ci(curve[n], boot=boot))) for n in ns}
    out["learning_curve_k1_gain_over_constant_paired"] = {
        str(n): dict(zip(("mean", "lo", "hi"), paired_ci(const_vec, curve[n], boot=boot))) for n in ns}
    out["learning_curve_interval_definition"] = (
        "95% percentile interval from 1000 resamples of the TEST PROMPTS (item-clustered; one vector per prompt, averaged over the subsample seeds). "
        "It reflects test-prompt sampling only; the spread over training-subsample seeds is in learning_curve_k1_seed_means. gain_over_constant_paired = test normalized RPS of "
        "the constant forecast minus that of the CE-trained head at its validation-chosen penalty; positive means the head is better.")
    if len(ns) >= 3:
        last, prev = curve[ns[-1]].mean(), curve[ns[-2]].mean()
        out["learning_curve_last_step_relative_gain"] = float((prev - last) / prev)
        m, lo, hi = paired_ci(curve[ns[-2]], curve[ns[-1]], boot=boot)
        out["learning_curve_last_step_paired"] = {"n_from": ns[-2], "n_to": ns[-1], "mean_drop": m, "lo": lo, "hi": hi}

    # ---- iso-budget allocation at a validation-chosen penalty per cell: B = N x k ----
    gd_old = json.loads((OUT / "gd_study_and_allocation.json").read_text())
    old = next(x for x in gd_old["datasets"] if x["name"] == data.name)
    out["g_study"] = old["g_study"]                      # independent of the penalty; copied
    out["ceiling_on_real_ratings"] = old["ceiling_on_real_ratings"]
    kstar = out["g_study"]["k_star_080"]
    k_used = None if kstar is None else min(kstar, cap)
    ks = sorted(set([1, 2, 4, 8, 16] + ([k_used] if k_used else [])))
    ks = [k for k in ks if k <= cap]
    alloc, dropped = {}, {}
    for b in (128, 256, 512):
        if b > n_train:
            dropped[str(b)] = f"budget {b} exceeds the {n_train} training items, so a one-rating-per-item design cannot spend it"
            continue
        plans = [(b // k, k) for k in ks if b // k >= 16]
        if len(plans) < 2 or 1 not in [k for _, k in plans]:
            dropped[str(b)] = "fewer than two plans with at least 16 items, or no k = 1 plan"
            continue
        per_k, lam_k = {}, {}
        for n, k in plans:
            per_seed = []
            for s in range(seeds):
                order = np.random.default_rng(200 + s).permutation(train)
                per_seed.append(eval_grid(data, np.sort(order[:n]), k, 17 + s, val, test))
            lam_star, vec, _vmean, _tmean = choose(per_seed)
            per_k[k], lam_k[k] = vec, lam_star
        base = per_k[1]
        entry = {"difference_definition": ("test normalized RPS of the k-ratings-per-item head minus test normalized RPS of the one-rating-per-item head, same total budget B, "
                                          "each head at its validation-chosen penalty; positive means the one-rating-per-item head is better"),
                 "baseline_k": 1, "plans": [{"N": n, "k": k, "B_spent": n * k, "test_nrps": float(per_k[k].mean()), "chosen_lambda": lam_k[k],
                                             "k_minus_k1_paired": dict(zip(("mean", "lo", "hi"), paired_ci(per_k[k], base, boot=boot)))} for n, k in plans]}
        entry["best_k"] = int(min(per_k, key=lambda kk_: per_k[kk_].mean()))
        entry["k_star_080"], entry["k_star_used"] = kstar, k_used
        entry["k_star_truncated_by_available_ratings"] = bool(kstar is not None and kstar > cap)
        alloc[str(b)] = entry
    out["iso_budget"], out["iso_budget_dropped"] = alloc, dropped

    # ---- full training set: validation-chosen penalty, then test-loss components and the paired comparison with the constant forecast ----
    full = eval_grid(data, train, None, 5, val, test, keep_probs=True)
    lam_full = min(LAMBDA_GRID, key=lambda lam: full[lam]["val"])
    prob = full[lam_full]["probs"]
    per_rating = []
    for row, i in enumerate(test):
        q = prob[row]
        per_rating.append(np.asarray([rps_one(q, int(yy)) / (kk - 1) for yy in np.asarray(data.ratings[i], int)]))
    lc = one_way_components([g for g in per_rating if g.size >= 2])
    sc2, sw2 = lc["sigma_c2"], lc["sigma_w2"]
    out["full_train_chosen_lambda"] = lam_full
    out["test_loss_components_ce_head"] = {"sigma_c2": sc2, "sigma_w2": sw2, "n_items": lc["n_items"], "n_ratings": lc["n_ratings"],
                                           "share_of_variance_between_items": sc2 / (sc2 + sw2),
                                           "risk_sd_N100_k1": math.sqrt((sc2 + sw2) / 100),
                                           "k_opt_if_item_costs_10x_rating": max(1.0, math.sqrt(10 * sw2 / sc2)) if sc2 > 0 else None}
    ce_items = np.array([g.mean() for g in per_rating])
    const_items = crec["rps"] / (kk - 1)
    diff = const_items - ce_items
    sd_d = float(diff.std(ddof=1))
    gm, glo, ghi = paired_ci(const_items, ce_items, boot=boot)
    out["paired_vs_constant"] = {
        "definition": ("test normalized RPS of the constant forecast minus test normalized RPS of the CE-trained head at its validation-chosen penalty "
                       "(full training set, split seed 0); positive means the CE-trained head is better"),
        "mean_gain": gm, "lo": glo, "hi": ghi, "ce_beats_constant_ci_excludes_0": bool(glo > 0), "sd_per_item_diff": sd_d,
        "n_test_items_for_80pct_power_at_observed_gain": (float(((Z95 + Z80) * sd_d / diff.mean()) ** 2) if diff.mean() > 0 else None),
        "n_test_items_for_80pct_power_at_0.005": float(((Z95 + Z80) * sd_d / 0.005) ** 2),
        "n_test_items_for_80pct_power_at_0.0016": float(((Z95 + Z80) * sd_d / 0.0016) ** 2)}
    out["elapsed_seconds"] = time.time() - t0
    print(f"[{data.name}] done in {out['elapsed_seconds']:.0f}s  curve {dict((n, round(float(v.mean()), 4)) for n, v in curve.items())}  lam* {chosen}", flush=True)
    return out


def main():
    if args.merge:
        shards = sorted(OUT.glob("gd_study_l2_validation_chosen_*.json"))
        assert shards, "no shard files"
        datasets = []
        for p in shards:
            datasets.extend(json.loads(p.read_text())["datasets"])
        order = ["popquorn_offensiveness", "popquorn_politeness", "summeval_expert", "summeval_crowd", "dices990", "chaosnli_snli_mnli", "chaosnli_alpha"]
        datasets.sort(key=lambda x: order.index(x["name"]) if x["name"] in order else 99)
        payload = {"preregistered": False, "seeds": args.seeds, "boot": args.boot, "lambda_grid": LAMBDA_GRID, "datasets": datasets,
                   "unavailable": {"veribench_fsc_rv1": "pinned file is in a repository this machine cannot read (HTTP 404)"},
                   "source_shards": [p.name for p in shards]}
        (OUT / "gd_study_l2_validation_chosen.json").write_text(json.dumps(jsonable(payload), indent=1))
        print("merged", [x["name"] for x in datasets])
        return
    sets = [d.load_popquorn("offensiveness"), d.load_popquorn("politeness"), d.load_summeval("expert"), d.load_summeval("crowd"), d._load_dices("990"),
            d.load_chaos_3way(), d.load_chaos_alpha()]
    if args.only:
        sets = [s for s in sets if any(o in s.name for o in args.only)]
    sets = [s for i, s in enumerate(sets) if i % args.of == args.shard]
    results = [run_dataset(s, args.seeds, args.boot) for s in sets]
    payload = {"preregistered": False, "seeds": args.seeds, "boot": args.boot, "lambda_grid": LAMBDA_GRID, "datasets": results}
    out = OUT / (f"gd_study_l2_validation_chosen_{args.tag}.json" if args.tag else f"gd_study_l2_validation_chosen_shard{args.shard}of{args.of}.json")
    out.write_text(json.dumps(jsonable(payload), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
