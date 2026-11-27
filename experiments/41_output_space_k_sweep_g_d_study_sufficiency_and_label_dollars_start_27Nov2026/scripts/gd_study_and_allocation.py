#!/usr/bin/env python3
"""Real-data G-study / D-study, the reliability ceiling on real ratings, budget allocation and learning curves (Experiment 41).

Question 1 (G-theory).  What do the one-way variance components of the real ratings say (ICC(1), the D-study number of
ratings k* for reliability 0.80), and how do they compare with what a proper-loss judge actually needs?
Question 2 (ceiling).   On real ratings, does a deterministic item-mean judge beat an honest sampled judge on correlation
(the sqrt(rho) versus rho fact), and what does rounding the mean to the rating lattice do?
Question 3 (allocation).At a fixed total number of ratings B = N x k, which k trains the best cross-entropy head?
                        Is the D-study k* better than k = 1?
Question 4 (sufficiency).How does held-out normalized RPS fall with the number of training items, and how many TEST items
                        does a paired comparison need?  (Dollars are computed in sufficiency_and_dollars.py.)

Predictions P6 and P7 in ../PREREGISTERED_PREDICTIONS_11-27-2026.md were fixed before this script produced a number.
Reuses Experiment 36's loaders and heads unchanged.  Writes ../results/gd_study_and_allocation.json.
Usage: python3 -I gd_study_and_allocation.py [--seeds 3] [--boot 1000]
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

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--only", nargs="*", default=None)
args = ap.parse_args()

OUT = EXP40 / "results"
OUT.mkdir(exist_ok=True)
Z95, Z80 = 1.959963984540054, 0.8416212335729143


# ----------------------------------------------------------------------------------------------------------------------
# One-way random-effects variance components (items random, ratings nested in items), unbalanced ANOVA estimator.
# ----------------------------------------------------------------------------------------------------------------------
def one_way_components(groups):
    """groups: list of 1-D arrays (the ratings, or the per-rating losses, of each item).  Returns dict."""
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


def d_study_k(sc2, sw2, target=0.80):
    if sc2 <= 0:
        return None
    return int(math.ceil(target / (1 - target) * sw2 / sc2))


def g_study(data, boot, seed=11):
    ratings = [np.asarray(data.class_values)[np.asarray(r)] for r in data.ratings]
    comp = one_way_components(ratings)
    rng = np.random.default_rng(seed)
    iccs, kstars = [], []
    n = len(ratings)
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        c = one_way_components([ratings[i] for i in idx])
        iccs.append(c["icc1"])
        kstars.append(d_study_k(c["sigma_c2"], c["sigma_w2"]))
    comp["icc1_ci95"] = [float(np.nanpercentile(iccs, 2.5)), float(np.nanpercentile(iccs, 97.5))]
    ks = [k for k in kstars if k is not None]
    comp["k_star_080"] = d_study_k(comp["sigma_c2"], comp["sigma_w2"])
    comp["k_star_080_ci95"] = [int(np.percentile(ks, 2.5)), int(np.percentile(ks, 97.5))] if ks else None
    comp["ratings_per_item_median"] = float(np.median([r.size for r in ratings]))
    comp["ratings_per_item_p20"] = float(np.percentile([r.size for r in ratings], 20))
    means = np.array([r.mean() for r in ratings])
    allr = np.concatenate(ratings)
    comp["naive_rho_var_item_means_over_var_total"] = float(means.var() / allr.var())  # biased upward by sigma_w^2/n_i
    return comp


def ceiling_on_real(data):
    """Leave-one-rating-out judges, pooled over (item, rating): collapsed mean of the other ratings, a sampled draw from them
    (exact expectation, no sampling noise) and the lattice-rounded mean.  Items with at least 3 ratings."""
    values = np.asarray(data.class_values, float)
    ys, om, o2 = [], [], []
    for r in data.ratings:
        v = values[np.asarray(r)]
        n = v.size
        if n < 3:
            continue
        s, q = v.sum(), (v ** 2).sum()
        for j in range(n):
            ys.append(v[j])
            om.append((s - v[j]) / (n - 1))
            o2.append((q - v[j] ** 2) / (n - 1))
    ys, om, o2 = map(np.asarray, (ys, om, o2))
    corr_mean = float(np.corrcoef(om, ys)[0, 1])
    rounded = np.clip(np.floor(om + 0.5), values.min(), values.max())
    corr_round = float(np.corrcoef(rounded, ys)[0, 1]) if rounded.std() > 0 else float("nan")
    ex_y, var_y = ys.mean(), ys.var()
    cov_s = float((ys * om).mean() - om.mean() * ex_y)
    var_x = float(o2.mean() - om.mean() ** 2)
    corr_sampled = cov_s / math.sqrt(var_x * var_y)
    return {"n_pairs": int(ys.size), "corr_collapsed_mean_of_others": corr_mean, "corr_sampled_draw_of_others": corr_sampled,
            "corr_lattice_rounded_mean_of_others": corr_round}


# ----------------------------------------------------------------------------------------------------------------------
# CE-trained head on a chosen training subset: N items, k ratings per item.
# ----------------------------------------------------------------------------------------------------------------------
def fit_eval_subset(data, prep, n_items, k, seed, order):
    idx = np.sort(order[:n_items])
    x_tr, (x_te,), _rank = fe.design_matrices(data.texts, idx, [prep["test"]])
    pack = fe.fit_objective(x_tr, data.ratings, idx, np.asarray(data.class_values, float), "ce", k, seed)
    prob, mu = fe.predict_distributions(pack, x_te, "ce")
    rec = fe._item_records(prob, mu, data.ratings, prep["test"], np.asarray(data.class_values, float))
    kk = data.n_classes
    return {"nrps": rec["rps"] / (kk - 1), "ce": rec["ce"]}


def run_dataset(data, seeds, boot):
    t0 = time.time()
    kk = data.n_classes
    values = np.asarray(data.class_values, float)
    train, val, test = d.split_by_group(data.groups, seed=0)
    d.assert_no_prompt_leakage(data.groups, data.rater_ids, train, val, test)
    prep = {"train": train, "val": val, "test": test}
    n_train = int(train.size)
    out = {"name": data.name, "n_items": data.n_items, "K": kk, "n_train": n_train, "n_test": int(test.size)}
    out["g_study"] = g_study(data, boot)
    out["ceiling_on_real_ratings"] = ceiling_on_real(data)
    cap = int(np.percentile([r.size for r in data.ratings], 20))
    kstar = out["g_study"]["k_star_080"]

    # constant reference on the test items
    cprob, cmu = fe.constant_marginal(data.ratings, train, kk, values)
    crec = fe._item_records(np.repeat(cprob.reshape(1, -1), len(test), 0), np.full(len(test), cmu), data.ratings, test, values)
    out["constant_test_nrps"] = float(crec["rps"].mean() / (kk - 1))

    # ---- learning curve (k = 1): training items N, averaged over subsample seeds ----
    ns = [n for n in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096) if n <= n_train]
    if n_train not in ns:
        ns.append(n_train)
    curve = {}
    for n in ns:
        vecs = []
        for s in range(seeds):
            order = np.random.default_rng(100 + s).permutation(train)
            vecs.append(fit_eval_subset(data, prep, n, 1, 7 + s, order)["nrps"])
        curve[n] = np.mean(vecs, axis=0)
    out["learning_curve_k1"] = {str(n): float(v.mean()) for n, v in curve.items()}
    if len(ns) >= 3:
        last, prev = curve[ns[-1]].mean(), curve[ns[-2]].mean()
        out["learning_curve_last_step_relative_gain"] = float((prev - last) / prev)
        m, lo, hi = paired_ci(curve[ns[-2]], curve[ns[-1]], boot=boot)
        out["learning_curve_last_step_paired"] = {"n_from": ns[-2], "n_to": ns[-1], "mean_drop": m, "lo": lo, "hi": hi}

    # ---- iso-budget allocation: total ratings B = N x k (N = floor(B / k) items, k ratings each) ----
    # Budgets are only those a k = 1 design can actually spend (B <= number of training items), so the k = 1 baseline always exists.
    # k* is the D-study recommendation; when fewer than k* ratings exist for the item at the 20th percentile, k is capped and flagged.
    k_used = None if kstar is None else min(kstar, cap)
    ks = sorted(set([1, 2, 4, 8, 16] + ([k_used] if k_used else [])))
    ks = [k for k in ks if k <= cap]
    alloc = {}
    for b in [b_ for b_ in (128, 256, 512) if b_ <= n_train]:
        plans = [(b // k, k) for k in ks if b // k >= 16]
        if len(plans) < 2 or 1 not in [k for _, k in plans]:
            continue
        per_k = {}
        for n, k in plans:
            vecs = []
            for s in range(seeds):
                order = np.random.default_rng(200 + s).permutation(train)
                vecs.append(fit_eval_subset(data, prep, n, k, 17 + s, order)["nrps"])
            per_k[k] = np.mean(vecs, axis=0)
        base = per_k[1]
        key = ("test normalized RPS of the k-ratings-per-item head minus test normalized RPS of the one-rating-per-item head, "
               "same total budget; positive means the one-rating-per-item head is better")
        entry = {"difference_definition": key, "baseline_k": 1,
                 "plans": [{"N": n, "k": k, "B_spent": n * k, "test_nrps": float(per_k[k].mean()),
                            "k_minus_k1_paired": dict(zip(("mean", "lo", "hi"), paired_ci(per_k[k], base, boot=boot)))}
                           for n, k in plans]}
        entry["best_k"] = int(min(per_k, key=lambda kk_: per_k[kk_].mean()))
        entry["k_star_080"] = kstar
        entry["k_star_used"] = k_used
        entry["k_star_truncated_by_available_ratings"] = bool(kstar is not None and kstar > cap)
        entry["k_star_plan_present"] = bool(k_used in per_k) if k_used else None
        alloc[str(b)] = entry
    out["iso_budget"] = alloc

    # ---- G-theory applied to the TEST LOSS: variance components of per-rating normalized RPS, and the paired-power size ----
    order = np.random.default_rng(300).permutation(train)
    x_tr, (x_te,), _ = fe.design_matrices(data.texts, train, [test])
    pack = fe.fit_objective(x_tr, data.ratings, train, values, "ce", None, 5)
    prob, mu = fe.predict_distributions(pack, x_te, "ce")
    per_rating_losses = []
    for row, i in enumerate(test):
        q = prob[row]
        per_rating_losses.append(np.asarray([rps_one(q, int(yy)) / (kk - 1) for yy in np.asarray(data.ratings[i], int)]))
    lc = one_way_components([g for g in per_rating_losses if g.size >= 2])
    out["test_loss_components_ce_head"] = {k_: lc[k_] for k_ in ("sigma_c2", "sigma_w2", "n_items", "n_ratings")}
    sc2, sw2 = lc["sigma_c2"], lc["sigma_w2"]
    out["test_loss_components_ce_head"]["share_of_variance_between_items"] = sc2 / (sc2 + sw2)
    # risk precision with N items and k ratings per item: sd = sqrt((sc2 + sw2/k)/N)
    out["test_loss_components_ce_head"]["risk_sd_N100_k1"] = math.sqrt((sc2 + sw2) / 100)
    out["test_loss_components_ce_head"]["k_opt_if_item_costs_10x_rating"] = max(1.0, math.sqrt(10 * sw2 / sc2)) if sc2 > 0 else None

    # paired power: CE head versus constant, and the per-item SD of paired differences
    ce_items = np.array([g.mean() for g in per_rating_losses])
    const_items = crec["rps"] / (kk - 1)
    diff = const_items - ce_items
    sd_d = float(diff.std(ddof=1))
    gm, glo, ghi = paired_ci(const_items, ce_items, boot=boot)
    out["paired_vs_constant"] = {
        "definition": "test normalized RPS of the constant forecast minus test normalized RPS of the CE-trained head (full training set, split seed 0); positive means the CE-trained head is better",
        "mean_gain": gm, "lo": glo, "hi": ghi, "ce_beats_constant_ci_excludes_0": bool(glo > 0),
        "sd_per_item_diff": sd_d,
        "n_test_items_for_80pct_power_at_observed_gain": (float(((Z95 + Z80) * sd_d / diff.mean()) ** 2) if diff.mean() > 0 else None),
        "n_test_items_for_80pct_power_at_0.005": float(((Z95 + Z80) * sd_d / 0.005) ** 2),
        "n_test_items_for_80pct_power_at_0.0016": float(((Z95 + Z80) * sd_d / 0.0016) ** 2)}
    out["elapsed_seconds"] = time.time() - t0
    print(f"[{data.name}] done in {out['elapsed_seconds']:.0f}s  ICC1 {out['g_study']['icc1']:.3f}  k* {kstar}  cap {cap}", flush=True)
    return out


def main():
    pop_off = d.load_popquorn("offensiveness")
    pop_pol = d.load_popquorn("politeness")
    se_exp = d.load_summeval("expert")
    se_crd = d.load_summeval("crowd")
    d990 = d._load_dices("990")
    chaos3 = d.load_chaos_3way()
    chaosa = d.load_chaos_alpha()
    sets = [pop_off, pop_pol, se_exp, se_crd, d990, chaos3, chaosa]
    if args.only:
        sets = [s for s in sets if any(o in s.name for o in args.only)]
    results = [run_dataset(s, args.seeds, args.boot) for s in sets]
    payload = {"seeds": args.seeds, "boot": args.boot, "datasets": results,
               "unavailable": {"veribench_fsc_rv1": "pinned file is in a repository this machine cannot read (HTTP 404)"}}
    out = OUT / "gd_study_and_allocation.json"
    out.write_text(json.dumps(jsonable(payload), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
