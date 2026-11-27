#!/usr/bin/env python3
"""How much human-rating data is enough, and what does it cost against researcher time?  (Experiment 41)

    python3 -I sufficiency_and_dollars.py [--gd results/gd_study_l2_validation_chosen.json --tag _l2_validation_chosen --penalty-label "..."]

Inputs
  --gd PATH                                   real-data G-study, D-study, learning curves, equal-budget allocation.  Default
                                              results/gd_study_and_allocation.json (fixed L2 penalty 0.01, the preregistered setting; known to over-shrink the CE-trained head);
                                              results/gd_study_l2_validation_chosen.json (penalty chosen per point on the validation split; NOT preregistered) supersedes it.
  results/sister_exp15_aggregates_pinned.json aggregate numbers copied from the sister repository's costed-allocation planning file
                                              (built once with --pin-from PATH; contains no rater names)
Outputs (suffix --tag)
  results/sufficiency_and_dollars{tag}.json, results/sufficiency_and_dollars_tables{tag}.md,
  figures/g_d_study_reliability_curves{tag}.png, figures/learning_curves_and_saturation{tag}.png, figures/label_dollars_vs_researcher_month{tag}.png

What is measured, derived and assumed (read this before quoting any dollar figure)
  MEASURED here:   variance components, intraclass correlation ICC(1), D-study k*, held-out normalized ranked probability score (RPS) learning curves, paired standard deviations.
  DERIVED:         about $0.05 per rating for the crowd.  POPQUORN reports $12 per hour (Prolific) and a median of 13 minutes for 50 offensiveness annotations,
                   which is $0.052 per annotation.  It is not a published per-rating price, and it is not separately measured for politeness or for any other task.
  ASSUMED:         $5 per rating (expert rate), $25/$50/$100 per hour for rater time, $15k/$30k/$75k per researcher-month, 160 hours per month.
                   Gemini's $15k to $75k researcher-month range was NOT verified against any source; it is only a grid of what-ifs.
  MEASURED (sister repo, aggregate): median 4.0 minutes per rating over 103 entries by 2 raters on formalization-quality ratings, and an
                   ASSUMED length-scaled 37.3 minutes because the main files are 9.3 times longer than the warm-up files.

Sufficiency definitions.  These were fixed in this script before it ran; they are NOT in the preregistration, and the script was not committed beforehand.
  train side: N_sat = the smallest number of training items N on the learning-curve grid such that no larger N improves held-out normalized RPS by more than 0.001
              (a descriptive summary of one noisy test curve, mean of 3 subsample draws on one prompt-level split; a margin this small is near the noise).
              Sensitivity: the margin is 10% of the gain of the full-training head over the constant forecast.  That gain is read at the LARGEST N on the curve,
              not at the minimum of the curve, so it is not selected on the test data.
  test side:  N_test(delta) = ((z_0.975 + z_0.80) * sd / delta)^2 items for a paired comparison of two forecasts at 80% power, two-sided 5%,
              sd = the per-test-item standard deviation of the paired difference, measured here for the CE-trained head versus the constant forecast.
  allocation: for a budget of B ratings split into N items with k ratings each, the variance of an item-clustered mean loss is (sc2 + sw2 / k) / N,
              so with c_item = cost of one extra item (beyond its ratings) and c_rating = cost of one rating, the cost-minimizing k is
              k_opt = max(1, sqrt((c_item / c_rating) * sw2 / sc2)).  c_item = 0 (a free pool of prompts) gives k_opt = 1.
              This is a statement about the PRECISION OF A HELD-OUT RISK ESTIMATE (a test-side quantity), not about how many ratings per item a judge should be trained on.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parent.parent
RES, FIG = EXP / "results", EXP / "figures"
FIG.mkdir(exist_ok=True)
Z = 1.959963984540054 + 0.8416212335729143
PRICE_CROWD, PRICE_EXPERT = 0.05, 5.00
WAGES = [25, 50, 100]
RESEARCHER_MONTH = [15_000, 30_000, 75_000]
HOURS_PER_MONTH = 160
DELTAS = [0.005, 0.0016]
SAT_MARGIN = 0.001
BLUE, ORANGE, AQUA, GRAY, TEXT1, TEXT2, GRID = "#257de0", "#f47230", "#00b894", "#9c9c9c", "#191919", "#666666", "#e3e3e3"
ORDER = ["popquorn_offensiveness", "popquorn_politeness", "summeval_expert", "summeval_crowd", "dices990", "chaosnli_snli_mnli", "chaosnli_alpha"]


def esc(t):
    """matplotlib reads a pair of dollar signs as math text; escape every dollar sign in figure text."""
    return t.replace("$", r"\$")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return jsonable(o.item())
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def pin_sister(path: Path) -> dict:
    """Copy only the aggregate numbers we use; no file names with people in them, no per-person lines."""
    d = json.loads(path.read_text())
    sc = d["scenarios"]
    keep = {k: {"new_development_tasks": sc[k]["new_development_tasks"], "evaluation_tasks": sc[k]["evaluation_tasks"],
                "total_independent_ratings": sc[k]["total_independent_ratings"], "hours": sc[k]["hours"]} for k in sc}
    pins = {"source": "trustworthy-theorem-judges, experiment 15 (estimate required tasks, solver configurations and human raters), results/costed_allocation.json, "
                      "read from the efdc8c3 checkout; planning arithmetic, not a measurement of any future study",
            "measured_minutes_per_rating": {"current_main": d["review_time"]["entered_minutes"]["current_main"],
                                            "warm_up": d["review_time"]["entered_minutes"]["warm_up"]},
            "assumed_minutes_if_time_scaled_with_length": d["review_time"]["assumed_minutes_if_time_scaled_with_length"],
            "main_to_warm_up_median_length_ratio": d["review_time"]["main_to_warm_up_median_length_ratio"],
            "paired_rps_contrast_sd_25_tasks": d["paired_rps_contrast_sd"]["averaged_forecasts_25_tasks"]["implied_sd"],
            "units_for_half_width_25_task_contrast": d["paired_rps_contrast_sd"]["averaged_forecasts_25_tasks"]["units_for_half_width"],
            "scenarios": keep}
    return pins


def components(ds):
    t = ds["test_loss_components_ce_head"]
    return t["sigma_c2"], t["sigma_w2"]


SAT_REL = 0.10  # NOT preregistered: margin = 10% of the gain of the full-training head over the constant forecast (read at the largest N, not at the test minimum)


def saturation(curve: dict, margin: float = SAT_MARGIN):
    ns = sorted(int(k) for k in curve)
    vals = {n: curve[str(n)] for n in ns}
    for i, n in enumerate(ns):
        best_later = min(vals[m] for m in ns[i + 1:]) if i + 1 < len(ns) else vals[n]
        if vals[n] - best_later <= margin:
            return n
    return ns[-1]


def smallest_n_from_which_gain_is_significant(gain_by_n):
    """gain_by_n: {n: {"mean","lo","hi"}}, gain = constant minus head (positive = head better).  Returns the smallest N on the grid such that the paired 95% interval's
    lower bound is above 0 at N and at every larger N on the grid, or None."""
    ns = sorted(int(k) for k in gain_by_n)
    for i, n in enumerate(ns):
        if all(gain_by_n[str(m)]["lo"] > 0 for m in ns[i:]):
            return n
    return None


# Names differ between the L2-sensitivity K-sweep views and the G/D datasets for one set.
BASE_TO_GD = {"chaosnli_alphanli": "chaosnli_alpha"}


def residual_power(sens_path: Path, gd_by_name: dict) -> dict | None:
    """Items needed to resolve the small gap that is LEFT between the Pearson-trained and the CE-trained head after each head gets its own validation-fitted calibration map.

    Source: results/k_sweep_l2_sensitivity_summary.json (NOT preregistered; penalties chosen on validation).  For each of the 16 K-sweep views the paired item-clustered 95% interval of
    (Pearson-trained head minus CE-trained head, both calibrated) was computed per split seed; the mean over the 5 seeds of its lower and upper ends gives a mean half-width hw.
    The implied standard deviation of the per-test-item paired difference is sd = hw * sqrt(n_test) / 1.96 (normal approximation; an approximation, not measured directly).
    Minimum detectable difference at the available test size (80% power, two-sided 5%) = (1.96 + 0.84) * sd / sqrt(n_test); items needed for a target difference d = ((1.96 + 0.84) * sd / d)^2.
    """
    if not sens_path.exists():
        return None
    sens = json.loads(sens_path.read_text())["summary"]
    rows = {}
    for view, v in sens.items():
        m = v["matched_pearson_minus_ce"]
        n = v["n_test"]
        hw = (m["hi_seed_mean"] - m["lo_seed_mean"]) / 2.0
        sd = hw * math.sqrt(n) / 1.959963984540054
        gd_name = BASE_TO_GD.get(v["base"], v["base"])
        k_med = gd_by_name[gd_name]["g_study"]["ratings_per_item_median"] if gd_name in gd_by_name else None
        row = {"K": v["K"], "n_test_items": n, "n_seeds": v["n_seeds"], "raw_gap_seed_mean": v["raw_pearson_minus_ce"]["seed_mean"],
               "calibrated_gap_seed_mean": m["seed_mean"], "calibrated_gap_seeds_ci_above_0": m["n_ci_above_0"], "calibrated_gap_seeds_ci_below_0": m["n_ci_below_0"],
               "mean_half_width_95": hw, "implied_sd_per_item": sd, "ratings_per_item_median": k_med,
               "min_detectable_difference_at_available_items": Z * sd / math.sqrt(n)}
        for tag, target in (("0.0022", 0.0022), ("0.001", 0.001)):
            items = (Z * sd / target) ** 2
            row[f"items_to_resolve_{tag}"] = items
            row[f"crowd_usd_at_median_k_{tag}"] = None if k_med is None else items * k_med * PRICE_CROWD
        rows[view] = row
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin-from", default=None, help="path to the sister repo's costed_allocation.json (aggregates only are copied)")
    ap.add_argument("--gd", default=str(RES / "gd_study_and_allocation.json"), help="real-data G/D result file (see the header)")
    ap.add_argument("--tag", default="", help="suffix for every output file name, e.g. _l2_validation_chosen")
    ap.add_argument("--penalty-label", default="L2 penalty fixed at 0.01 for every point (the preregistered setting)",
                    help="one line saying how the ridge penalty of the judge was chosen; printed on every figure and table")
    ap.add_argument("--sens", default=str(RES / "k_sweep_l2_sensitivity_summary.json"), help="L2-sensitivity K-sweep summary (for the residual-gap power table)")
    args = ap.parse_args()
    tag = args.tag
    pin_path = RES / "sister_exp15_aggregates_pinned.json"
    if args.pin_from:
        pin_path.write_text(json.dumps(jsonable(pin_sister(Path(args.pin_from))), indent=1))
    pins = json.loads(pin_path.read_text())
    gd_path = Path(args.gd)
    gd = json.loads(gd_path.read_text())
    gd_by_name = {d_["name"]: d_ for d_ in gd["datasets"]}
    out = {"source_gd_file": gd_path.name, "penalty_label": args.penalty_label, "preregistered_gd_run": bool(gd.get("preregistered", True)),
           "assumption_grid": {"price_crowd_per_rating_usd": PRICE_CROWD, "price_expert_per_rating_usd": PRICE_EXPERT, "wage_per_hour_usd": WAGES,
                               "researcher_month_usd": RESEARCHER_MONTH, "hours_per_month": HOURS_PER_MONTH,
                               "note": "the crowd price is DERIVED (POPQUORN's $12 per hour and 13 minutes per 50 offensiveness annotations, about $0.052 each), not a published per-rating price; "
                                       "every other dollar figure is an assumption"},
           "definitions": {"saturation_margin_nrps": SAT_MARGIN, "saturation_relative_margin": SAT_REL, "delta_grid_nrps": DELTAS,
                           "note": "margins and the power rule were fixed in this script before it ran; they are not in the preregistration"},
           "datasets": {}}

    for ds in gd["datasets"]:
        name = ds["name"]
        sc2, sw2 = components(ds)
        g = ds["g_study"]
        n_items, n_ratings_all = g["n_items"], g["n_ratings"]
        k_med = g["ratings_per_item_median"]
        rec = {"K": ds["K"], "n_train_items": ds["n_train"], "n_test_items": ds["n_test"], "ratings_per_item_median": k_med,
               "all_ratings_in_dataset": n_ratings_all,
               "whole_dataset_cost_crowd_usd": n_ratings_all * PRICE_CROWD, "whole_dataset_cost_expert_usd": n_ratings_all * PRICE_EXPERT,
               "icc1": g["icc1"], "icc1_ci95": g["icc1_ci95"], "k_star_080": g["k_star_080"], "k_star_080_ci95": g["k_star_080_ci95"]}
        pv = ds["paired_vs_constant"]
        rec["carries_signal"] = bool(pv["ce_beats_constant_ci_excludes_0"])
        curve = ds["learning_curve_k1"]
        ns_sorted = sorted(int(k) for k in curve)
        largest = ns_sorted[-1]
        # train side.  A saturation point is only defined when the CE-trained head beats the constant forecast; otherwise the curve is flat at the constant's level
        # and "saturation" would price labels that buy nothing.
        n_sat = saturation(curve) if rec["carries_signal"] else None
        # gain of the full-training head over the constant forecast, read at the LARGEST N (a fixed point of the curve, not the minimum of the test curve)
        rec["gain_of_full_training_head_over_constant"] = ds["constant_test_nrps"] - curve[str(largest)]
        rec["largest_n_on_curve"] = largest
        rec["train_saturation_items_k1_relative10"] = (saturation(curve, SAT_REL * rec["gain_of_full_training_head_over_constant"])
                                                       if rec["carries_signal"] and rec["gain_of_full_training_head_over_constant"] > 0 else None)
        rec["train_saturation_items_k1"] = n_sat
        rec["train_saturation_ratings_k1"] = n_sat
        rec["train_saturation_cost_crowd_usd"] = None if n_sat is None else n_sat * PRICE_CROWD
        rec["train_saturation_cost_expert_usd"] = None if n_sat is None else n_sat * PRICE_EXPERT
        rec["current_train_items_exceed_saturation"] = None if n_sat is None else bool(ds["n_train"] >= n_sat)
        gain_curve = ds.get("learning_curve_k1_gain_over_constant_paired")
        rec["learning_curve_gain_over_constant_paired"] = gain_curve
        rec["smallest_n_from_which_gain_over_constant_is_significant"] = (smallest_n_from_which_gain_is_significant(gain_curve) if gain_curve else None)
        rec["learning_curve_seed_means"] = ds.get("learning_curve_k1_seed_means")
        rec["learning_curve_last_step"] = ds.get("learning_curve_last_step_paired")
        # test side (paired CE head versus the constant forecast)
        sd = pv["sd_per_item_diff"]
        rec["ce_head_vs_constant"] = {"gain": pv["mean_gain"], "ci95": [pv["lo"], pv["hi"]], "ci_excludes_0": pv["ce_beats_constant_ci_excludes_0"], "sd_per_item": sd}
        rec["test_items_to_detect_observed_gain_80pct_power"] = ((Z * sd / pv["mean_gain"]) ** 2) if pv["mean_gain"] > 0 else None

        # scale the paired sd from the median ratings per item down to k ratings per item with the single-judge variance-component ratio
        def sd_at_k(k, sd=sd, sc2=sc2, sw2=sw2, k_med=k_med):
            return sd * math.sqrt((sc2 + sw2 / k) / (sc2 + sw2 / max(k_med, 1)))

        rec["test_items_needed"] = {}
        for delta in DELTAS:
            row = {}
            for k in (1, 2, 4, 8):
                n_t = (Z * sd_at_k(k) / delta) ** 2
                row[f"k{k}"] = {"items": n_t, "ratings": n_t * k, "cost_crowd_usd": n_t * k * PRICE_CROWD, "cost_expert_usd": n_t * k * PRICE_EXPERT}
            rec["test_items_needed"][str(delta)] = row
        rec["test_items_have"] = ds["n_test"]
        rec["test_items_scaling_assumption"] = ("paired sd measured with each test item's own ratings (median k); sd at other k scaled by "
                                                "sqrt((sc2 + sw2/k)/(sc2 + sw2/k_med)) using the single-judge loss components: an approximation, not measured for the paired difference")
        # allocation optimum (a TEST-SIDE statement: precision of a held-out risk estimate)
        ratio = (sw2 / sc2) if sc2 > 0 else float("inf")
        rec["loss_components"] = {"sc2": sc2, "sw2": sw2, "share_between_items": sc2 / (sc2 + sw2) if sc2 + sw2 > 0 else None}
        rec["k_opt_by_item_to_rating_cost_ratio"] = {str(r): (max(1.0, math.sqrt(r * ratio)) if np.isfinite(ratio) else None) for r in (0, 1, 10, 100)}
        rec["iso_budget"] = ds.get("iso_budget")
        rec["iso_budget_dropped"] = ds.get("iso_budget_dropped")
        out["datasets"][name] = rec

    out["residual_gap_power"] = residual_power(Path(args.sens), gd_by_name)

    # ---- researcher-month comparison: three definitions of "enough training items" ----
    def usd_block(n_items_):
        c_crowd, c_exp = n_items_ * PRICE_CROWD, n_items_ * PRICE_EXPERT
        return {"items": n_items_, "crowd_usd": c_crowd, "expert_usd": c_exp,
                "crowd_researcher_months_at": {str(m): c_crowd / m for m in RESEARCHER_MONTH}, "expert_researcher_months_at": {str(m): c_exp / m for m in RESEARCHER_MONTH}}

    comp = {}
    for name, rec in out["datasets"].items():
        comp[name] = {}
        for label, n_ in (("first_N_with_significant_gain_over_constant", rec["smallest_n_from_which_gain_over_constant_is_significant"]),
                          ("saturation_margin_0.001", rec["train_saturation_items_k1"]),
                          ("saturation_margin_10pct_of_gain", rec["train_saturation_items_k1_relative10"])):
            if n_ is not None:
                comp[name][label] = usd_block(n_)
        comp[name]["whole_dataset_crowd"] = {"usd": rec["whole_dataset_cost_crowd_usd"], "researcher_months_at": {str(m): rec["whole_dataset_cost_crowd_usd"] / m for m in RESEARCHER_MONTH}}
        comp[name]["whole_dataset_expert"] = {"usd": rec["whole_dataset_cost_expert_usd"], "researcher_months_at": {str(m): rec["whole_dataset_cost_expert_usd"] / m for m in RESEARCHER_MONTH}}
    out["label_cost_in_researcher_months"] = comp

    # sister-repo formalization-quality packet: hours and dollars from measured versus length-scaled minutes per rating
    sc = pins["scenarios"]
    mins = {"measured_median_minutes_per_rating": pins["measured_minutes_per_rating"]["current_main"]["median_minutes"],
            "assumed_length_scaled_minutes_per_rating": pins["assumed_minutes_if_time_scaled_with_length"]}
    packet = {}
    for sk, s in sc.items():
        row = {"new_development_tasks": s["new_development_tasks"], "evaluation_tasks": s["evaluation_tasks"], "ratings": s["total_independent_ratings"], "scenarios": {}}
        for label, mpr in mins.items():
            hours = s["total_independent_ratings"] * mpr / 60.0
            row["scenarios"][label] = {"hours": hours, "usd_by_wage": {str(w): hours * w for w in WAGES},
                                       "researcher_months_by_wage_and_month_cost": {f"wage{w}_month{m}": hours * w / m for w in WAGES for m in RESEARCHER_MONTH},
                                       "rater_months_of_time_at_160h": hours / HOURS_PER_MONTH}
        packet[sk] = row
    out["formalization_packet_from_sister_planning_file"] = {"minutes_per_rating": mins, "scenarios": packet,
                                                           "caveat": "minutes per rating: 4.0 is a measured median over 103 entries from 2 raters; 37.3 is an assumption that time scales with file length; "
                                                                     "ratings counts are the sister repo's planning arithmetic for a half-width-0.005 paired RPS contrast at 25-task paired sd 0.0307"}
    (RES / f"sufficiency_and_dollars{tag}.json").write_text(json.dumps(jsonable(out), indent=1))

    # ---- markdown tables ----
    present = [n for n in ORDER if n in out["datasets"]]
    L = [f"Judge: CE-trained 24-component linear head on hashed text. How its ridge penalty was chosen: {args.penalty_label}. Source file: {gd_path.name}.", "",
         "Prices: the $0.05 crowd price per rating is DERIVED from POPQUORN's $12 per hour and 13 minutes per 50 offensiveness annotations (about $0.052), not a published per-rating price; "
         "every other dollar figure is an assumption. See the header of scripts/sufficiency_and_dollars.py.", "",
         "### A. Real-data G-study and D-study (items random, ratings nested in items; one-way unbalanced analysis of variance; independent of the ridge penalty)", "",
         "| dataset | K | single-rating reliability, intraclass correlation ICC(1) [95% bootstrap] | ratings per item for reliability 0.80 (k*) [95%] | median ratings per item available |",
         "|---|---|---|---|---|"]
    for name in present:
        r = out["datasets"][name]
        ci, kci = r["icc1_ci95"], r["k_star_080_ci95"]
        L.append(f"| {name} | {r['K']} | {r['icc1']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {r['k_star_080']} [{kci[0]}, {kci[1]}] | {r['ratings_per_item_median']:.0f} |")

    L += ["", "### B. Training items: when does the CE-trained head beat the constant forecast, and when does its curve flatten?", "",
          "| dataset | carries signal | train items available | gain of the full-training head over the constant forecast [95% paired, item-clustered] | first N from which the gain stays significant | saturation N (k=1, margin 0.001) | saturation N (margin 10% of the full-training gain) | last doubling: drop in test normalized RPS [95%] |",
          "|---|---|---|---|---|---|---|---|"]
    for name in present:
        r = out["datasets"][name]
        pv = r["ce_head_vs_constant"]
        sig = r["smallest_n_from_which_gain_over_constant_is_significant"]
        sig_s = "n/a (no signal)" if not r["carries_signal"] else (f"{sig}" if sig is not None else "not reached on the grid")
        sat = f"{r['train_saturation_items_k1']}" if r["carries_signal"] else "n/a (no signal)"
        rel10 = r["train_saturation_items_k1_relative10"]
        rel_s = f"{rel10}" if rel10 is not None else "n/a (no signal)"
        ls = r["learning_curve_last_step"]
        ls_s = f"{ls['n_from']} to {ls['n_to']}: {ls['mean_drop']:+.4f} [{ls['lo']:+.4f}, {ls['hi']:+.4f}]" if ls else "n/a"
        L.append(f"| {name} | {'yes' if r['carries_signal'] else 'no'} | {r['n_train_items']} | {pv['gain']:+.4f} [{pv['ci95'][0]:+.4f}, {pv['ci95'][1]:+.4f}] | {sig_s} | {sat} | {rel_s} | {ls_s} |")
    L += ["", "gain = test normalized ranked probability score (RPS; RPS divided by K - 1; 0 to 1; lower is better) of the constant forecast minus that of the CE-trained head at its penalty (positive = the head is better). "
          "Carries signal = that gain has a paired 95% item-clustered interval above 0 at the full training set. Saturation N = the smallest N on the grid such that no larger N improves the head's mean test curve by more than the margin "
          "(a descriptive summary of one noisy curve: mean of 3 training-subsample draws, one prompt-level split; the margins were fixed in the analysis script, not preregistered). "
          "The 10% margin is 10% of the gain of the full-training head over the constant, read at the largest N, not at the minimum of the test curve. An N equal to the training items available means the curve had not flattened within the data.", "",
          "### C. Test items needed (paired comparison at 80% power, two-sided 5%)", "",
          "| dataset | test items available | test items to detect the head's actual gain over the constant | test items for a gain of 0.005 (k = 1) | for a gain of 0.0016 (k = 1) |", "|---|---|---|---|---|"]
    for name in present:
        r = out["datasets"][name]
        obs = r["test_items_to_detect_observed_gain_80pct_power"]
        obs_s = f"{obs:,.0f}" if obs is not None else "n/a (no gain)"
        L.append(f"| {name} | {r['n_test_items']} | {obs_s} | {r['test_items_needed']['0.005']['k1']['items']:,.0f} | {r['test_items_needed']['0.0016']['k1']['items']:,.0f} |")
    L += ["", "Items to detect the actual gain use each test item's own ratings; the 0.005 and 0.0016 columns rescale the paired standard deviation to one rating per item (approximation, see the header). "
          "Where the head's gain over the constant is estimated near zero the first column is not meaningful.", "",
          "### D. Cost-minimizing ratings per item for the proper-score RISK ESTIMATE (a test-side precision statement, not a training recommendation)", "",
          "| dataset | share of loss variance between items | k_opt if an item costs 0 extra | 1x a rating | 10x | 100x | D-study k* for reliability 0.80 |", "|---|---|---|---|---|---|---|"]
    for name in present:
        r = out["datasets"][name]
        ko = r["k_opt_by_item_to_rating_cost_ratio"]
        L.append(f"| {name} | {r['loss_components']['share_between_items']:.2f} | {ko['0']:.1f} | {ko['1']:.1f} | {ko['10']:.1f} | {ko['100']:.1f} | {r['k_star_080']} |")

    L += ["", "### E. Label dollars against researcher-months (assumption grid), three definitions of enough training items", "",
          "| dataset | definition of enough | training items (k = 1) | crowd, $0.05 derived | expert, $5 assumed | expert, in researcher-months at $30k assumed |", "|---|---|---|---|---|---|"]
    no_signal = []
    names_def = (("first_N_with_significant_gain_over_constant", "first N with a significant gain"), ("saturation_margin_0.001", "saturation, margin 0.001"),
                 ("saturation_margin_10pct_of_gain", "saturation, margin 10% of the gain"))
    for name in present:
        if not out["datasets"][name]["carries_signal"]:
            no_signal.append(name)
            continue
        for key, lab in names_def:
            b = out["label_cost_in_researcher_months"][name].get(key)
            if b is None:
                continue
            full = " (curve not flat within the data)" if b["items"] >= out["datasets"][name]["n_train_items"] else ""
            L.append(f"| {name} | {lab} | {b['items']:,}{full} | ${b['crowd_usd']:,.2f} | ${b['expert_usd']:,.0f} | {b['expert_researcher_months_at']['30000']:.3f} |")
    L += ["", "Not priced (the CE-trained head does not beat the constant forecast on these, so there is no point to price): " + (", ".join(no_signal) if no_signal else "none") + "."]

    rp = out["residual_gap_power"]
    if rp:
        L += ["", "### F. Test items needed to resolve the small gap that is LEFT after each head gets its own calibration map", "",
              "Pearson-trained head minus CE-trained head, both followed by their own validation-fitted calibration map (L2-sensitivity K-sweep; five split seeds; NOT preregistered). "
              "Difference = test normalized RPS of the Pearson-trained head minus test normalized RPS of the CE-trained head; positive means the CE-trained head is better. "
              "The standard deviation of the per-item paired difference is implied by the interval half-width (normal approximation).", "",
              "| view | K | test items available | calibrated gap, mean of 5 seeds | seeds with the 95% interval above / below 0 | minimum detectable difference at the available items (80% power) | items to resolve 0.0022 | items to resolve 0.001 | crowd dollars for 0.0022 at the dataset's own ratings per item |",
              "|---|---|---|---|---|---|---|---|---|"]
        for view, r in sorted(rp.items()):
            usd = r["crowd_usd_at_median_k_0.0022"]
            usd_s = f"${usd:,.0f}" if usd is not None else "n/a"
            L.append(f"| {view} | {r['K']} | {r['n_test_items']} | {r['calibrated_gap_seed_mean']:+.4f} | {r['calibrated_gap_seeds_ci_above_0']} / {r['calibrated_gap_seeds_ci_below_0']} of {r['n_seeds']} | "
                     f"{r['min_detectable_difference_at_available_items']:.4f} | {r['items_to_resolve_0.0022']:,.0f} | {r['items_to_resolve_0.001']:,.0f} | {usd_s} |")
        L += ["", "Dollars = items x the dataset's median ratings per item x $0.05 (derived crowd price); datasets with many ratings per item (ChaosNLI 100, DICES 66) show large dollars here because that is how they were collected, "
              "not because the item count requires it. A larger target difference needs quadratically fewer items. The raw gap between the heads (before calibration) is 0.005 to 0.10 and is resolved by a few dozen to a few hundred test items.", ""]
    L += ["### G. Formalization-quality rating packet (sister repository planning numbers; minutes per rating is the uncertain input)", "",
          f"Minutes per rating: measured median {mins['measured_median_minutes_per_rating']} (103 entries, 2 raters) versus an assumed length-scaled {mins['assumed_length_scaled_minutes_per_rating']:.1f}.", "",
          "| scenario | ratings | hours at measured minutes | hours at length-scaled minutes | dollars at $50/hour (measured to scaled) | in researcher-months at $30k (measured to scaled) |", "|---|---|---|---|---|---|"]
    for sk, row in packet.items():
        a, b = row["scenarios"]["measured_median_minutes_per_rating"], row["scenarios"]["assumed_length_scaled_minutes_per_rating"]
        L.append(f"| {sk} | {row['ratings']} | {a['hours']:.1f} | {b['hours']:.1f} | ${a['usd_by_wage']['50']:,.0f} to ${b['usd_by_wage']['50']:,.0f} | "
                 f"{a['researcher_months_by_wage_and_month_cost']['wage50_month30000']:.2f} to {b['researcher_months_by_wage_and_month_cost']['wage50_month30000']:.2f} |")
    (RES / f"sufficiency_and_dollars_tables{tag}.md").write_text("\n".join(L) + "\n")

    # ---- figures (one axis per chart; every axis gridded, minor gridlines on log axes) ----
    cols = {n: c for n, c in zip(ORDER, [BLUE, ORANGE, AQUA, "#6a4ce2", "#f2a600", "#ee7fac", "#008a17"])}   # one fixed colour per dataset in every figure

    def style(ax, log_x=False):
        ax.grid(True, which="major", linewidth=0.6, color=GRID)
        if log_x:
            ax.grid(True, which="minor", axis="x", linewidth=0.3, color=GRID, alpha=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#bdbdbd")
        ax.tick_params(colors=TEXT2, labelsize=9)

    judge_line = f"Judge: CE-trained 24-component linear head on hashed text. Penalty: {args.penalty_label}. Mean of 3 training-subsample draws, one prompt-level split (seed 0)."

    # Figure 1: D-study reliability curves
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    style(ax)
    ks = np.arange(1, 41)
    for name in present:
        r = out["datasets"][name]
        icc = r["icc1"]
        ax.plot(ks, icc / (icc + (1 - icc) / ks), color=cols[name], linewidth=2.0, label=f"{name} (single-rating intraclass correlation {icc:.2f}, k* {r['k_star_080']})")
    ax.axhline(0.8, color="#8a8a8a", linewidth=1.0)
    ax.text(40, 0.805, "reliability 0.80", ha="right", va="bottom", fontsize=8.5, color=TEXT2)
    ax.set_xlabel("Ratings per item k (D-study projection from the real ratings' variance components)")
    ax.set_ylabel("Reliability of an item's mean rating, E-rho-squared(k)\n(higher is better; fixed by the variance components, independent of any judge)")
    fig.suptitle("Classical D-study on real ratings: ratings per item needed for the item MEAN to reach reliability 0.80", x=0.01, ha="left", fontsize=11.5, color=TEXT1, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / f"g_d_study_reliability_curves{tag}.png", dpi=150)
    plt.close(fig)

    # Figure 2: learning curves at k = 1 as the gain over the constant forecast (does not use the minimum of the test curve)
    fig, ax = plt.subplots(figsize=(11.0, 7.4))
    style(ax, log_x=True)
    for name in present:
        r = out["datasets"][name]
        ds = gd_by_name[name]
        cur = ds["learning_curve_k1"]
        ns = sorted(int(k) for k in cur)
        y = np.array([ds["constant_test_nrps"] - cur[str(n)] for n in ns])
        sig = r["carries_signal"]
        ax.plot(ns, y, color=cols[name], linewidth=2.0 if sig else 1.4, linestyle="-" if sig else (0, (4, 3)), marker="o", markersize=5,
                label=f"{name} (K={ds['K']}{'' if sig else ', no significant gain at full training'})")
        gc = r["learning_curve_gain_over_constant_paired"]
        if gc:
            lo = np.array([gc[str(n)]["lo"] for n in ns])
            hi = np.array([gc[str(n)]["hi"] for n in ns])
            ax.fill_between(ns, lo, hi, color=cols[name], alpha=0.12, linewidth=0)
        n10 = r["train_saturation_items_k1_relative10"]
        if n10 is not None:
            ax.plot([n10], [ds["constant_test_nrps"] - cur[str(n10)]], marker="D", markersize=8, color=cols[name], markeredgecolor="white", markeredgewidth=1.5, linestyle="none")
    ax.axhline(0.0, color="#8a8a8a", linewidth=1.0)
    ax.text(0.01, 0.0, "constant forecast (no judge)", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=8.5, color=TEXT2)
    ax.set_xscale("log")
    ax.set_xlabel("Training items N (one rating per item, k = 1; log scale)")
    ax.set_ylabel("Held-out test normalized ranked probability score (RPS) of the constant forecast\nminus that of the CE-trained head (positive = the head is better; lower RPS is better)")
    fig.suptitle(textwrap.fill("Learning curves: gain of the CE-trained head over the constant forecast. Diamond = smallest N after which no larger N adds more than 10% of the full-training gain", 118),
                 x=0.01, ha="left", fontsize=11, color=TEXT1, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(rect=(0, 0.075, 1, 0.925))
    foot = judge_line + (" Band = 95% interval from resampling test prompts (test-prompt sampling only; the spread over training-subsample draws is in the JSON)." if any(
        out["datasets"][n]["learning_curve_gain_over_constant_paired"] for n in present) else " No band: this input file has no per-point intervals.")
    fig.text(0.01, 0.008, textwrap.fill(foot, 175), fontsize=7.5, color=TEXT2, va="bottom")
    fig.savefig(FIG / f"learning_curves_and_saturation{tag}.png", dpi=150)
    plt.close(fig)

    # Figure 3: label dollars in researcher-months (log x), one chart, priced at the 10%-margin saturation point
    fig, ax = plt.subplots(figsize=(12.0, 7.6))
    style(ax, log_x=True)
    rows = []
    for name in [n for n in present if out["datasets"][n]["carries_signal"]]:
        b = out["label_cost_in_researcher_months"][name].get("saturation_margin_10pct_of_gain")
        if b is None:
            continue
        rows.append((f"{name}: {b['items']:,} training labels, crowd $0.05 (derived)", b["crowd_researcher_months_at"]["30000"], BLUE))
        rows.append((f"{name}: {b['items']:,} training labels, expert $5 (assumed)", b["expert_researcher_months_at"]["30000"], ORANGE))
    for sk, row in packet.items():
        if "development_pool_and_reach_half_width_0.005" in sk:
            a = row["scenarios"]["measured_median_minutes_per_rating"]["researcher_months_by_wage_and_month_cost"]["wage50_month30000"]
            b2 = row["scenarios"]["assumed_length_scaled_minutes_per_rating"]["researcher_months_by_wage_and_month_cost"]["wage50_month30000"]
            rows.append((f"formalization packet, {row['ratings']:,} ratings: measured 4 min each, $50/h (assumed wage)", a, AQUA))
            rows.append((f"formalization packet, {row['ratings']:,} ratings: length-scaled 37 min each, $50/h (assumed)", b2, AQUA))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, val, col) in zip(ys, rows):
        ax.plot([min(val, 1e-4), val], [y, y], color=col, linewidth=2.0)
        ax.plot([val], [y], marker="o", markersize=8, color=col, markeredgecolor="white", markeredgewidth=1.5, linestyle="none")
    ax.axvline(1.0, color="#8a8a8a", linewidth=1.0)
    ax.text(0.93, ys.max() - 0.3, esc("one researcher-month\n($30k, assumed)"), fontsize=8.5, color=TEXT2, va="top", ha="right")
    ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels([esc(r[0]) for r in rows], fontsize=8)
    ax.set_xlabel(esc("Cost of the label set, in researcher-months at an assumed $30,000 per month\n(log scale; lower is cheaper)"))
    fig.suptitle(textwrap.fill("Labels versus researcher time: training labels at the point where the learning curve flattens. Every price is derived or assumed (see legend)", 110),
                 x=0.01, ha="left", fontsize=11.0, color=TEXT1, fontweight="bold")
    handles = [plt.Line2D([], [], marker="o", color=c, markersize=8, markeredgecolor="white", markeredgewidth=1.2, linewidth=2.0, label=lab)
               for c, lab in ((BLUE, esc("crowd price $0.05 per rating (derived from POPQUORN's $12 per hour and 13 minutes per 50 annotations)")), (ORANGE, esc("expert price $5 per rating (assumed)")),
                              (AQUA, esc("formalization packet from the sister repo's planning file ($50/hour assumed)")))]
    fig.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5, bbox_to_anchor=(0.01, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 0.90))
    foot3 = (f"Training labels = the smallest N after which no larger N improves the CE-trained head's test curve by more than 10% of its full-training gain over the constant. "
             f"Judge: CE-trained 24-component linear head on hashed text. Penalty: {args.penalty_label}.")
    fig.text(0.01, 0.125, esc(textwrap.fill(foot3, 185)), fontsize=7.5, color=TEXT2, va="bottom")
    fig.savefig(FIG / f"label_dollars_vs_researcher_month{tag}.png", dpi=150)
    plt.close(fig)
    print("wrote", RES / f"sufficiency_and_dollars{tag}.json")
    print((RES / f"sufficiency_and_dollars_tables{tag}.md").read_text()[:7000])


if __name__ == "__main__":
    main()
