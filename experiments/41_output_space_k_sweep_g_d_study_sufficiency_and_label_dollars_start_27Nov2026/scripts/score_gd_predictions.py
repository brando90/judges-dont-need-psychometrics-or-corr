#!/usr/bin/env python3
"""Scores preregistered predictions P6 and P7 and tabulates the real-data ceiling check (Experiment 41).

    python3 -I score_gd_predictions.py [--gd results/gd_study_l2_validation_chosen.json] [--tag _l2_validation_chosen]
    python3 -I score_gd_predictions.py --gd results/gd_study_and_allocation.json --tag _preregistered_l2_0.01     # the preregistered fixed-penalty run

Reads  --gd (default results/gd_study_l2_validation_chosen.json)
Writes results/gd_prediction_scoring{tag}.json, results/gd_prediction_scoring_tables{tag}.md

P6 (D-study).  On POPQUORN (both tasks) and SummEval crowd the D-study k* for reliability 0.80 is 8 or more.  (Independent of any judge and of the ridge penalty.)
P7 (allocation). At the same total budget B = N x k, the CE-trained head trained at k = 1 is not worse on held-out normalized RPS than the same head trained at the D-study k*, within the
                 bootstrap interval.  Per (dataset, budget) cell:
                 diff = test normalized RPS of the k*-ratings-per-item head minus test normalized RPS of the one-rating-per-item head, same total budget.
                 diff > 0 = the one-rating-per-item head is better.   A cell REFUTES P7 only if the 95% paired item-clustered interval of diff is entirely below 0.
                 A cell is a TIE if the interval contains 0.   Every (dataset, budget) cell is listed, including a cell where no k* design can be formed
                 (fewer than 16 training items at that k*) and a cell where k* exceeds the ratings per item available (k* is then capped).
                 Cells on sets where the CE-trained head does not beat the constant forecast are counted separately: they compare two heads that both carry no signal, so they
                 say little about label allocation.
Ceiling check.   Theory (Lean: ceiling_sq, icc_eq_rho; numerics: numerics_theorems.py E3, F1-F3): a sampled (honest) judge has Corr = rho = ICC(1);
                 the deterministic item mean has Corr = sqrt(rho).  Here both are computed from the real ratings by leave-one-rating-out:
                 for each (item, rating j) the "judge" is built from the OTHER ratings of the item.  With n - 1 other ratings the collapsed-mean
                 correlation is rho / sqrt(rho + (1 - rho)/(n - 1)), which tends to sqrt(rho); the table prints that finite-n value beside the measured one.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
RES = EXP / "results"
ORDER = ["popquorn_offensiveness", "popquorn_politeness", "summeval_expert", "summeval_crowd", "dices990", "chaosnli_snli_mnli", "chaosnli_alpha"]
BUDGETS = ("128", "256", "512")


def verdict(m):
    if m["hi"] < 0:
        return "k*_better (refutes P7)"
    if m["lo"] > 0:
        return "k=1_better"
    return "tie"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gd", default=str(RES / "gd_study_l2_validation_chosen.json"))
    ap.add_argument("--tag", default="_l2_validation_chosen")
    args = ap.parse_args()
    tag = args.tag
    gd_path = Path(args.gd)
    gd = json.loads(gd_path.read_text())
    by = {d["name"]: d for d in gd["datasets"]}
    names = [n for n in ORDER if n in by]
    out = {"source_gd_file": gd_path.name, "preregistered_gd_run": bool(gd.get("preregistered", True))}

    # ---- P6 ----
    p6 = {n: {"k_star_080": by[n]["g_study"]["k_star_080"], "ci95": by[n]["g_study"]["k_star_080_ci95"], "icc1": by[n]["g_study"]["icc1"]}
          for n in ("popquorn_offensiveness", "popquorn_politeness", "summeval_crowd")}
    out["P6"] = {"rule": "D-study k* for reliability 0.80 is at least 8 on POPQUORN (both tasks) and SummEval crowd", "values": p6,
                 "supported": all(v["k_star_080"] is not None and v["k_star_080"] >= 8 for v in p6.values()),
                 "meets_8": [n for n, v in p6.items() if v["k_star_080"] >= 8], "below_8": [n for n, v in p6.items() if v["k_star_080"] < 8]}

    # ---- P7: every (dataset, budget) cell ----
    cells, not_formed = [], []
    for n in names:
        d = by[n]
        sig = bool(d["paired_vs_constant"]["ce_beats_constant_ci_excludes_0"])
        for b in BUDGETS:
            if b not in d["iso_budget"]:
                not_formed.append({"dataset": n, "budget_ratings": int(b), "carries_signal": sig,
                                   "reason": (d.get("iso_budget_dropped") or {}).get(b, "no iso-budget entry for this budget in the input file")})
                continue
            e = d["iso_budget"][b]
            k_used = e["k_star_used"]
            plans = {p["k"]: p for p in e["plans"]}
            if k_used not in plans:
                not_formed.append({"dataset": n, "budget_ratings": int(b), "carries_signal": sig, "k_star_080": e["k_star_080"],
                                   "reason": f"the k* = {k_used} design would have {int(b) // k_used} training items, fewer than the 16 required; plans formed: k = {sorted(plans)}"})
                continue
            plan = plans[k_used]
            m = plan["k_minus_k1_paired"]
            best_k = e["best_k"]
            cell = {"dataset": n, "budget_ratings": int(b), "k_star_080": e["k_star_080"], "k_star_used": k_used,
                    "k_star_truncated_by_available_ratings": e["k_star_truncated_by_available_ratings"], "N_items_at_k_star": plan["N"],
                    "diff_mean": m["mean"], "diff_lo": m["lo"], "diff_hi": m["hi"], "carries_signal": sig, "verdict": verdict(m),
                    "best_k_by_test_mean": best_k,
                    "all_plans": [{"k": p["k"], "N": p["N"], "test_nrps": p["test_nrps"], "chosen_lambda": p.get("chosen_lambda", "0.01 (fixed)"),
                                   "diff_vs_k1": p["k_minus_k1_paired"], "verdict": verdict(p["k_minus_k1_paired"]) if p["k"] != 1 else "baseline"} for p in e["plans"]]}
            cells.append(cell)
    refuting = [c for c in cells if c["verdict"].startswith("k*_better")]
    better = [c for c in cells if c["verdict"] == "k=1_better"]
    ties = [c for c in cells if c["verdict"] == "tie"]
    sig_cells = [c for c in cells if c["carries_signal"]]
    nosig_cells = [c for c in cells if not c["carries_signal"]]
    n_cells_total = len(cells) + len(not_formed)
    out["P7"] = {"rule": "at the same total budget the one-rating-per-item head is not worse than the D-study-k* head (refuted by a cell only when the paired 95% interval of "
                         "[k* head minus one-rating head] lies entirely below 0)",
                 "n_cells_total": n_cells_total, "n_cells_evaluable": len(cells), "n_cells_k_star_design_not_formed": len(not_formed), "not_formed_cells": not_formed,
                 "n_k1_better": len(better), "n_tie": len(ties), "n_refuting": len(refuting), "refuting_cells": refuting,
                 "tied_cells": [(c["dataset"], c["budget_ratings"]) for c in ties],
                 "on_signal_views": {"n_cells": len(sig_cells), "n_k1_better": sum(c["verdict"] == "k=1_better" for c in sig_cells), "n_tie": sum(c["verdict"] == "tie" for c in sig_cells),
                                     "n_refuting": sum(c["verdict"].startswith("k*_better") for c in sig_cells),
                                     "n_cells_k_star_design_not_formed": sum(1 for x in not_formed if x["carries_signal"])},
                 "on_no_signal_views": {"n_cells": len(nosig_cells), "n_k1_better": sum(c["verdict"] == "k=1_better" for c in nosig_cells), "n_tie": sum(c["verdict"] == "tie" for c in nosig_cells),
                                        "n_refuting": sum(c["verdict"].startswith("k*_better") for c in nosig_cells)},
                 "k_star_truncated_cells": [(c["dataset"], c["budget_ratings"], c["k_star_080"], c["k_star_used"]) for c in cells if c["k_star_truncated_by_available_ratings"]],
                 "supported_every_evaluable_cell": len(refuting) == 0, "cells": cells}

    # ---- ceiling check on real ratings ----
    ceil = {}
    for n in names:
        d = by[n]
        c = d["ceiling_on_real_ratings"]
        rho = d["g_study"]["icc1"]
        n_other = d["g_study"]["ratings_per_item_median"] - 1
        pred_collapsed = rho / math.sqrt(rho + (1 - rho) / max(n_other, 1))
        ceil[n] = {"icc1_rho": rho, "sampled_draw_corr_measured": c["corr_sampled_draw_of_others"],
                   "sqrt_rho": math.sqrt(rho), "collapsed_mean_corr_measured": c["corr_collapsed_mean_of_others"],
                   "collapsed_mean_corr_finite_n_formula_at_median_n": pred_collapsed,
                   "lattice_rounded_mean_corr_measured": c["corr_lattice_rounded_mean_of_others"],
                   "rounded_still_above_sampled": (c["corr_lattice_rounded_mean_of_others"] is not None and c["corr_lattice_rounded_mean_of_others"] > c["corr_sampled_draw_of_others"]),
                   "n_pairs": c["n_pairs"]}
    out["ceiling_check"] = ceil
    (RES / f"gd_prediction_scoring{tag}.json").write_text(json.dumps(out, indent=1, default=float))

    L = [f"Source file: {gd_path.name} (preregistered fixed-penalty run: {out['preregistered_gd_run']}).", "",
         "### P6 (D-study k* for reliability 0.80 on POPQUORN and SummEval crowd, predicted at least 8)", "",
         "| dataset | single-rating intraclass correlation ICC(1) | k* [95% bootstrap] | meets the predicted 8 or more |", "|---|---|---|---|"]
    for n, v in p6.items():
        L.append(f"| {n} | {v['icc1']:.3f} | {v['k_star_080']} [{v['ci95'][0]}, {v['ci95'][1]}] | {'yes' if v['k_star_080'] >= 8 else 'NO'} |")
    L += ["", f"P6: **{'SUPPORTED' if out['P6']['supported'] else 'NOT SUPPORTED'}** (met on {out['P6']['meets_8']}; missed on {out['P6']['below_8']}).", "",
          "### P7 (equal total rating budget: the one-rating-per-item head against the head trained at the D-study k*)", "",
          "diff = test normalized ranked probability score (RPS divided by K - 1; lower is better) of the k*-ratings-per-item head minus that of the one-rating-per-item head, same total budget, "
          "each head at its own penalty (" + ("chosen per cell on validation prompts" if not out["preregistered_gd_run"] else "fixed at 0.01") + "); positive = the one-rating-per-item head is better. "
          "Interval = 95% paired, item-clustered (test prompts resampled, 1000 draws).", "",
          "| dataset | carries signal | budget (ratings) | k* used (truncated?) | items at k* | diff [95% paired item-clustered] | verdict | k with the lowest test mean |", "|---|---|---|---|---|---|---|---|"]
    order_idx = {n: i for i, n in enumerate(ORDER)}
    for c in sorted(cells + [dict(x, _nf=True) for x in not_formed], key=lambda c: (order_idx[c["dataset"]], c["budget_ratings"])):
        if c.get("_nf"):
            L.append(f"| {c['dataset']} | {'yes' if c['carries_signal'] else 'no'} | {c['budget_ratings']} | not formed | n/a | n/a | not evaluated: {c['reason']} | n/a |")
            continue
        L.append(f"| {c['dataset']} | {'yes' if c['carries_signal'] else 'no'} | {c['budget_ratings']} | {c['k_star_used']}{' (capped from ' + str(c['k_star_080']) + ')' if c['k_star_truncated_by_available_ratings'] else ''} | "
                 f"{c['N_items_at_k_star']} | {c['diff_mean']:+.4f} [{c['diff_lo']:+.4f}, {c['diff_hi']:+.4f}] | {c['verdict']} | {c['best_k_by_test_mean']} |")
    p7 = out["P7"]
    n_sig_sets = sum(bool(by[n]["paired_vs_constant"]["ce_beats_constant_ci_excludes_0"]) for n in names)
    L += ["", f"P7 over all {p7['n_cells_total']} cells: the k* design could be formed in {p7['n_cells_evaluable']}; in those, the one-rating-per-item head is strictly better in {p7['n_k1_better']}, "
              f"tied in {p7['n_tie']}, worse in {p7['n_refuting']}; {p7['n_cells_k_star_design_not_formed']} cell(s) not evaluated. "
              f"On the {n_sig_sets} set(s) where the CE-trained head beats the constant forecast: {p7['on_signal_views']['n_k1_better']} better, {p7['on_signal_views']['n_tie']} tied, "
              f"{p7['on_signal_views']['n_refuting']} worse of {p7['on_signal_views']['n_cells']} evaluable cells, plus {p7['on_signal_views']['n_cells_k_star_design_not_formed']} not formed. "
              f"On sets with no signal over the constant: {p7['on_no_signal_views']['n_k1_better']} better, {p7['on_no_signal_views']['n_tie']} tied, {p7['on_no_signal_views']['n_refuting']} worse of "
              f"{p7['on_no_signal_views']['n_cells']} (both heads carry no signal there, so those cells say little about label allocation). "
              f"**{'NOT REFUTED in any evaluable cell' if not refuting else 'REFUTED in ' + str(len(refuting)) + ' cell(s): ' + ', '.join(c['dataset'] + ' B=' + str(c['budget_ratings']) for c in refuting)}**.", "",
          "### P7, every design in every cell (diff of each k against k = 1 at the same budget)", "",
          "| dataset | budget | k | training items | test nRPS | penalty chosen | diff vs k = 1 [95%] | verdict |", "|---|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda c: (order_idx[c["dataset"]], c["budget_ratings"])):
        for p in c["all_plans"]:
            m = p["diff_vs_k1"]
            L.append(f"| {c['dataset']} | {c['budget_ratings']} | {p['k']}{' (k*)' if p['k'] == c['k_star_used'] and p['k'] != 1 else ''} | {p['N']} | {p['test_nrps']:.4f} | {p['chosen_lambda']} | "
                     f"{m['mean']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}] | {p['verdict']} |")
    L += ["", "### Ceiling check on real ratings (leave-one-rating-out; pooled over items and ratings)", "",
          "| dataset | ICC(1) = rho | sampled-judge Corr (theory: rho) | collapsed-mean Corr measured | finite-n formula | sqrt(rho) | lattice-rounded mean Corr | rounded still above sampled |", "|---|---|---|---|---|---|---|---|"]
    for n in names:
        c = ceil[n]
        rd = c["lattice_rounded_mean_corr_measured"]
        L.append(f"| {n} | {c['icc1_rho']:.3f} | {c['sampled_draw_corr_measured']:.3f} | {c['collapsed_mean_corr_measured']:.3f} | {c['collapsed_mean_corr_finite_n_formula_at_median_n']:.3f} | "
                 f"{c['sqrt_rho']:.3f} | {rd:.3f} | {'yes' if c['rounded_still_above_sampled'] else 'no'} |")
    (RES / f"gd_prediction_scoring_tables{tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
