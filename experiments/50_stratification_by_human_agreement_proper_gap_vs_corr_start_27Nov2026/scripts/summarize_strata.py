"""Experiment 50 summary: prediction checks R1-R3 and tables from results/arm_a_exp41_heads.json and results/arm_b_llm_judges.json.
Usage: python summarize_strata.py RESULTS_DIR   (writes strata_tables.md and strata_summary.json)
Operationalizations (fixed here, after the protocol, and stated in the README):
  R1 per view: seed mean of raw_gap(Q5) - raw_gap(Q1) > 0 and at least 3 of 5 seeds' 95% intervals above 0.
  R2: within-bin Pearson below the pooled test Pearson, counted per (view, seed, bin, head).
  R3 per set: top-1 agreement Q5 < Q1 for every judge; ranking stability = Kendall tau between judge rankings in Q1 and Q5, by top-1 and by S."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau

res = Path(sys.argv[1])
A = json.loads((res / "arm_a_exp41_heads.json").read_text())
B = json.loads((res / "arm_b_llm_judges.json").read_text())
SIGNAL_FALSE = ("summeval_crowd", "chaosnli_alphanli")
out, summ = [], {}

byv = defaultdict(list)
for r in A:
    byv[r["view"]].append(r)
out.append("## Arm A: Experiment 41 heads (raw and matched gap per entropy quintile, seed means; positive = CE-trained head better)\n")
out.append("| view | binning | raw gap Q1 | Q2 | Q3 | Q4 | Q5 | Q5 - Q1 (seed mean) | seeds Q5-Q1 CI > 0 / < 0 | R1 | matched gap Q1 / Q5 | Pearson (Pearson-trained head) pooled / mean within-bin |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
r1_rows, r2_counts = [], [0, 0]
for v, rs in byv.items():
    gaps = np.array([[b["raw_gap"] for b in r["bins"]] for r in rs])
    mg = np.array([[b["matched_gap"] for b in r["bins"]] for r in rs])
    d = np.array([r["raw_gap_Q5_minus_Q1"] for r in rs])
    up = sum(r["raw_gap_Q5_minus_Q1_ci"][0] > 0 for r in rs); dn = sum(r["raw_gap_Q5_minus_Q1_ci"][1] < 0 for r in rs)
    signal = not v.startswith(SIGNAL_FALSE)
    r1 = bool(d.mean() > 0 and up >= 3)
    for r in rs:
        for b in r["bins"]:
            for head, pooled in (("pearson_ce_head", r["pooled_pearson_ce_head"]), ("pearson_pearson_head", r["pooled_pearson_pearson_head"])):
                if b[head] is not None and np.isfinite(b[head]):
                    r2_counts[0] += 1; r2_counts[1] += bool(b[head] < pooled)
    pp = np.mean([r["pooled_pearson_pearson_head"] for r in rs])
    wb = np.nanmean([[b["pearson_pearson_head"] if b["pearson_pearson_head"] is not None else np.nan for b in r["bins"]] for r in rs])
    row = {"view": v, "signal": signal, "same_ratings_binning": rs[0]["same_ratings_binning"], "raw_gap_by_bin": gaps.mean(0).tolist(),
           "matched_gap_by_bin": mg.mean(0).tolist(), "Q5_minus_Q1_mean": float(d.mean()), "seeds_ci_above": int(up), "seeds_ci_below": int(dn),
           "R1_supported": r1, "pooled_pearson_pearson_head": float(pp), "mean_within_bin_pearson_pearson_head": float(wb)}
    r1_rows.append(row)
    out.append(f"| {v}{'' if signal else ' (no signal)'} | {'same ratings' if row['same_ratings_binning'] else 'split halves'} | " + " | ".join(f"{g:+.4f}" for g in gaps.mean(0))
               + f" | {d.mean():+.4f} | {up} / {dn} | {'yes' if r1 else 'no'} | {mg.mean(0)[0]:+.4f} / {mg.mean(0)[-1]:+.4f} | {pp:.3f} / {wb:.3f} |")
sig = [r for r in r1_rows if r["signal"]]
summ["arm_a"] = r1_rows
summ["R1"] = {"views_with_signal": len(sig), "supported": sum(r["R1_supported"] for r in sig),
              "opposite_direction_views": sum(r["Q5_minus_Q1_mean"] < 0 and r["seeds_ci_below"] >= 3 for r in sig)}
summ["R2"] = {"cells": r2_counts[0], "within_bin_below_pooled": r2_counts[1]}
out.append(f"\nR1: supported on {summ['R1']['supported']} of {len(sig)} views with signal; significantly the opposite way (Q5 gap smaller, >=3 seeds) on {summ['R1']['opposite_direction_views']}.")
out.append(f"R2: within-bin Pearson below pooled Pearson in {r2_counts[1]} of {r2_counts[0]} (view, seed, bin, head) cells.")

bys = defaultdict(dict)
for r in B:
    bys[r["set"]][r["bin"]] = r
pooled = {s_: bins.pop("pooled") for s_, bins in bys.items() if "pooled" in bins}
r2b = [0, 0]
for s_, bins in bys.items():
    for k, b in bins.items():
        for j, v in b["judges"].items():
            pv = pooled[s_]["judges"][j]["pearson"]
            if v["pearson"] is not None and pv is not None and np.isfinite(v["pearson"]):
                r2b[0] += 1; r2b[1] += bool(v["pearson"] < pv)
summ["R2_arm_b"] = {"cells": r2b[0], "within_bin_below_pooled": r2b[1]}
out.append(f"\nR2 (Arm B): within-bin Pearson below the judge's pooled Pearson in {r2b[1]} of {r2b[0]} (set, bin, judge) cells.")
out.append("\n## Arm B: real LLM judges per entropy quintile\n")
out.append("| set | binning | judge | S Q1 -> Q5 | skill Q1 -> Q5 | Pearson Q1 -> Q5 | top-1 Q1 -> Q5 |\n|---|---|---|---|---|---|---|")
r3 = []
for s, bins in bys.items():
    q1, q5 = bins[min(bins)], bins[max(bins)]
    judges = list(q1["judges"])
    for j in judges:
        a, b = q1["judges"][j], q5["judges"][j]
        f = lambda x: "nan" if x is None else f"{x:.3f}"
        out.append(f"| {s} | {'same ratings' if q1['same_ratings_binning'] else 'split halves'} | {j} | {f(a['S'])} -> {f(b['S'])} | {f(a['skill'])} -> {f(b['skill'])} | {f(a['pearson'])} -> {f(b['pearson'])} | {f(a['top1'])} -> {f(b['top1'])} |")
    row = {"set": s, "n_judges": len(judges), "top1_falls_all": all(q5["judges"][j]["top1"] < q1["judges"][j]["top1"] for j in judges),
           "kendall_pearson_vs_S_by_bin": [bins[k].get("kendall_pearson_vs_S") for k in sorted(bins)],
           "best_by_pearson_by_bin": [bins[k].get("best_by_pearson") for k in sorted(bins)], "best_by_S_by_bin": [bins[k].get("best_by_S") for k in sorted(bins)]}
    if len(judges) >= 3:
        row["tau_top1_Q1_vs_Q5"] = float(kendalltau([q1["judges"][j]["top1"] for j in judges], [q5["judges"][j]["top1"] for j in judges]).statistic)
        row["tau_S_Q1_vs_Q5"] = float(kendalltau([q1["judges"][j]["S"] for j in judges], [q5["judges"][j]["S"] for j in judges]).statistic)
    r3.append(row)
summ["arm_b"] = r3
summ["R3"] = {"sets": len(r3), "top1_falls_for_every_judge": sum(r["top1_falls_all"] for r in r3)}
out.append("\n| set | top-1 falls Q1 -> Q5 for every judge | Kendall tau Q1 vs Q5 ranking: by top-1 / by S | Kendall tau(Pearson ranking, S ranking) per bin Q1..Q5 | best by Pearson / best by S, per bin |\n|---|---|---|---|---|")
for r in r3:
    t = f"{r['tau_top1_Q1_vs_Q5']:+.2f} / {r['tau_S_Q1_vs_Q5']:+.2f}" if "tau_S_Q1_vs_Q5" in r else "n/a (1 judge)"
    kt = ", ".join("nan" if x is None else f"{x:+.1f}" for x in r["kendall_pearson_vs_S_by_bin"])
    bb = "; ".join(f"{p}/{q}" for p, q in zip(r["best_by_pearson_by_bin"], r["best_by_S_by_bin"]))
    out.append(f"| {r['set']} | {'yes' if r['top1_falls_all'] else 'no'} | {t} | {kt} | {bb} |")
(res / "strata_tables.md").write_text("\n".join(out) + "\n")
(res / "strata_summary.json").write_text(json.dumps(summ, indent=1))
print("\n".join(out))
