"""Experiment 48 summary: tables of reopened gaps from results/arm_a_llm_judges.json and results/arm_b_exp41_heads.json.
Usage: python summarize_shift.py RESULTS_DIR   (writes RESULTS_DIR/shift_tables.md and shift_summary.json)"""
import json, sys
from collections import defaultdict
from pathlib import Path

res = Path(sys.argv[1])
a = json.loads((res / "arm_a_llm_judges.json").read_text())
b = json.loads((res / "arm_b_exp41_heads.json").read_text())
SHORT = {"chaosnli_snli": "SNLI", "chaosnli_mnli": "MNLI", "dices350": "DICES-350", "dices990": "DICES-990",
         "pooled_other_three": "other three", "summeval_relevance": "SummEval relevance", "summeval_consistency": "SummEval consistency"}
out, summ = [], {"arm_a_by_shift": {}, "arm_a_by_pair": [], "arm_b": b["table"]}

out.append("## Arm A: frozen LLM judges, primary score\n")
out.append("| shift type | pairs (judge x direction) | gap reopens | calibration fails outright (worse than raw) |\n|---|---|---|---|")
for sh in ["near", "far", "leave_one_set_out", "rubric"]:
    rs = [r for r in a if r["shift"] == sh]
    d = {"n": len(rs), "reopens": sum(r["reopens"] for r in rs), "fails": sum(r["calibration_fails_outright"] for r in rs)}
    summ["arm_a_by_shift"][sh] = d
    out.append(f"| {sh} | {d['n']} | {d['reopens']} | {d['fails']} |")

out.append("\nPer source -> target (over judges): primary score S, gain retained F = (S_raw - S_shift)/(S_raw - S_in), range over judges.\n")
out.append("| shift | source -> target | score | judges | F range | D = S_shift - S_in range | reopens | fails |\n|---|---|---|---|---|---|---|---|")
grp = defaultdict(list)
for r in a:
    grp[(r["shift"], r["source"], r["target"])].append(r)
for (sh, s, t), rs in sorted(grp.items(), key=lambda kv: ["near", "far", "leave_one_set_out", "rubric"].index(kv[0][0])):
    p = rs[0]["primary"]
    F = [r[p]["F_gain_retained"] for r in rs]; D = [r[p]["D_shift_minus_in"] for r in rs]
    row = {"shift": sh, "source": s, "target": t, "score": p, "n_judges": len(rs), "F_min": min(F), "F_max": max(F),
           "D_min": min(D), "D_max": max(D), "reopens": sum(r["reopens"] for r in rs), "fails": sum(r["calibration_fails_outright"] for r in rs)}
    summ["arm_a_by_pair"].append(row)
    out.append(f"| {sh} | {SHORT.get(s, s)} -> {SHORT.get(t, t)} | {p} | {len(rs)} | {min(F):.2f} to {max(F):.2f} | "
               f"{min(D):+.4f} to {max(D):+.4f} | {row['reopens']} | {row['fails']} |")

out.append("\n## Arm B: Experiment 41 linear heads, G = nRPS(Pearson head + map) - nRPS(CE head, raw)\n")
out.append("| arm | source -> target | in-domain G (seed mean) | shifted G (seed mean) | shifted G seed range | seeds with CI > 0 | reopens |\n|---|---|---|---|---|---|---|")
for r in b["table"]:
    lo, hi = r["G_shift_seed_range"]
    out.append(f"| {r['arm']} | {r['source']} -> {r['target']} | {r['G_in_seed_mean']:+.4f} | {r['G_shift_seed_mean']:+.4f} | "
               f"{lo:+.4f} to {hi:+.4f} | {r['seeds_G_shift_ci_above_0']}/5 | {r['reopens']} |")
arms = defaultdict(lambda: [0, 0])
for r in b["table"]:
    arms[r["arm"]][0] += 1; arms[r["arm"]][1] += bool(r["reopens"])
summ["arm_b_counts"] = {k: {"views": v[0], "reopens": v[1]} for k, v in arms.items()}
out.append("\n" + "; ".join(f"{k}: {v[1]} of {v[0]} reopen" for k, v in arms.items()))
(res / "shift_tables.md").write_text("\n".join(out) + "\n")
(res / "shift_summary.json").write_text(json.dumps(summ, indent=1))
print("\n".join(out))
