#!/usr/bin/env python3
"""Writes ../results/skill_tables.md from ../results/skill_reanalysis.json (deterministic)."""
import json
from pathlib import Path
R = Path(__file__).resolve().parent.parent / "results"
d = json.loads((R / "skill_reanalysis.json").read_text())
pf = lambda p: "<0.001" if p < 0.001 else f"{p:.3g}"
f = lambda x: f"{x['mean_over_seeds']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]"
L = ["# Experiment 53: Experiment 41's heads as skill scores over the human base rate", "",
     "Held-out test items of Experiment 41's preregistered K-sweep (ridge 0.01, at most 35 L-BFGS iterations), five group-level split seeds. "
     "Skill = 1 - nRPS(head)/nRPS(base rate); nRPS = ranked probability score divided by K - 1 (0 to 1, lower is better); base rate = training-split marginal. "
     "Skill: 1 = perfect, 0 = base rate, negative = worse than the base rate. Values are means over the five seeds; brackets are item-clustered percentile bootstrap 95% intervals "
     "(2,000 draws, test items resampled within each seed, paired across heads). Differences are CE-trained head minus Pearson-trained head (positive = CE-trained better). "
     "'raw' = the head's own forecast; 'cal' = the head's scalar mean through its own validation-fitted multinomial-logit map. Holm p across 16 views (bootstrap p; '<0.001' means no bootstrap draw crossed zero).", "",
     "| view | base-rate nRPS | skill CE-trained raw | skill Pearson-trained raw | skill difference raw (CE minus Pearson) | Holm p | skill CE-trained cal | skill Pearson-trained cal | skill difference cal (CE minus Pearson) | Holm p | raw nRPS gap (Pearson minus CE) |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for v, r in d["rows"].items():
    L.append(f"| {v} | {r['base_rate_nrps']:.4f} | {f(r['skill_ce_raw'])} | {f(r['skill_pearson_raw'])} | {f(r['diff_raw_ce_minus_pearson'])} | {pf(r['diff_raw_ce_minus_pearson']['p_holm'])} | "
             f"{f(r['skill_ce_cal'])} | {f(r['skill_pearson_cal'])} | {f(r['diff_cal_ce_minus_pearson'])} | {pf(r['diff_cal_ce_minus_pearson']['p_holm'])} | {r['raw_nrps_gap_pearson_minus_ce']:+.4f} |")
rows = d["rows"].values()
raw = [r["diff_raw_ce_minus_pearson"]["mean_over_seeds"] for r in rows]
cal = [r["diff_cal_ce_minus_pearson"]["mean_over_seeds"] for r in rows]
ceraw = [r["skill_ce_raw"]["mean_over_seeds"] for r in rows]
cecal = [r["skill_ce_cal"]["mean_over_seeds"] for r in rows]
praw = [r["skill_pearson_raw"]["mean_over_seeds"] for r in rows]
mse = [r["diff_cal_ce_minus_mse"]["mean_over_seeds"] for r in rows]
L += ["", "## Summary", "",
      f"- Raw skill difference (CE-trained minus Pearson-trained): {min(raw):+.4f} to {max(raw):+.4f}; positive on {sum(x > 0 for x in raw)}/16 views.",
      f"- Pearson-trained raw skill: {min(praw):+.4f} to {max(praw):+.4f}; below zero (worse than the base rate) on {sum(x < 0 for x in praw)}/16 views.",
      f"- CE-trained raw skill: {min(ceraw):+.4f} to {max(ceraw):+.4f}; CE-trained calibrated skill: {min(cecal):+.4f} to {max(cecal):+.4f}.",
      f"- Calibrated skill difference (CE minus Pearson): {min(cal):+.4f} to {max(cal):+.4f}; |value| <= 0.01 on {sum(abs(x) <= 0.01 for x in cal)}/16.",
      f"- Calibrated skill difference (CE minus MSE): {min(mse):+.4f} to {max(mse):+.4f}.",
      f"- Sanity check: {d['sanity_check']}; max |deviation| by head: " + ", ".join(f"{k} {v:.1e}" for k, v in d["sanity_max_abs_deviation"].items()) + ".",
      "", "## Preregistered predictions", "",
      f"- P1 (raw difference > 0, Holm p < 0.05 where Exp 41 raw gap > 0.01 nRPS): holds on {len(d['predictions']['P1_holds_on'])}/{len(d['predictions']['P1_raw_gap_views'])} such views.",
      f"- P2 (|calibrated difference| <= 0.025): holds on {len(d['predictions']['P2_holds_on'])}/16.",
      f"- P3 (descriptive; CE-trained raw skill < 0.10): {len(d['predictions']['P3_ce_raw_skill_below_0.10_on'])}/16."]
(R / "skill_tables.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
