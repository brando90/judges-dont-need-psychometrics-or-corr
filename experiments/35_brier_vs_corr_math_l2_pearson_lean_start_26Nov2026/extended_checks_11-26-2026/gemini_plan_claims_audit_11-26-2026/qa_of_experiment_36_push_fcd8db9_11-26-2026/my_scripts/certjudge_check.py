#!/usr/bin/env python3
"""Recompute the CertJudge numbers that Experiment 36 (commit fcd8db9) quotes, from the four pinned source files.

Claude, 11-26-2026. Read-only. Experiment 36 does not recompute these panels; its loader `load_certjudge_real_panels`
reads summaries that cert-judge Experiments 67 and 68 computed (their own calibration protocols, splits and bootstraps).
This script checks that the four pinned files have the SHA-256 values Experiment 36 states, that the numbers quoted in its
SANMI_PITCH.md bottom line and ready-to-send note are in those files, and prints the context the quote leaves out:
the constant-forecast reference and the size of each difference.

Fetch the four files first (commit f6047849853e65fd533a6e5df2416d6a9d9447d6 of brando90/cert-judge; a shallow
`git fetch --depth 1 origin f6047849853e65fd533a6e5df2416d6a9d9447d6` followed by `git show FETCH_HEAD:<path>` works):
    experiments/68_proper_scoring_expansion_e39_e63_and_vb_ratings_start_24Nov2026/results/study_a_e63_proper_scores.json
    experiments/68_proper_scoring_expansion_e39_e63_and_vb_ratings_start_24Nov2026/results/study_b_e39_proper_scores.json
    experiments/67_new_vb_team_ratings_rps_ce_tvd_rerun_of_the_certjudge_panel/results/interim_11-24-2026/judged_B3_calibrated.csv
    experiments/67_new_vb_team_ratings_rps_ce_tvd_rerun_of_the_certjudge_panel/results/interim_11-24-2026/judged_summary.json
and put the four files, under their base names, in one directory DIR.

Usage:
    python3 -I certjudge_check.py DIR
Exit status is nonzero when a check fails.
"""
import csv
import hashlib
import json
import math
import statistics as st
import sys

d = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "."
SHAS = {
    "study_a_e63_proper_scores.json": "24dcaaebd280d1d2c6e356be2ebadee8c28d376721350758bed70e56e7ec1655",
    "study_b_e39_proper_scores.json": "32056ae51d646f2232923c7c5fb12015d7bcdf5cd2dd3faecf880bb0bcac46fb",
    "judged_B3_calibrated.csv": "fac4f0c4cc1ca7459a75b4cb2362b13fafc30828e7e63c9133cd7853ed73b30f",
    "judged_summary.json": "5e4a75b6e1b56e2adb86ae59ac67ec24bbe7f8e61633b2807e99e0f2b774839b",
}
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (("  " + detail) if detail else ""))
    if not ok:
        fails += 1


for fn, sha in SHAS.items():
    check(f"{fn}: SHA-256 equals the value Experiment 36 pins", hashlib.sha256(open(f"{d}/{fn}", "rb").read()).hexdigest() == sha)
e63 = json.load(open(f"{d}/study_a_e63_proper_scores.json"))
e39 = json.load(open(f"{d}/study_b_e39_proper_scores.json"))
rows = list(csv.DictReader(open(f"{d}/judged_B3_calibrated.csv", newline="")))

# --- Study A (Experiment 63, Qwen3-8B arms), target all5 ---
tb = e63["targets"]["all5"]
arms = {k: v["ensemble"] for k, v in tb["arms"].items()}
cst = tb["const_trainval_baseline"]
print(f"FACT  E63 all5 held-out set: {cst['n_items']} items from {cst['n_tasks']} tasks; constant train+val forecast RPS {cst['rps']['val']:.3f} [{cst['rps']['ci_lo']:.3f}, {cst['rps']['ci_hi']:.3f}], CE {cst['ce']['val']:.3f}")
g = arms["rl_grpo"]
check("E63 rl_grpo Spearman is 0.543", abs(g["spearman"]["val"] - 0.543) < 5e-4, f"{g['spearman']['val']:.3f} [{g['spearman']['ci_lo']:.3f}, {g['spearman']['ci_hi']:.3f}]")
check("E63 rl_grpo CE is 5.556", abs(g["ce"]["val"] - 5.556) < 5e-4, f"{g['ce']['val']:.3f} [{g['ce']['ci_lo']:.3f}, {g['ce']['ci_hi']:.3f}]")
check("E63 shuffled_control CE is 1.831", abs(arms["shuffled_control"]["ce"]["val"] - 1.831) < 5e-4, f"{arms['shuffled_control']['ce']['val']:.3f}")
# Every arm against the constant forecast (the quoted note compares rl_grpo only with the shuffled control).
for an, av in tb["arms"].items():
    ens = av["ensemble"]
    ce_seeds = [s["ce"]["val"] for s in av["per_seed"]]
    print(f"FACT  E63 arm {an}: {av['n_seeds']} seed(s); per-seed CE {min(ce_seeds):.3f} to {max(ce_seeds):.3f}; "
          f"ensemble CE {ens['ce']['val']:.3f} [{ens['ce']['ci_lo']:.3f}, {ens['ce']['ci_hi']:.3f}], RPS {ens['rps']['val']:.3f}, "
          f"Spearman {ens['spearman']['val']:.3f}; ensemble CE above the constant forecast's {cst['ce']['val']:.3f}: {ens['ce']['val'] > cst['ce']['val']}")
h = arms["human_mix_sft"]
check("E63 human_mix_sft CE 1.620, RPS 0.142, Spearman 0.746", abs(h["ce"]["val"] - 1.620) < 5e-4 and abs(h["rps"]["val"] - 0.142) < 5e-4 and abs(h["spearman"]["val"] - 0.746) < 5e-4)
print(f"FACT  E63 rl_grpo RPS {g['rps']['val']:.3f} [{g['rps']['ci_lo']:.3f}, {g['rps']['ci_hi']:.3f}] against the constant forecast's {cst['rps']['val']:.3f}; seeds {tb['arms']['rl_grpo'].get('n_seeds')}")
order_sp = sorted(arms, key=lambda a: -arms[a]["spearman"]["val"])
order_ce = sorted(arms, key=lambda a: arms[a]["ce"]["val"])
print("FACT  E63 arms from best to worst Spearman: " + ", ".join(order_sp))
print("FACT  E63 arms from best to worst CE:       " + ", ".join(order_ce))

# --- Study B (Experiment 39): CE-calibrated against Pearson-calibrated over judge configs ---
units = e39["units"]
check("E39: 56 LLM configs, 18 in the test split", e39["n_all_llm"] == 56 and e39["n_test_llm"] == 18)
print(f"FACT  E39 held-out items: {e39['human_loro_heldout_all5']['n_items']}; constant forecast RPS {e39['const_train_baseline_heldout_all5']['rps']:.4f}, CE {e39['const_train_baseline_heldout_all5']['ce']:.4f}")


def binom_two_sided(k, n):
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * p)


for label, sel in (("all 56 LLM configs", [u for u in units if u["kind"] == "llm"]), ("18 test-split LLM configs", [u for u in units if u["kind"] == "llm" and u["split"] == "test"])):
    for met in ("rps", "ce"):
        dd = [u["metrics"][f"pe_cal_all5_{met}"] - u["metrics"][f"ce_cal_all5_{met}"] for u in sel]
        wins = sum(1 for x in dd if x > 0)
        print(f"FACT  E39 {label}, {met.upper()}: CE-calibrated better in {wins} of {len(sel)}; mean of (Pearson-calibrated minus CE-calibrated) {st.mean(dd):+.4f} (sd {st.pstdev(dd):.4f}); a sign test would give p {binom_two_sided(wins, len(sel)):.2g} but the configs are scored on the same items, so they are not independent")
all_l = [u for u in units if u["kind"] == "llm"]
check("E39: CE-calibrated beats Pearson-calibrated on RPS in 40 of 56 configs", sum(u["metrics"]["ce_cal_all5_rps"] < u["metrics"]["pe_cal_all5_rps"] for u in all_l) == 40)
check("E39: CE-calibrated beats Pearson-calibrated on CE in 48 of 56 configs", sum(u["metrics"]["ce_cal_all5_ce"] < u["metrics"]["pe_cal_all5_ce"] for u in all_l) == 48)

# --- Experiment 67 (100 native files, 25 tasks) ---
new_diff = [r for r in rows if r["set"] == "new" and r["comparison"] == "CE-fit minus Pearson-fit"]
new_lev = {r["family"]: r for r in rows if r["set"] == "new" and r["comparison"] == "levels"}
check("Exp 67: three judge families", len(new_diff) == 3)
check("Exp 67: every family's CE-fit minus Pearson-fit RPS interval includes 0", all(float(r["dRPS_lo"]) < 0 < float(r["dRPS_hi"]) for r in new_diff))
for r in new_diff:
    lv = new_lev[r["family"]]
    print(f"FACT  Exp 67 {r['family']} ({r['n_tasks']} tasks): constant RPS {float(lv['CONST_RPS']):.4f}, CE {float(lv['CONST_CE']):.4f} | CE-fit RPS {float(lv['CE-fit_RPS']):.4f}, CE {float(lv['CE-fit_CE']):.4f} | Pearson-fit RPS {float(lv['Pearson-fit_RPS']):.4f}, CE {float(lv['Pearson-fit_CE']):.4f} | CE-fit minus Pearson-fit dRPS {float(r['dRPS']):+.4f} [{float(r['dRPS_lo']):+.4f}, {float(r['dRPS_hi']):+.4f}], dCE {float(r['dCE']):+.4f} [{float(r['dCE_lo']):+.4f}, {float(r['dCE_hi']):+.4f}]")
print("\n" + ("ALL CHECKS PASS" if fails == 0 else f"{fails} CHECK(S) FAILED"))
sys.exit(1 if fails else 0)
