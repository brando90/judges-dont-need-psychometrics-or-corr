#!/usr/bin/env python3
"""Tally the Opus 5.5 max review's paired comparisons from its saved JSON files (standard library only).

Reads agent_outputs/fair_*.json (written by agent_scripts/fairness.py on 11-26-2026) and prints, for each of the five
real rating sets, the paired differences between a CE-trained head and Pearson-trained heads whose L2 penalty was
chosen on the validation split. It adds nothing new to the data: it only counts which saved intervals exclude 0.

Conventions copied from fairness.py (checked by reading `paired()` and the loop that builds the keys):
  key "B vs A" means  B = the Pearson-trained head, A = the cross-entropy (CE)-trained head.
  loss_b_minus_a = held-out test loss of the Pearson-trained head minus that of the CE-trained head, same metric,
                   same items; POSITIVE means the CE-trained head is better (lower loss).
  r_a_minus_r_b  = held-out test Pearson correlation of the CE-trained head minus that of the Pearson-trained head;
                   POSITIVE means the CE-trained head correlates more with the human ratings.
  Intervals are 95% percentile intervals of an item-clustered paired bootstrap, B = 1000 (the reviewer's setting).
  A difference is "significant" here only when its saved interval excludes 0; there is no multiplicity correction.

Which comparisons are counted:
  the four validation-selected Pearson variants against the CE head selected on validation normalized RPS:
    psamp_sel_own, psamp_sel_rps (Pearson head trained on the correlation of a SAMPLED rating, penalty chosen on
    validation correlation or on validation normalized RPS), pmean_sel_own, pmean_sel_rps (Pearson head trained on
    the correlation of the EXPECTED score, same two selection rules).
  The committed Experiment 36 heads (exp36_pearson vs exp36_ce) are listed separately, because they use no validation.

Usage: python3 -I tally_fair_json.py DIR_WITH_fair_JSON_FILES
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

FILES = [
    ("summeval_expert", "fair_summeval_expert.json"),
    ("summeval_expert_binary", "fair_seb.json"),
    ("chaosnli_snli_mnli", "fair_chaos.json"),
    ("dices990", "fair_dices990.json"),
    ("dices990_binary_yes", "fair_db.json"),
]
SELECTED = ["psamp_sel_own", "psamp_sel_rps", "pmean_sel_own", "pmean_sel_rps"]


def sign_of(lo: float, hi: float) -> str:
    if math.isnan(lo) or math.isnan(hi):
        return "undefined"
    if lo > 0:
        return "CE higher"
    if hi < 0:
        return "Pearson higher"
    return "no difference"


def main(argv: list[str]) -> int:
    base = Path(argv[1])
    tally = {"CE higher": 0, "Pearson higher": 0, "no difference": 0, "undefined": 0}
    points_defined = 0
    loss_tally = {"CE better": 0, "Pearson better": 0, "no difference": 0}
    print("held-out test Pearson correlation, CE-trained head minus Pearson-trained head (positive = CE-trained head higher)")
    print("dataset | Pearson variant (penalty chosen on validation) | difference [95% paired item-clustered interval] | reading")
    for name, fname in FILES:
        d = json.loads((base / fname).read_text(), parse_constant=lambda c: float("nan") if c == "NaN" else float(c))
        assert d["dataset"] == name, (d["dataset"], name)
        for sel in SELECTED:
            key = f"{sel} vs ce_sel_rps"
            rec = d["paired"][key]["rps_norm"]
            lo, hi = rec["ci_r"]
            point = rec["r_a_minus_r_b"]
            reading = sign_of(lo, hi)
            tally[reading] += 1
            points_defined += 0 if math.isnan(point) else 1
            pt = "undefined (a constant prediction has no correlation)" if math.isnan(point) else f"{point:+.4f}"
            print(f"{name} | {sel} | {pt} [{lo:+.4f}, {hi:+.4f}] | {reading}")
    n = sum(tally.values())
    print()
    print(f"correlation comparisons counted: {n} (5 sets x 4 validation-selected Pearson variants)")
    for k, v in tally.items():
        print(f"  {k}: {v}")
    print(f"  intervals defined: {n - tally['undefined']}; point estimates defined: {points_defined} (a head with constant predictions has no correlation, so its point estimate is NaN while the saved interval, from the bootstrap draws where the correlation was defined, is reported)")
    print()
    print("held-out test normalized RPS, Pearson-trained head minus CE-trained head, same items (positive = CE-trained head better)")
    for name, fname in FILES:
        d = json.loads((base / fname).read_text(), parse_constant=lambda c: float("nan") if c == "NaN" else float(c))
        for sel in SELECTED:
            rec = d["paired"][f"{sel} vs ce_sel_rps"]["rps_norm"]
            lo, hi = rec["ci_loss"]
            if lo > 0:
                loss_tally["CE better"] += 1
            elif hi < 0:
                loss_tally["Pearson better"] += 1
            else:
                loss_tally["no difference"] += 1
    for k, v in loss_tally.items():
        print(f"  {k}: {v} of {sum(loss_tally.values())}")
    print()
    print("committed Experiment 36 heads (no validation selection), correlation difference, CE-trained head minus Pearson-trained head")
    for name, fname in FILES:
        d = json.loads((base / fname).read_text(), parse_constant=lambda c: float("nan") if c == "NaN" else float(c))
        rec = d["paired"]["exp36_pearson vs exp36_ce"]["rps_norm"]
        lo, hi = rec["ci_r"]
        print(f"  {name}: {rec['r_a_minus_r_b']:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {sign_of(lo, hi)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
