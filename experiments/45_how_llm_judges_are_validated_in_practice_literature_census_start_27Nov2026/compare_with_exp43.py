#!/usr/bin/env python3
"""Cross-census check: Experiment 45 census.csv against Experiment 43 papers.py.

Experiment 43 (experiments/43_llm_judge_validation_practice_survey_start_27Nov2026) is a parallel,
single-coder literature check of the same question that was already on main when this experiment was
merged. It coded each paper for every metric family it reports. This script joins the two censuses on
the arXiv identifier and reports (1) how much the samples overlap and (2) for papers coded in both,
how often the two independent codings agree on whether a metric family is reported anywhere.

Mapping of the families (the definitions are close but not identical):
  Experiment 43 C  <->  Experiment 45 corr_any
  Experiment 43 A  <->  Experiment 45 acc_any or f1_any   (Experiment 43 A includes precision, recall and F1)
  Experiment 43 K  <->  Experiment 45 kappa_any
  Experiment 43 P or Ps <-> Experiment 45 dist_any         (Experiment 45 dist_any includes a system-pair Brier score)

Read-only on Experiment 43. Deterministic: no randomness, no timestamps.
Usage: python3 compare_with_exp43.py
"""
import csv
import hashlib
import importlib.util
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OTHER = HERE.parent / "43_llm_judge_validation_practice_survey_start_27Nov2026" / "papers.py"
BASE = ["L1", "L2", "L3", "L4", "L5"]
ARXIV = re.compile(r"(\d{4}\.\d{4,5})")


def load_exp43():
    spec = importlib.util.spec_from_file_location("exp43_papers", OTHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.R


def main():
    out = []
    P = out.append
    R43 = load_exp43()
    with open(HERE / "census.csv", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["id"])
    digest = hashlib.md5(OTHER.read_bytes()).hexdigest()[:10]
    P(f"CROSS-CENSUS CHECK: Experiment 45 census.csv versus Experiment 43 papers.py (md5 prefix {digest})")

    n43 = len(R43)
    codes43 = [set(r[5].split(",")) for r in R43]
    P(f"Experiment 43: {n43} papers coded; C {sum('C' in c for c in codes43)}/{n43}; A {sum('A' in c for c in codes43)}/{n43}; "
      f"K {sum('K' in c for c in codes43)}/{n43}; P {sum('P' in c for c in codes43)}/{n43}; Ps {sum('Ps' in c for c in codes43)}/{n43}; "
      f"no item-level P {sum('P' not in c for c in codes43)}/{n43}")
    grp = {}
    for r, c in zip(R43, codes43):
        grp.setdefault(r[3].split("+")[0], []).append(c)
    P("Experiment 43 groups: " + "; ".join(f"{g} n={len(cs)}, C {sum('C' in c for c in cs)}, A {sum('A' in c for c in cs)}, K {sum('K' in c for c in cs)}, P {sum('P' in c for c in cs)}" for g, cs in sorted(grp.items())))
    P("")

    by43 = {}
    for r, c in zip(R43, codes43):
        if re.fullmatch(r"\d{4}\.\d{4,5}", r[6]):
            by43[r[6]] = (r, c)
    P(f"Experiment 43 papers with an arXiv identifier: {len(by43)} of {n43}")

    def arx(row):
        m = ARXIV.search(row["source_url"])
        return m.group(1) if m else None

    base = [r for r in rows if r["lineage"] in BASE and r["human_validation"] == "Y" and r["coverage"] in ("full", "partial")]
    for label, scope in (("D_all (the base-rate denominator, n=%d)" % len(base), base), ("all census rows with an arXiv identifier (n=%d)" % sum(1 for r in rows if arx(r)), [r for r in rows if arx(r)])):
        pairs = [(r, by43[arx(r)]) for r in scope if arx(r) in by43]
        P(f"\nSCOPE: {label}")
        P(f"overlap with Experiment 43: {len(pairs)} papers; share of this scope also in Experiment 43: {len(pairs)}/{len(scope)} = {len(pairs) / len(scope):.3f}; share of Experiment 43 also in this scope: {len(pairs)}/{n43} = {len(pairs) / n43:.3f}")
        checks = [
            ("correlation reported (43 C, 45 corr_any)", lambda r, c: ("C" in c, r["corr_any"] == "Y")),
            ("raw agreement, accuracy or F1 reported (43 A, 45 acc_any or f1_any)", lambda r, c: ("A" in c, r["acc_any"] == "Y" or r["f1_any"] == "Y")),
            ("kappa-family reported (43 K, 45 kappa_any)", lambda r, c: ("K" in c, r["kappa_any"] == "Y")),
            ("proper score or divergence against humans (43 P or Ps, 45 dist_any)", lambda r, c: (("P" in c) or ("Ps" in c), r["dist_any"] == "Y")),
        ]
        for name, fn in checks:
            agree = 0
            dis = []
            for r, (r43, c) in pairs:
                a, b = fn(r, c)
                if a == b:
                    agree += 1
                else:
                    dis.append(f"{r['id']} ({r43[0][:38]}): 43 says {'yes' if a else 'no'}, 45 says {'yes' if b else 'no'}")
            P(f"  {name}: agreement {agree}/{len(pairs)}" + (f" = {agree / len(pairs):.3f}" if pairs else ""))
            for d in dis:
                P(f"     disagreement: {d}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
