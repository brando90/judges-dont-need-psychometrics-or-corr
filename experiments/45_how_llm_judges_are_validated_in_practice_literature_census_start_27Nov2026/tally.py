#!/usr/bin/env python3
"""Experiment 45 tally: how LLM judges are validated against humans in published sources.

Reads census.csv in this folder, applies the decision rules V1 to V4 declared in
PROTOCOL_AND_PREDICTIONS_11-27-2026.md, and prints every number used in the report.
Deterministic: bootstrap seed 0, verification-sample seed 0, no timestamps, so two
runs print identical text.

Usage:
  python3 tally.py                   print the tally
  python3 tally.py --verify-sample   print the seeded verification sample
  python3 tally.py --figure          print the tally and write fig_headline_share_by_lineage.png

Every proportion describes this purposive sample of sources. It is not an estimate
for the field. Wilson intervals are descriptive of the sample.
"""
import csv
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = ["L1", "L2", "L3", "L4", "L5"]
CORR = {"CORR_INST", "CORR_SYS", "CORR_UNSTATED"}
FAMILIES = ["CORR_INST", "CORR_SYS", "CORR_UNSTATED", "ACC_AGREE", "KAPPA_ALPHA_ICC", "F1_PRF", "DIST", "OTHER", "NA"]
MARK = re.compile(r"MIXED_HEADLINE|ALT_HEADLINE_CORR")
B_DRAWS = 2000


def load():
    with open(HERE / "census.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r["id"])
    return rows


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def share(k, n):
    if n == 0:
        return "0/0 (no rows)"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.3f} (Wilson 95% {lo:.3f} to {hi:.3f})"


def in_base(r, allow_abstract=False):
    cov_ok = r["coverage"] in ("full", "partial") or (allow_abstract and r["coverage"] == "abstract_only")
    return r["human_validation"] == "Y" and r["era"] == "llm" and r["lineage"] in BASE and cov_ok


def balanced(groups, pred):
    """Equal weight per lineage, stratified bootstrap over sources within lineage (seed 0)."""
    rng = np.random.default_rng(0)
    lins = [l for l in BASE if len(groups.get(l, [])) > 0]
    flags = {l: np.array([1.0 if pred(r) else 0.0 for r in groups[l]]) for l in lins}
    point = float(np.mean([flags[l].mean() for l in lins]))
    draws = np.empty(B_DRAWS)
    for b in range(B_DRAWS):
        draws[b] = np.mean([rng.choice(flags[l], size=len(flags[l]), replace=True).mean() for l in lins])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def group_by_lineage(rows):
    g = {}
    for r in rows:
        g.setdefault(r["lineage"], []).append(r)
    return g


def verdict_v2(k, n):
    lo, _ = wilson(k, n)
    p = k / n
    if p >= 0.70 and lo >= 0.55:
        return "standard for LLM-judge validation in general"
    if p >= 0.40:
        extra = " (share at or above 70 percent but Wilson lower bound below 55 percent)" if p >= 0.70 else ""
        return "one of several common rulers" + extra
    return "not the standard ruler"


def tally(rows, out, make_fig):
    P = out.append
    P("EXPERIMENT 45 TALLY (descriptive of this purposive sample of published sources)")
    P(f"rows in census.csv: {len(rows)}")
    P("rows by lineage: " + ", ".join(f"{l}={c}" for l, c in sorted(Counter(r['lineage'] for r in rows).items())))
    P("rows by coverage: " + ", ".join(f"{k}={c}" for k, c in sorted(Counter(r['coverage'] for r in rows).items())))
    P("rows with human_validation=N (listed, excluded): " + (", ".join(r["id"] for r in rows if r["human_validation"] == "N") or "none"))
    P("")

    D = [r for r in rows if in_base(r)]
    Dabs = [r for r in rows if in_base(r, allow_abstract=True)]
    g = group_by_lineage(D)
    P("1. DENOMINATOR D_all (human_validation=Y, era=llm, lineages L1 to L5, coverage full or partial)")
    P(f"D_all size: {len(D)}; by lineage: " + ", ".join(f"{l}={len(g.get(l, []))}" for l in BASE))
    P("D_all by label_type: " + ", ".join(f"{k}={c}" for k, c in sorted(Counter(r['label_type'] for r in D).items())))
    P("abstract-only rows with human_validation=Y (added only in sensitivity A): " + (", ".join(r["id"] for r in Dabs if r["coverage"] == "abstract_only") or "none"))
    P("D_all by coverage: " + ", ".join(f"{k}={c}" for k, c in sorted(Counter(r['coverage'] for r in D).items())))
    reads = [r for r in rows if r["lineage"] in BASE and r["coverage"] in ("full", "partial")]
    P(f"minimum for a verdict: at least 40 sources coded from a full or partial read among L1 to L5 (any human_validation); observed {len(reads)} -> {'MET' if len(reads) >= 40 else 'NOT MET: report as underpowered and give no verdict'}")
    P("")

    P("2. HEADLINE FAMILY BY LINEAGE (family of the first agreement number in the abstract, else first main results table)")
    for l in BASE + ["pooled"]:
        sub = D if l == "pooled" else g.get(l, [])
        c = Counter(r["headline_family"] for r in sub)
        P(f"{l:7s} n={len(sub):2d}  " + "  ".join(f"{f}={c.get(f, 0)}" for f in FAMILIES if c.get(f, 0)))
    P("")

    P("3. SHARE WITH A CORRELATION-FAMILY HEADLINE (CORR_INST, CORR_SYS, CORR_UNSTATED)")
    res3 = {}
    for l in BASE + ["pooled"]:
        sub = D if l == "pooled" else g.get(l, [])
        k = sum(1 for r in sub if r["headline_family"] in CORR)
        res3[l] = (k, len(sub))
        P(f"{l:7s} headline correlation: {share(k, len(sub))}")
    pt, lo, hi = balanced(g, lambda r: r["headline_family"] in CORR)
    P(f"lineage-balanced (equal weight L1 to L5, stratified bootstrap seed 0, {B_DRAWS} draws): {pt:.3f} (95% percentile {lo:.3f} to {hi:.3f})")
    res3["balanced"] = (pt, lo, hi)
    P("")

    P("4. ANY CORRELATION REPORTED (corr_any=Y), AND OTHER METRIC FAMILIES REPORTED ANYWHERE IN THE SOURCE")
    for flag in ["corr_any", "acc_any", "kappa_any", "f1_any", "dist_any"]:
        parts = []
        for l in BASE + ["pooled"]:
            sub = D if l == "pooled" else g.get(l, [])
            k = sum(1 for r in sub if r[flag] == "Y")
            parts.append(f"{l} {k}/{len(sub)}")
        P(f"{flag:9s} " + "  ".join(parts))
    ptc, loc, hic = balanced(g, lambda r: r["corr_any"] == "Y")
    P(f"corr_any lineage-balanced: {ptc:.3f} (95% percentile {loc:.3f} to {hic:.3f})")
    ca = [r for r in D if r["corr_any"] == "Y"]
    P("corr_level among corr_any=Y rows: " + ", ".join(f"{k}={c}" for k, c in sorted(Counter(r['corr_level'] for r in ca).items())))
    P("")

    P("5. BY LABEL TYPE (headline correlation share within D_all)")
    for lt in ["GRADED", "PAIRWISE", "CATEGORICAL", "SYSTEM_RANK", "MIXED"]:
        sub = [r for r in D if r["label_type"] == lt]
        k = sum(1 for r in sub if r["headline_family"] in CORR)
        P(f"{lt:12s} {share(k, len(sub))}")
    P("")

    P("6. DECISION RULES DECLARED BEFORE DATA")
    k1, n1 = res3["L1"]
    lo1, _ = wilson(k1, n1)
    v1 = (k1 / n1 >= 0.80) and (lo1 >= 0.65)
    P(f"V1 graded-score ruler: L1 correlation headline {share(k1, n1)}; rule needs share >= 0.80 and Wilson lower >= 0.65 -> {'HOLDS' if v1 else 'DOES NOT HOLD'}")
    non1 = [r["id"] for r in g.get("L1", []) if r["headline_family"] not in CORR]
    P(f"   L1 rows without a correlation-family headline: {', '.join(non1) or 'none'}")
    for kk, lab in ((k1 - 1, "one fewer correlation headline"), (k1 + 1, "one more correlation headline")):
        if 0 <= kk <= n1:
            lo_, _ = wilson(kk, n1)
            P(f"   fragility of V1 to a one-row change, {lab}: {share(kk, n1)} -> {'HOLDS' if (kk / n1 >= 0.80 and lo_ >= 0.65) else 'DOES NOT HOLD'}")
    kp, np_ = res3["pooled"]
    P(f"V2 general ruler: pooled correlation headline {share(kp, np_)} -> {verdict_v2(kp, np_)}")
    S = [r for r in D if r["selection_by_corr"] == "Y" or r["selection_loose"] == "Y" or r["trained_on_human_loss"] == "Y"]
    strict = [r for r in S if r["selection_by_corr"] == "Y"]
    corr_only = [r for r in strict if r["selection_metric"] in ("CORR", "BOTH")]
    P(f"V3 selection objective: sources that select, tune or train on human data (strict selection, loose selection or training on human loss) in D_all: {len(S)}")
    P(f"   pre-declared reading, selection_by_corr=Y (agreement or correlation maximized on human data): {share(len(strict), len(S))} -> {'HOLDS (below 15 percent)' if len(S) and len(strict) / len(S) < 0.15 else 'DOES NOT HOLD (15 percent or more)'}")
    P(f"   refined reading, correlation only (selection_metric CORR or BOTH): {share(len(corr_only), len(S))} -> {'below 15 percent' if len(S) and len(corr_only) / len(S) < 0.15 else '15 percent or more'}")
    P("   V3 grid, share with the stated numerator among the stated denominator (the declared reading is strict any-metric over S):")
    for dn, dd in (("S (select, tune or train on human data)", S), ("all of D_all", D)):
        P(f"     denominator {dn}: strict any-metric {share(len(strict), len(dd))}; strict correlation-only {share(len(corr_only), len(dd))}")
    loo = []
    for r0 in strict:
        S2 = [r for r in S if r["id"] != r0["id"]]
        st2 = [r for r in strict if r["id"] != r0["id"]]
        co2 = [r for r in st2 if r["selection_metric"] in ("CORR", "BOTH")]
        loo.append((len(st2) / len(S2), len(co2) / len(S2)))
    if loo:
        P(f"   leave-one-out over the {len(strict)} strict rows (one strict row dropped from numerator and denominator): any-metric share {min(a for a, _ in loo):.3f} to {max(a for a, _ in loo):.3f}; correlation-only share {min(b for _, b in loo):.3f} to {max(b for _, b in loo):.3f}; correlation-only share is below 0.15 in {sum(1 for _, b in loo if b < 0.15)} of {len(loo)} leave-one-out cases")
    P("   strict selection rows: " + ", ".join(f"{r['id']}[{r['selection_metric']}]" for r in strict))
    P("   loose-only rows (selection informed by human agreement, not stated as maximized): " + ", ".join(r["id"] for r in S if r["selection_by_corr"] != "Y" and r["selection_loose"] == "Y"))
    P("   training-on-human-loss rows: " + (", ".join(r["id"] for r in D if r["trained_on_human_loss"] == "Y") or "none"))
    kd = sum(1 for r in D if r["dist_any"] == "Y")
    P(f"V4 distributional: dist_any=Y {share(kd, len(D))} -> {'HOLDS (below 10 percent): proper or distributional scoring against humans is rare' if kd / len(D) < 0.10 else 'DOES NOT HOLD (10 percent or more)'}")
    P("   dist_any=Y rows in D_all: " + (", ".join(r["id"] for r in D if r["dist_any"] == "Y") or "none"))
    P("")

    P("7. SENSITIVITY ANALYSES (declared in the protocol, plus the added co-headline reading)")
    gA = group_by_lineage(Dabs)
    kA = sum(1 for r in Dabs if r["headline_family"] in CORR)
    note_a = " (no abstract-only rows remain among L1 to L5 with human_validation=Y after the coverage upgrades, so this equals the main analysis)" if len(Dabs) == len(D) else ""
    P(f"A. abstract-only rows added: D size {len(Dabs)}; pooled headline correlation {share(kA, len(Dabs))}{note_a}")
    k1A = sum(1 for r in gA.get("L1", []) if r["headline_family"] in CORR)
    P(f"   L1 {share(k1A, len(gA.get('L1', [])))}")
    ptA, loA, hiA = balanced(gA, lambda r: r["headline_family"] in CORR)
    P(f"   lineage-balanced {ptA:.3f} (95% percentile {loA:.3f} to {hiA:.3f})")
    kB = sum(1 for r in D if r["headline_family"] in CORR or r["headline_family"] == "KAPPA_ALPHA_ICC")
    P(f"B. kappa, alpha and ICC headlines counted as correlation-like (upper bound; the closed vocabulary cannot separate ICC from kappa): pooled {share(kB, len(D))}")
    P("   KAPPA_ALPHA_ICC headline rows: " + ", ".join(r["id"] for r in D if r["headline_family"] == "KAPPA_ALPHA_ICC"))
    kC = sum(1 for r in D if r["headline_family"] in CORR or MARK.search(r["notes"]))
    P(f"C. co-headline reading (notes marked MIXED_HEADLINE or ALT_HEADLINE_CORR counted as correlation): pooled {share(kC, len(D))}")
    P("   marked rows: " + (", ".join(r["id"] for r in D if MARK.search(r["notes"]) and r["headline_family"] not in CORR) or "none"))
    k1C = sum(1 for r in g.get("L1", []) if r["headline_family"] in CORR or MARK.search(r["notes"]))
    P(f"   L1 under the co-headline reading {share(k1C, len(g.get('L1', [])))}")
    P(f"D. any correlation reported anywhere (corr_any=Y), pooled {share(sum(1 for r in D if r['corr_any'] == 'Y'), len(D))}")
    P("")

    P("8. PRE-DECLARED PREDICTIONS VERSUS RESULTS (headline correlation share; any correlation share)")
    pred = {"L1": 0.90, "L2": 0.15, "L3": 0.85, "L4": 0.25, "L5": 0.30, "pooled": 0.50}
    for l in BASE + ["pooled"]:
        k, n = res3[l]
        sub = D if l == "pooled" else g.get(l, [])
        ka = sum(1 for r in sub if r["corr_any"] == "Y")
        P(f"{l:7s} predicted {pred[l]:.2f}  observed {k / n:.3f} ({k}/{n})  any correlation observed {ka / n:.3f} ({ka}/{n})")
    P(f"pooled any-correlation predicted 0.60, observed {sum(1 for r in D if r['corr_any'] == 'Y') / len(D):.3f}")
    P(f"dist_any predicted 0.03, observed {kd / len(D):.3f}; strict selection_by_corr=Y predicted at most 2 sources, observed {len(strict)}")
    P("")

    def why_excluded(r):
        if r["lineage"] == "P":
            return "pre-LLM context row"
        if r["lineage"] == "L6":
            return "L6 critique or alternative"
        if r["human_validation"] == "N":
            return "human_validation=N"
        if r["coverage"] == "abstract_only":
            return "abstract_only"
        return "other"

    P("9. ROWS EXCLUDED FROM BASE RATES (descriptive only)")
    for r in rows:
        if not in_base(r):
            P(f"{r['id']:6s} {why_excluded(r):28s} headline={r['headline_family']:16s} corr_any={r['corr_any']} dist_any={r['dist_any']} trained_on_human_loss={r['trained_on_human_loss']} coverage={r['coverage']}")
    P("")

    Prow = [r for r in rows if r["lineage"] == "P"]
    cP = Counter(r["headline_family"] for r in Prow)
    P("10. PRE-LLM CONTEXT ROWS (lineage P; in no base rate)")
    P(f"P rows n={len(Prow)}; headline families: " + ", ".join(f"{f}={cP[f]}" for f in FAMILIES if cP.get(f, 0))
      + f"; any correlation {sum(1 for r in Prow if r['corr_any'] == 'Y')}/{len(Prow)}; strict selection rows: " + (", ".join(f"{r['id']}[{r['selection_metric']}]" for r in Prow if r["selection_by_corr"] == "Y") or "none"))
    L6 = [r for r in rows if r["lineage"] == "L6"]
    P(f"L6 critique rows n={len(L6)}; with human_validation=Y {sum(1 for r in L6 if r['human_validation'] == 'Y')}; any correlation {sum(1 for r in L6 if r['corr_any'] == 'Y')}; dist_any=Y {sum(1 for r in L6 if r['dist_any'] == 'Y')}; trained_on_human_loss=Y {sum(1 for r in L6 if r['trained_on_human_loss'] == 'Y')}")
    P("")

    vpath = HERE / "verification.csv"
    if vpath.exists():
        with open(vpath, newline="") as f:
            V = list(csv.DictReader(f))
        vf = ["headline_family", "corr_any", "corr_level", "acc_any", "kappa_any", "dist_any", "selection_by_corr"]
        main = [v for v in V if v["sample_role"] in ("seeded", "targeted extra")]
        comp = [v for v in main if all(v[k] != "not_compared" for k in vf)]
        void = [v["id"] for v in main if v not in comp]
        P("11. SECOND-READ VERIFICATION (verification.csv; second read with the verification prompt, plus a narrowed re-read wherever the two disagreed)")
        P(f"rows re-read: {len(main)} (seeded {sum(1 for v in main if v['sample_role'] == 'seeded')}, targeted extras {sum(1 for v in main if v['sample_role'] == 'targeted extra')}); rows with usable field-level answers: {len(comp)}; void re-reads excluded: {', '.join(void) or 'none'}")
        tot = len(comp) * len(vf)
        dis = [(v["id"], k) for v in comp for k in vf if v[k] == "disagree"]
        P(f"field-level agreement before resolution: {tot - len(dis)}/{tot} = {(tot - len(dis)) / tot:.3f} over {len(vf)} fields; rows with at least one disagreement: {len({i for i, _ in dis})}/{len(comp)}")
        for k in vf:
            kd = sum(1 for _, kk in dis if kk == k)
            P(f"   {k:18s} agreement {len(comp) - kd}/{len(comp)}")
        for v in comp:
            if any(v[k] == "disagree" for k in vf):
                P(f"   disagreement {v['id']}: fields {', '.join(k for k in vf if v[k] == 'disagree')}; resolution: {v['resolution']}; census changed: {v['census_changed']}")
        ch = [v["id"] for v in V if v["census_changed"] == "Y"]
        P(f"census rows corrected after verification: {', '.join(ch) or 'none'}")
        ext = [v for v in V if v["sample_role"] == "added selection check"]
        strict_ids = {r["id"] for r in rows if r["selection_by_corr"] == "Y" and in_base(r)}
        checked = {v["id"] for v in V if v["selection_by_corr"] != "not_compared"}
        P(f"added selection checks (decided after the first-pass pattern, only the selection field): {', '.join(v['id'] for v in ext)}; all confirmed: {all(v['selection_by_corr'] == 'agree' for v in ext)}")
        P(f"strict selection rows in D_all that were re-read for selection: {len(strict_ids & checked)}/{len(strict_ids)}")
        P("")

    if make_fig:
        make_figure(res3, g, D)
        P("figure written: fig_headline_share_by_lineage.png")


def make_figure(res3, g, D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    labels = ["L1 graded-score judges", "L2 pairwise-preference judges", "L3 benchmark and leaderboard validation",
              "L4 frontier-lab graders", "L5 retrieval and annotation replacement", "Pooled over all sources"]
    keys = BASE + ["pooled"]
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ys = list(range(len(keys)))[::-1]
    color = "#1f5f8b"
    for y, key in zip(ys, keys):
        k, n = res3[key]
        lo, hi = wilson(k, n)
        ax.plot([lo, hi], [y, y], color=color, lw=2.0, solid_capstyle="round", zorder=3)
        ax.plot([k / n], [y], marker="o", ms=9, color=color, zorder=4, ls="none")
        sub = D if key == "pooled" else g.get(key, [])
        ka = sum(1 for r in sub if r["corr_any"] == "Y")
        ax.plot([ka / n], [y], marker="D", ms=8, mfc="white", mec=color, mew=1.6, ls="none", zorder=4)
        ax.text(1.02, y, f"{k}/{n}", va="center", ha="left", fontsize=10, transform=ax.get_yaxis_transform())
    ax.axvline(0.40, color="#8a8a8a", ls="--", lw=1, zorder=1)
    ax.axvline(0.70, color="#8a8a8a", ls="--", lw=1, zorder=1)
    ax.text(0.40, len(keys) - 0.45, "0.40", ha="center", va="bottom", fontsize=9, color="#555555")
    ax.text(0.70, len(keys) - 0.45, "0.70", ha="center", va="bottom", fontsize=9, color="#555555")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_ylim(-0.6, len(keys) - 0.1)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Share of sources in this census (higher means correlation is more common)", fontsize=10)
    ax.text(1.02, len(keys) - 0.45, "headline / sources", va="bottom", ha="left", fontsize=8, color="#555555", transform=ax.get_yaxis_transform())
    ax.set_title("How often correlation with human ratings is the headline ruler for an LLM judge\n"
                 f"Census of {res3['pooled'][1]} published validations with human labels; descriptive of this sample, not of the field", fontsize=11)
    ax.grid(True, alpha=0.35)
    handles = [Line2D([0], [0], marker="o", color=color, lw=2.0, ms=9, label="Headline agreement number is a correlation (dot, Wilson 95% interval)"),
               Line2D([0], [0], marker="D", color=color, mfc="white", mec=color, mew=1.6, lw=0, ms=8, label="Any correlation reported anywhere in the source (hollow diamond)"),
               Line2D([0], [0], color="#8a8a8a", ls="--", lw=1, label="Pooled-verdict thresholds, 0.40 and 0.70 (declared before data collection)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "fig_headline_share_by_lineage.png", dpi=160)
    plt.close(fig)


def verify_sample(rows, out):
    P = out.append
    D = [r for r in rows if in_base(r)]
    rng = random.Random(0)
    alloc = {"L1": 3, "L2": 3, "L3": 2, "L4": 2, "L5": 2}
    chosen = []
    for l in BASE:
        pool = sorted(r["id"] for r in D if r["lineage"] == l)
        chosen += sorted(rng.sample(pool, min(alloc[l], len(pool))))
    P("VERIFICATION SAMPLE (seed 0, stratified: L1=3, L2=3, L3=2, L4=2, L5=2)")
    P("seeded sample: " + ", ".join(chosen))
    extras = sorted(
        ({r["id"] for r in D if r["dist_any"] == "Y"} | {r["id"] for r in D if r["selection_by_corr"] == "Y" and r["selection_metric"] in ("CORR", "BOTH")}) - set(chosen)
    )
    P("targeted extras (dist_any=Y or correlation-selection rows, checked because they drive V3 and V4): " + (", ".join(extras) or "none"))


def main():
    rows = load()
    out = []
    if "--verify-sample" in sys.argv:
        verify_sample(rows, out)
    else:
        tally(rows, out, "--figure" in sys.argv)
    print("\n".join(out))


if __name__ == "__main__":
    main()
