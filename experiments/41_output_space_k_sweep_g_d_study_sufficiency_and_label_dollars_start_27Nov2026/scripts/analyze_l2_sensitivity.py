#!/usr/bin/env python3
"""Summarizes the Experiment 41 L2-SENSITIVITY K-sweep (NOT preregistered) and draws two figures.

Reads  results/k_sweep_l2_sensitivity_shard*of*.json           (written by k_sweep_l2_sensitivity.py; 16 views x 5 split seeds)
       results/k_sweep_matched_calibration_maxiter300.json      (the preregistered run with the robustness cap; used for a replication check)
Writes results/k_sweep_l2_sensitivity_summary.json, results/k_sweep_l2_sensitivity_tables.md,
       figures/k_sweep_l2_sensitivity_by_penalty.png, figures/k_sweep_l2_sensitivity_selected_penalty.png

What was run.  The preregistered K-sweep fixed the ridge penalty at 0.01 for every head.  An independent review showed that value is arbitrary
(it does not shrink with the training-set size, and the mean-squared-error head's penalty was about 0.5 N times weaker than the softmax heads'),
so this run repeats the SAME pipeline over a grid of penalties lambda in {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1} for the cross-entropy-trained (CE-trained),
Pearson-trained and mean-squared-error-trained (MSE-trained) heads, with the same penalty FORM for every head (0.5 lambda times the sum of squared weights
on the mean loss; the MSE head is ridge with 0.5 N lambda on X'X).  Everything below is a sensitivity analysis: the grid, the selection rule and the
statistics were fixed in k_sweep_l2_sensitivity.py before it produced a number, but they are not in the preregistration file.

How the judge (head) was chosen.  Per head, per split seed: lambda is the grid value with the lowest VALIDATION normalized RPS of the deployed forecast
(raw forecast for the "raw" columns, calibrated forecast for the "matched" columns).  Every reported test number is out of sample.  The search space is the
six-value grid times a linear softmax head on 24 train-only principal components; the optimizer is L-BFGS (at most 300 iterations; fits that hit the cap
are counted below); the objective is the head's own training loss; the penalty is chosen on validation.  A non-convex head (the Pearson-trained one)
therefore reflects this optimizer, this start (zero) and this budget.

Metric.  Normalized ranked probability score (RPS divided by K - 1; range 0 to 1; lower is better), TEST prompts, averaged over each item's ratings first and then
over items (the unbiased per-sample proper risk).  A "gap" always subtracts the SAME metric on the SAME test items for two heads named by how they were trained:
    gap = test normalized RPS of the Pearson-trained (or MSE-trained) head  minus  test normalized RPS of the CE-trained head;  positive = the CE-trained head is better.
Per-seed uncertainty = paired item-clustered percentile bootstrap (2,000 draws, seed 1, 95%); across split seeds we report the mean, the range, and how many of the
five seeds have an interval above 0 / straddling 0 / below 0.  "Matched" = each head's scalar mean score passes through a multinomial-logit map fitted by
cross-entropy to the VALIDATION ratings, separately per head (same map family, not a shared map).

    python3 -I analyze_l2_sensitivity.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP40 = Path(__file__).resolve().parent.parent
RES, FIG = EXP40 / "results", EXP40 / "figures"
FIG.mkdir(exist_ok=True)
LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
TOL, BIG = 0.003, 0.005          # the preregistered tolerance and the preregistered "shape learning is indicated" level, reused for rescoring
N_VIEWS, N_SEEDS = 16, 5
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
GRAY_MARK, TEXT1, TEXT2, GRID = "#9c9c9c", "#191919", "#666666", "#e3e3e3"
K_STYLE = {2: (BLUE, "o", "K = 2 (binary)"), 3: (ORANGE, "s", "K = 3"), 5: (AQUA, "D", "K = 5 (native 1 to 5 scale)")}
ORDER = ["popquorn_offensiveness", "popquorn_politeness", "summeval_expert", "summeval_crowd", "dices990", "chaosnli_snli_mnli", "chaosnli_alphanli"]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return jsonable(o.item())
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


def load_rows():
    paths = sorted(glob.glob(str(RES / "k_sweep_l2_sensitivity_shard*of*.json")))
    if not paths:
        sys.exit("INCOMPLETE: no results/k_sweep_l2_sensitivity_shard*of*.json found")
    rows, meta = [], []
    for p in paths:
        j = json.loads(Path(p).read_text())
        rows.extend(j["rows"])
        meta.append({"file": Path(p).name, "maxiter": j["maxiter"], "seeds": j["seeds"], "boot": j["boot"], "elapsed_seconds": j["elapsed_seconds"],
                     "n_rows": len(j["rows"])})
    keys = [(r["view"], r["seed"]) for r in rows]
    if len(set(keys)) != len(keys):
        sys.exit("duplicate (view, seed) rows across shards")
    views = sorted({r["view"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    if len(views) != N_VIEWS or len(seeds) != N_SEEDS or len(rows) != N_VIEWS * N_SEEDS:
        sys.exit(f"INCOMPLETE: {len(views)} views, {len(seeds)} seeds, {len(rows)} rows (expected {N_VIEWS} x {N_SEEDS})")
    by = OrderedDict()
    for r in sorted(rows, key=lambda r: (r["view"], r["seed"])):
        by.setdefault(r["view"], []).append(r)
    return by, meta


def seedstats(rs, name):
    m = np.array([r["gaps_at_chosen"][name]["mean"] for r in rs], float)
    lo = np.array([r["gaps_at_chosen"][name]["lo"] for r in rs], float)
    hi = np.array([r["gaps_at_chosen"][name]["hi"] for r in rs], float)
    return {"seed_mean": float(m.mean()), "seed_min": float(m.min()), "seed_max": float(m.max()), "lo_seed_mean": float(lo.mean()), "hi_seed_mean": float(hi.mean()),
            "n_ci_above_0": int((lo > 0).sum()), "n_ci_straddles_0": int(((lo <= 0) & (hi >= 0)).sum()), "n_ci_below_0": int((hi < 0).sum())}


def summarize(by):
    out = OrderedDict()
    for v, rs in by.items():
        s = {"base": rs[0]["base"], "K": rs[0]["K"], "n_seeds": len(rs), "n_train": rs[0]["n_train"], "n_test": rs[0]["n_test"]}
        for name in ("raw_pearson_minus_ce", "matched_pearson_minus_ce", "matched_mse_minus_ce", "matched_mse_minus_pearson", "ce_raw_minus_ce_calibrated",
                     "constant_minus_ce_raw", "constant_minus_ce_calibrated"):
            s[name] = seedstats(rs, name)
        s["levels_test_nrps"] = {
            "constant": float(np.mean([r["constant_test_nrps"] for r in rs])),
            "ce_raw_at_selected": float(np.mean([r["grid"]["ce"][str(r["chosen_lambda"]["raw"]["ce"])]["test_raw"] for r in rs])),
            "ce_matched_at_selected": float(np.mean([r["grid"]["ce"][str(r["chosen_lambda"]["cal"]["ce"])]["test_cal"] for r in rs]))}
        s["chosen_lambda"] = {kind: {o: [r["chosen_lambda"][kind][o] for r in rs] for o in ("ce", "pearson", "mse")} for kind in ("raw", "cal")}
        st = {}
        for name in ("consensus", "disputed"):
            sub = [r["strata_at_chosen"][name] for r in rs if r.get("strata_at_chosen") and name in r["strata_at_chosen"]]
            if sub:
                st[name] = {"n_seeds": len(sub), "n_items_mean": float(np.mean([x["n_items"] for x in sub])),
                            "raw_gap_seed_mean": float(np.mean([x["raw_pearson_minus_ce"]["mean"] for x in sub])),
                            "matched_gap_seed_mean": float(np.mean([x["matched_pearson_minus_ce"]["mean"] for x in sub]))}
        s["strata"] = st
        # signal: the CE-trained head at its selected penalty beats the constant forecast on average over seeds (primary) / with the interval above 0 on >= 3 of 5 seeds (strict)
        s["signal_raw"] = s["constant_minus_ce_raw"]["seed_mean"] > 0
        s["signal_strict"] = s["constant_minus_ce_calibrated"]["seed_mean"] > 0 and s["constant_minus_ce_calibrated"]["n_ci_above_0"] >= 3
        out[v] = s
    return out


def by_common_lambda(by, summ):
    names = ("raw_pearson_minus_ce", "matched_pearson_minus_ce", "raw_mse_minus_ce", "matched_mse_minus_ce")
    out = OrderedDict()
    for lam in LAMBDAS:
        key = str(lam)
        per_view = {n: [] for n in names}
        per_view["signal_over_constant_matched_ce"] = []
        per_view["ce_raw_minus_ce_cal"] = []
        per_view_sig = {n: [] for n in names}
        cap = {"ce": [], "pearson": []}
        for v, rs in by.items():
            for n in names:
                x = float(np.mean([r["gaps_at_common_lambda"][key][n] for r in rs]))
                per_view[n].append(x)
                if summ[v]["signal_raw"]:
                    per_view_sig[n].append(x)
            per_view["signal_over_constant_matched_ce"].append(float(np.mean([r["gaps_at_common_lambda"][key]["constant"] - r["gaps_at_common_lambda"][key]["ce_cal"] for r in rs])))
            per_view["ce_raw_minus_ce_cal"].append(float(np.mean([r["gaps_at_common_lambda"][key]["ce_raw"] - r["gaps_at_common_lambda"][key]["ce_cal"] for r in rs])))
            for o in ("ce", "pearson"):
                cap[o].extend(bool(r["grid"][o][key]["hit_cap"]) for r in rs)
        out[key] = {"per_view": per_view, "per_view_signal_only": per_view_sig, "n_signal_views": int(sum(s["signal_raw"] for s in summ.values())),
                    "hit_cap_share": {o: float(np.mean(c)) for o, c in cap.items()}}
    return out


def replication_check(by):
    pre = json.loads((RES / "k_sweep_matched_calibration_maxiter300.json").read_text())
    pre_by = {(r["view"], r["seed"]): r for r in pre["rows"]}
    d_raw, d_mat, d_mse_raw, d_mse_mat, n = [], [], [], [], 0
    for v, rs in by.items():
        for r in rs:
            p = pre_by.get((v, r["seed"]))
            if p is None:
                continue
            c = r["gaps_at_common_lambda"]["0.01"]
            d_raw.append(c["raw_pearson_minus_ce"] - p["gaps"]["raw_pearson_minus_ce"]["nrps"]["mean"])
            d_mat.append(c["matched_pearson_minus_ce"] - p["gaps"]["matched_pearson_minus_ce"]["nrps"]["mean"])
            d_mse_raw.append(c["raw_mse_minus_ce"] - p["gaps"]["raw_mse_minus_ce"]["nrps"]["mean"])
            d_mse_mat.append(c["matched_mse_minus_ce"] - p["gaps"]["matched_mse_minus_ce"]["nrps"]["mean"])
            n += 1
    f = lambda a: {"max_abs": float(np.max(np.abs(a))), "mean": float(np.mean(a))}
    return {"n_cells_compared": n, "raw_pearson_minus_ce": f(d_raw), "matched_pearson_minus_ce": f(d_mat),
            "raw_mse_minus_ce": f(d_mse_raw), "matched_mse_minus_ce": f(d_mse_mat),
            "reading": "Pearson-minus-CE gaps at lambda = 0.01 should reproduce the preregistered 300-iteration run (same penalty function); the MSE gaps should NOT, because the ridge form changed (0.5 N lambda on X'X instead of lambda)."}


def rescore(summ):
    """Mirrors analyze_k_sweep.score() on the validation-selected penalties.  Labeled L2-sensitivity, not preregistered."""
    p = OrderedDict()
    b = {v: s["matched_pearson_minus_ce"]["seed_mean"] for v, s in summ.items() if s["K"] == 2}
    p["P1_binary_matched_gap"] = {"rule": f"on every binary view the absolute seed-mean matched gap is below {TOL}", "values": b, "max_abs": max(abs(x) for x in b.values()),
                                  "supported": all(abs(x) < TOL for x in b.values()), "violations": [v for v, x in b.items() if abs(x) >= TOL]}
    for tag, key in (("primary: signal = seed-mean (constant minus CE-trained head, raw) > 0", "signal_raw"), ("strict: calibrated CE-trained head beats constant with the interval above 0 on >= 3 of 5 seeds", "signal_strict")):
        sv = [v for v, s in summ.items() if s[key]]
        pos = [v for v in sv if summ[v]["raw_pearson_minus_ce"]["seed_mean"] > 0]
        p["P2_raw_gap_positive_" + key] = {"rule": "seed-mean raw gap positive on at least 90% of signal views (" + tag + ")", "n_signal_views": len(sv), "n_positive": len(pos),
                                          "share_positive": len(pos) / len(sv) if sv else None, "supported": bool(sv) and len(pos) / len(sv) >= 0.9, "exceptions": [v for v in sv if v not in pos]}
    k3 = {v: s["matched_pearson_minus_ce"]["seed_mean"] for v, s in summ.items() if s["K"] >= 3}
    within = [v for v, x in k3.items() if abs(x) < TOL]
    big = [v for v, x in k3.items() if x > BIG and summ[v]["matched_pearson_minus_ce"]["n_ci_above_0"] >= 3]
    mod = [v for v, x in k3.items() if x >= TOL and summ[v]["matched_pearson_minus_ce"]["n_ci_above_0"] >= 3]
    p["P3_K3plus_matched_gap"] = {"rule": f"for K of 3 or more the absolute seed-mean matched gap is below {TOL} on at least 80% of views; refuted if a K >= 3 view has a gap above {BIG} with the interval above 0 on at least 3 of 5 seeds",
                                  "n_views": len(k3), "n_within_tolerance": len(within), "share_within": len(within) / len(k3), "max_matched_gap": max(k3.values()), "min_matched_gap": min(k3.values()),
                                  "supported": len(within) / len(k3) >= 0.8 and not big, "shape_learning_indicated_on": big,
                                  "views_at_or_above_tolerance_with_interval_above_0_on_3_of_5_seeds": mod}
    rows, fails = {}, []
    for v, s in summ.items():
        if len(s["strata"]) < 2:
            continue
        c, d_ = s["strata"]["consensus"], s["strata"]["disputed"]
        ok_raw = c["raw_gap_seed_mean"] > 0 and d_["raw_gap_seed_mean"] > 0
        ok_mat = abs(c["matched_gap_seed_mean"]) < TOL and abs(d_["matched_gap_seed_mean"]) < TOL
        rows[v] = {"raw_positive_in_both": ok_raw, "matched_below_tol_in_both": ok_mat}
        if not (ok_raw and ok_mat):
            fails.append(v)
    p["P5_strata"] = {"rule": f"in both entropy strata the seed-mean raw gap is positive and the absolute seed-mean matched gap is below {TOL}", "n_views_with_strata": len(rows),
                      "n_views_both_conditions": len(rows) - len(fails), "violations": fails, "supported_literal": not fails}
    return p


# ----------------------------------------------------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------------------------------------------------
def cell(s, name, digits=4):
    g = s[name]
    return f"{g['seed_mean']:+.{digits}f} ({g['seed_min']:+.{digits}f}..{g['seed_max']:+.{digits}f}); {g['n_ci_above_0']}/{g['n_ci_straddles_0']}/{g['n_ci_below_0']}"


def tables(summ, common, rep, resc, meta):
    L = []
    L.append("# Experiment 41: L2-sensitivity K-sweep, tables (NOT preregistered; generated by scripts/analyze_l2_sensitivity.py)\n")
    L.append("**Reading guide.** Every number is the TEST split, normalized ranked probability score (RPS divided by K - 1; 0 to 1; lower is better), averaged over each test item's ratings first and then over items. "
             "gap = test normalized RPS of the Pearson-trained (or MSE-trained) head minus test normalized RPS of the cross-entropy-trained (CE-trained) head; positive = the CE-trained head is better. "
             "\"raw\" = the head's own distribution, ridge penalty chosen per head on VALIDATION normalized RPS of the raw forecast; \"matched\" = the head's scalar mean score passed through a multinomial-logit map "
             "fitted by cross-entropy to the validation ratings (separately per head), penalty chosen on VALIDATION normalized RPS of the calibrated forecast. "
             "Each cell: seed-mean (min..max over five split seeds); number of seeds whose paired item-clustered 95% bootstrap interval is above 0 / straddles 0 / is below 0. "
             "Heads: linear softmax on 24 train-only principal components of a 256-dimensional signed token hash, L-BFGS (at most 300 iterations), penalty grid 1e-5 to 1.\n")
    L.append(f"Files merged: {', '.join(m['file'] for m in meta)}; rows {sum(m['n_rows'] for m in meta)} (= {N_VIEWS} views x {N_SEEDS} seeds).\n")

    L.append("## 1. Per view at the validation-selected penalties\n")
    L.append("| view | K | signal (raw) | constant | CE-trained matched | raw gap, Pearson-trained minus CE-trained | matched gap, Pearson-trained minus CE-trained | matched gap, MSE-trained minus CE-trained | CE-trained raw minus CE-trained matched |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for v, s in summ.items():
        lv = s["levels_test_nrps"]
        L.append(f"| {v} | {s['K']} | {'yes' if s['signal_raw'] else 'no'} | {lv['constant']:.4f} | {lv['ce_matched_at_selected']:.4f} | {cell(s, 'raw_pearson_minus_ce')} | "
                 f"{cell(s, 'matched_pearson_minus_ce')} | {cell(s, 'matched_mse_minus_ce')} | {cell(s, 'ce_raw_minus_ce_calibrated')} |")
    L.append("")

    L.append("## 2. The same gaps at every COMMON penalty (mean over the 16 views of the 5-seed mean; min..max over views in brackets)\n")
    L.append("Every head uses the same penalty in a row, so no head is tuned. The 0.01 row is the preregistered penalty for the CE-trained and Pearson-trained heads (the MSE-trained head's ridge form differs from the preregistered run).\n")
    L.append("| penalty | raw gap, Pearson-trained minus CE-trained | matched gap, Pearson-trained minus CE-trained | raw gap, MSE-trained minus CE-trained | matched gap, MSE-trained minus CE-trained | signal: constant minus matched CE-trained | CE-trained raw minus matched | CE-trained / Pearson-trained fits at the iteration cap |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key, c in common.items():
        pv = c["per_view"]
        f = lambda n: f"{np.mean(pv[n]):+.4f} [{np.min(pv[n]):+.4f}, {np.max(pv[n]):+.4f}]"
        L.append(f"| {key} | {f('raw_pearson_minus_ce')} | {f('matched_pearson_minus_ce')} | {f('raw_mse_minus_ce')} | {f('matched_mse_minus_ce')} | {f('signal_over_constant_matched_ce')} | "
                 f"{f('ce_raw_minus_ce_cal')} | {c['hit_cap_share']['ce']:.0%} / {c['hit_cap_share']['pearson']:.0%} |")
    L.append("")
    L.append("Signal views only (CE-trained head beats the constant forecast at the selected penalty; n = " + str(next(iter(common.values()))["n_signal_views"]) + "):\n")
    L.append("| penalty | raw gap, Pearson-trained minus CE-trained | matched gap, Pearson-trained minus CE-trained | matched gap, MSE-trained minus CE-trained |")
    L.append("|---|---|---|---|")
    for key, c in common.items():
        pv = c["per_view_signal_only"]
        L.append(f"| {key} | {np.mean(pv['raw_pearson_minus_ce']):+.4f} | {np.mean(pv['matched_pearson_minus_ce']):+.4f} | {np.mean(pv['matched_mse_minus_ce']):+.4f} |")
    L.append("")

    L.append("## 3. Which penalties the validation set chose (counts over 16 views x 5 seeds = 80 fits per head)\n")
    L.append("A choice at 1e-05 or 1 sits on the edge of the grid; the grid may be too narrow for those fits.\n")
    L.append("| forecast | head | " + " | ".join(str(x) for x in LAMBDAS) + " |")
    L.append("|---|---|" + "---|" * len(LAMBDAS))
    for kind, label in (("raw", "raw"), ("cal", "matched")):
        for o in ("ce", "pearson", "mse"):
            allv = [x for s in summ.values() for x in s["chosen_lambda"][kind][o]]
            L.append(f"| {label} | {o} | " + " | ".join(str(sum(1 for a in allv if a == lam)) for lam in LAMBDAS) + " |")
    L.append("")

    L.append("## 4. Replication check against the preregistered run (penalty 0.01, 300 iterations)\n")
    L.append(f"Cells compared: {rep['n_cells_compared']}. Largest absolute difference in the seed gap, this run minus the preregistered run: "
             f"raw Pearson-trained minus CE-trained {rep['raw_pearson_minus_ce']['max_abs']:.2e}; matched {rep['matched_pearson_minus_ce']['max_abs']:.2e}; "
             f"raw MSE-trained minus CE-trained {rep['raw_mse_minus_ce']['max_abs']:.2e}; matched {rep['matched_mse_minus_ce']['max_abs']:.2e}. {rep['reading']}\n")

    L.append("## 5. Preregistered rules re-scored at the validation-selected penalties (L2-sensitivity; NOT preregistered)\n")
    for k, r in resc.items():
        flag = r.get("supported", r.get("supported_literal"))
        L.append(f"- **{k}**: {'supported' if flag else 'not supported'}. Rule: {r['rule']}. " + "; ".join(f"{a} = {b}" for a, b in r.items() if a in ("max_abs", "n_signal_views", "n_positive", "share_positive", "exceptions", "n_views", "n_within_tolerance",
                 "share_within", "max_matched_gap", "shape_learning_indicated_on", "views_at_or_above_tolerance_with_interval_above_0_on_3_of_5_seeds", "n_views_with_strata", "n_views_both_conditions", "violations")))
    L.append("")

    L.append("## 6. Entropy strata at the selected penalties (seed-mean over views with strata)\n")
    L.append("| view | K | consensus: raw gap | consensus: matched gap | disputed: raw gap | disputed: matched gap |")
    L.append("|---|---|---|---|---|---|")
    for v, s in summ.items():
        if len(s["strata"]) == 2:
            c, d_ = s["strata"]["consensus"], s["strata"]["disputed"]
            L.append(f"| {v} | {s['K']} | {c['raw_gap_seed_mean']:+.4f} | {c['matched_gap_seed_mean']:+.4f} | {d_['raw_gap_seed_mean']:+.4f} | {d_['matched_gap_seed_mean']:+.4f} |")
    L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------------------
def style_axis(ax, logx=False):
    ax.grid(True, which="major", linewidth=0.7, color=GRID, linestyle="-")
    if logx:
        ax.minorticks_on()
        ax.grid(True, which="minor", axis="x", linewidth=0.4, color="#efefef", linestyle="-")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bdbdbd")
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.axhline(0, color="#8a8a8a", linewidth=1.0, zorder=1)


def fig_by_penalty(by, common):
    keys = [str(l) for l in LAMBDAS]
    panels = (("raw_pearson_minus_ce", "A. Raw forecasts", "Pearson-trained", "Pearson-trained"),
              ("matched_pearson_minus_ce", "B. After the per-head calibration map", "Pearson-trained", "Pearson-trained"),
              ("matched_mse_minus_ce", "C. After the per-head calibration map", "mean-squared-error-trained (MSE-trained)", "MSE-trained"))
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.0))
    for ax, (name, title, head, short) in zip(axes, panels):
        style_axis(ax, logx=True)
        ax.set_xscale("log")
        mat = np.array([common[k]["per_view"][name] for k in keys])   # penalties x views
        for j in range(mat.shape[1]):
            ax.plot(LAMBDAS, mat[:, j], color=GRAY_MARK, linewidth=0.9, alpha=0.75, zorder=2)
        ax.plot(LAMBDAS, np.median(mat, axis=1), color=BLUE, linewidth=2.6, marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        ax.axvline(0.01, color=ORANGE, linewidth=1.3, linestyle="--", zorder=3)
        ax.set_title(f"{title}: {short} minus CE-trained", loc="left", fontsize=10.5, color=TEXT1, fontweight="bold")
        ax.set_xlabel("Ridge penalty, the same for both heads in a comparison (log scale)", fontsize=9, color=TEXT1)
        ax.set_ylabel(f"Test normalized ranked probability score (RPS) of the {head} head\nminus that of the cross-entropy-trained (CE-trained) head;\npositive: CE-trained head better", fontsize=8.6, color=TEXT1)
    handles = [plt.Line2D([], [], color=GRAY_MARK, linewidth=1.2, label="each of the 16 views (mean over 5 split seeds)"),
               plt.Line2D([], [], color=BLUE, linewidth=2.6, marker="o", markeredgecolor="white", label="median over the 16 views"),
               plt.Line2D([], [], color=ORANGE, linewidth=1.3, linestyle="--", label="preregistered penalty 0.01")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("Experiment 41, L2-sensitivity run (not preregistered): gap between heads versus the ridge penalty", x=0.01, ha="left", fontsize=12, color=TEXT1)
    fig.text(0.01, 0.008, "Test prompts only. Every point: both heads trained with the same penalty (nothing tuned). Linear softmax head on 24 principal components of hashed text, L-BFGS (at most 300 iterations); "
             "the calibration map is fitted by cross-entropy to validation ratings, separately per head.\nThe vertical scale differs by panel: panel A spans 0.4, panel B 0.007, panel C 0.002.", fontsize=8.2, color=TEXT2, va="bottom")
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.savefig(FIG / "k_sweep_l2_sensitivity_by_penalty.png", dpi=150)
    plt.close(fig)


def layout(summ):
    ys, y = OrderedDict(), 0.0
    for base in ORDER:
        vs = [v for v, s in summ.items() if s["base"] == base]
        for v in sorted(vs, key=lambda x: summ[x]["K"]):
            ys[v] = y
            y -= 1.0
        y -= 0.55
    return ys


def fig_selected(summ):
    ys = layout(summ)
    panels = (("raw_pearson_minus_ce", "A. Raw forecasts, Pearson-trained minus CE-trained", "Test normalized ranked probability score (RPS) of the Pearson-trained head\nminus that of the cross-entropy-trained (CE-trained) head; positive: CE-trained better", None, None),
              ("matched_pearson_minus_ce", "B. After the per-head calibration map, Pearson-trained minus CE-trained", "Test normalized ranked probability score (RPS) of the Pearson-trained head\nminus that of the CE-trained head; positive: CE-trained better", True, (-0.0045, 0.0050)),
              ("matched_mse_minus_ce", "C. After the per-head calibration map, MSE-trained minus CE-trained", "Test normalized ranked probability score (RPS) of the mean-squared-error-trained\n(MSE-trained) head minus that of the CE-trained head; positive: CE-trained better", False, (-0.0022, 0.0022)))
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 8.8), sharey=True)
    for ax, (name, title, xl, band, xlim) in zip(axes, panels):
        ax.grid(True, linewidth=0.7, color=GRID)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#bdbdbd")
        ax.tick_params(colors=TEXT2, labelsize=9)
        ax.axvline(0, color="#8a8a8a", linewidth=1.0, zorder=1)
        if band:
            ax.axvspan(-TOL, TOL, color="#eef4fd", zorder=0)
        for v, y in ys.items():
            g = summ[v][name]
            col, mk, _ = K_STYLE[summ[v]["K"]]
            ax.plot([g["seed_min"], g["seed_max"]], [y, y], color=col, linewidth=2.0, solid_capstyle="round", zorder=3)
            ax.plot([g["seed_mean"]], [y], marker=mk, markersize=8, color=col, markeredgecolor="white", markeredgewidth=1.4, linestyle="none", zorder=4)
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_title(title, loc="left", fontsize=10, color=TEXT1, fontweight="bold")
        ax.set_xlabel(xl, fontsize=8.8, color=TEXT1)
    axes[0].set_yticks(list(ys.values()))
    axes[0].set_yticklabels([f"{summ[v]['base']}  K={summ[v]['K']}" for v in ys], fontsize=9, color=TEXT1)
    handles = [plt.Line2D([], [], marker=m, color=c, markersize=8, markeredgecolor="white", markeredgewidth=1.2, linewidth=2.0, label=l) for c, m, l in K_STYLE.values()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("Experiment 41, L2-sensitivity run (not preregistered): gaps between heads with the ridge penalty chosen per head on validation", x=0.01, ha="left", fontsize=12, color=TEXT1)
    fig.text(0.01, 0.008, "Test prompts only. Dot = mean over five split seeds, bar = range over the seeds. Shaded band in panel B = the preregistered tolerance of +/-0.003 (it lies outside the horizontal range of panel C). Penalty grid 1e-5 to 1, chosen\n"
             "per head and per seed on validation normalized RPS; linear softmax head on 24 principal components of hashed text; L-BFGS (at most 300 iterations). The horizontal scale differs by panel. VeriBench could not be rebuilt on this machine.", fontsize=8.2, color=TEXT2, va="bottom")
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(FIG / "k_sweep_l2_sensitivity_selected_penalty.png", dpi=150)
    plt.close(fig)


def main():
    by, meta = load_rows()
    summ = summarize(by)
    common = by_common_lambda(by, summ)
    rep = replication_check(by)
    resc = rescore(summ)
    (RES / "k_sweep_l2_sensitivity_summary.json").write_text(json.dumps(jsonable({"files": meta, "summary": summ, "common_penalty": common, "replication_check": rep, "rescored_rules": resc}), indent=1))
    (RES / "k_sweep_l2_sensitivity_tables.md").write_text(tables(summ, common, rep, resc, meta))
    fig_by_penalty(by, common)
    fig_selected(summ)
    print(f"views {len(summ)}; signal views (raw) {sum(s['signal_raw'] for s in summ.values())}; strict signal {sum(s['signal_strict'] for s in summ.values())}")
    for k, r in resc.items():
        print(k, "supported" if r.get("supported", r.get("supported_literal")) else "NOT supported", {a: b for a, b in r.items() if a in ("max_abs", "share_positive", "share_within", "max_matched_gap", "shape_learning_indicated_on", "violations")})
    print("replication check:", {k: v for k, v in rep.items() if k != "reading"})


if __name__ == "__main__":
    main()
