#!/usr/bin/env python3
"""Scores preregistered predictions P1 to P5 on the Experiment 41 K-sweep and draws the figures.

Reads  results/k_sweep_matched_calibration_maxiter35.json   (the preregistered cap: at most 35 L-BFGS iterations)
       results/k_sweep_matched_calibration_maxiter300.json  (the robustness run: at most 300 iterations)
Writes results/k_sweep_summary.json, results/k_sweep_summary_table.md,
       figures/k_sweep_raw_vs_matched_gap.png, figures/k_sweep_entropy_strata.png, figures/k_sweep_signal_over_constant.png

Reading guide (also in HOW_TO_READ_RESULTS.md).  Every number here is the TEST split (prompts never used for fitting or for choosing
the calibration map), normalized ranked probability score (RPS divided by K minus 1; 0 to 1; lower is better), averaged over each
test item's ratings first and then over items (the unbiased per-sample risk).  A "gap" always subtracts the SAME metric on the SAME
test items for two methods, named by how they were trained:
    gap = test normalized RPS of the Pearson-trained head  minus  test normalized RPS of the CE-trained head
    positive = the cross-entropy-trained (CE-trained) head is better.
"raw" = each head's own distribution; "matched" = each head's scalar mean score is standardized with VALIDATION moments and passed through its OWN multinomial-logit
calibration map (the same family for every head), fitted by cross-entropy to the VALIDATION ratings.  Dot = mean over five split seeds, bar = range over the five seeds.
Heads: linear softmax on 24 train-only principal components of a 256-dimensional signed token hash (Experiment 36), ridge penalty FIXED at 0.01 for every head and every training
size (the preregistered setting; nothing is tuned).  analyze_l2_sensitivity.py shows that this penalty over-shrinks the CE-trained head, so read these results as "at L2 = 0.01".
Operationalizations fixed in this script, not in the preregistration: P2's "signal" and P3's "interval excludes 0" use at least 3 of 5 seeds; P3's clause "while its K = 2 view
shows about 0" is not applied (it can only make a refutation harder to trigger).

    python3 -I analyze_k_sweep.py
"""
from __future__ import annotations

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
TOL = 0.003            # preregistered tolerance for "the matched gap is zero up to noise"
BIG = 0.005            # preregistered level for "shape learning is indicated"

# ---- one validated categorical palette (dataviz skill, slots 1-3, validated light, all pairs); text uses text tokens, never the data colour ----
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


def load(cap):
    d = json.loads((RES / f"k_sweep_matched_calibration_maxiter{cap}.json").read_text())
    by = OrderedDict()
    for r in d["rows"]:
        by.setdefault(r["view"], []).append(r)
    return d, by


def vals(rows, getter):
    return np.array([getter(r) for r in rows], float)


def summarize(by):
    out = OrderedDict()
    for v, rs in by.items():
        base, k = rs[0]["base"], rs[0]["K"]
        s = {"base": base, "K": k, "n_seeds": len(rs), "n_train": rs[0]["n_train"], "n_test": rs[0]["n_test"]}
        g = lambda name, key="mean": vals(rs, lambda r: r["gaps"][name]["nrps"][key])
        for name in ("raw_pearson_minus_ce", "matched_pearson_minus_ce", "ce_raw_minus_ce_calibrated", "raw_mse_minus_ce",
                     "matched_mse_minus_ce", "constant_minus_ce_raw"):
            m, lo, hi = g(name, "mean"), g(name, "lo"), g(name, "hi")
            s[name] = {"seed_mean": float(m.mean()), "seed_min": float(m.min()), "seed_max": float(m.max()),
                       "bootstrap_lo_seed_mean": float(lo.mean()), "bootstrap_hi_seed_mean": float(hi.mean()),
                       "n_seeds_ci_excludes_0_positive": int((lo > 0).sum()), "n_seeds_ci_excludes_0_negative": int((hi < 0).sum())}
        lv = lambda head, kind: vals(rs, lambda r: r["mean"][head][kind]).mean()
        s["levels_test_nrps"] = {"constant": float(lv("constant", "raw")), "ce_raw": float(lv("ce", "raw")), "ce_matched": float(lv("ce", "cal")),
                                 "pearson_raw": float(lv("pearson", "raw")), "pearson_matched": float(lv("pearson", "cal")),
                                 "mse_raw": float(lv("mse", "raw")), "mse_matched": float(lv("mse", "cal"))}
        s["hit_cap_count_of_seeds"] = {h: int(sum(bool(r["optimizer"][h]["hit_cap"]) for r in rs)) for h in ("ce", "pearson", "mse")}
        st = {}
        for name in ("consensus", "disputed"):
            sub = [r["strata"][name] for r in rs if r.get("strata") and name in r["strata"]]
            if sub:
                st[name] = {"n_items_mean": float(np.mean([x["n_items"] for x in sub])), "n_seeds": len(sub),
                            "raw_gap_seed_mean": float(np.mean([x["raw_pearson_minus_ce"]["mean"] for x in sub])),
                            "matched_gap_seed_mean": float(np.mean([x["matched_pearson_minus_ce"]["mean"] for x in sub])),
                            "matched_gap_lo_seed_mean": float(np.mean([x["matched_pearson_minus_ce"]["lo"] for x in sub])),
                            "matched_gap_hi_seed_mean": float(np.mean([x["matched_pearson_minus_ce"]["hi"] for x in sub]))}
        s["strata"] = st
        raw, mat = s["raw_pearson_minus_ce"]["seed_mean"], s["matched_pearson_minus_ce"]["seed_mean"]
        s["share_of_raw_gap_removed_by_calibration"] = (1 - mat / raw) if abs(raw) > 1e-4 else None
        out[v] = s
    return out


def score(s35, s300):
    sig = {v: s["constant_minus_ce_raw"]["seed_mean"] > 0 for v, s in s35.items()}
    sig_strict = {v: (s["constant_minus_ce_raw"]["seed_mean"] > 0 and s["constant_minus_ce_raw"]["n_seeds_ci_excludes_0_positive"] >= 3)
                  for v, s in s35.items()}
    p = OrderedDict()
    # P1: binary views, |seed-mean matched gap| < TOL on every one
    b = {v: s["matched_pearson_minus_ce"]["seed_mean"] for v, s in s35.items() if s["K"] == 2}
    p["P1"] = {"rule": f"on every binary view the absolute seed-mean matched gap is below {TOL}",
               "values": b, "max_abs": max(abs(x) for x in b.values()), "n_views": len(b),
               "supported": all(abs(x) < TOL for x in b.values()),
               "violations": [v for v, x in b.items() if abs(x) >= TOL]}
    # P2: raw gap positive on >= 90% of signal views
    for tag, flags in (("primary: signal = seed-mean (constant minus CE-trained head) > 0", sig),
                       ("stricter: signal also needs the paired interval above 0 on at least 3 of 5 seeds", sig_strict)):
        sv = [v for v, f in flags.items() if f]
        pos = [v for v in sv if s35[v]["raw_pearson_minus_ce"]["seed_mean"] > 0]
        key = "P2" if tag.startswith("primary") else "P2_strict_signal"
        p[key] = {"rule": "the seed-mean raw gap is positive on at least 90% of views that carry signal (" + tag + ")",
                  "n_signal_views": len(sv), "n_positive": len(pos), "share_positive": len(pos) / len(sv) if sv else None,
                  "supported": bool(sv) and len(pos) / len(sv) >= 0.9, "signal_views": sv,
                  "exceptions": [v for v in sv if v not in pos]}
    # P3: K>=3 views, matched gap below TOL on >= 80%
    k3 = {v: s["matched_pearson_minus_ce"]["seed_mean"] for v, s in s35.items() if s["K"] >= 3}
    within = [v for v, x in k3.items() if abs(x) < TOL]
    big = [v for v, x in k3.items() if x > BIG and s35[v]["matched_pearson_minus_ce"]["n_seeds_ci_excludes_0_positive"] >= 3]
    p["P3"] = {"rule": f"for K of 3 or more the absolute seed-mean matched gap is below {TOL} on at least 80% of views; "
                       f"refuted if some K >= 3 view has a matched gap above {BIG} with the interval above 0 on at least 3 of 5 seeds "
                       "(the preregistration says 'whose interval excludes 0' and adds 'while its K = 2 view shows about 0'; the 3-of-5 reading is this script's operationalization and the K = 2 clause is not applied)",
               "n_views": len(k3), "n_within_tolerance": len(within), "share_within": len(within) / len(k3), "max_matched_gap": max(k3.values()),
               "min_matched_gap": min(k3.values()), "supported": len(within) / len(k3) >= 0.8 and not big, "shape_learning_indicated_on": big}
    # P4: cap 35 versus 300
    dif = {v: s300[v]["matched_pearson_minus_ce"]["seed_mean"] - s35[v]["matched_pearson_minus_ce"]["seed_mean"] for v in s35 if v in s300}
    draw = {v: s300[v]["raw_pearson_minus_ce"]["seed_mean"] - s35[v]["raw_pearson_minus_ce"]["seed_mean"] for v in s35 if v in s300}
    p["P4"] = {"rule": f"raising the iteration cap from 35 to 300 changes no seed-mean matched gap by more than {TOL}",
               "max_abs_change_matched": max(abs(x) for x in dif.values()), "max_abs_change_raw": max(abs(x) for x in draw.values()),
               "supported": all(abs(x) <= TOL for x in dif.values()),
               "pearson_hit_cap_35_total": int(sum(s["hit_cap_count_of_seeds"]["pearson"] for s in s35.values())),
               "pearson_hit_cap_300_total": int(sum(s["hit_cap_count_of_seeds"]["pearson"] for s in s300.values())),
               "ce_hit_cap_35_total": int(sum(s["hit_cap_count_of_seeds"]["ce"] for s in s35.values())),
               "ce_hit_cap_300_total": int(sum(s["hit_cap_count_of_seeds"]["ce"] for s in s300.values())),
               "n_view_seed_cells": int(sum(s["n_seeds"] for s in s35.values()))}
    # P5: strata (literal reading: every view that has strata; the signal-view reading is reported beside it)
    rows, fails = {}, []
    for v, s in s35.items():
        if not s["strata"] or len(s["strata"]) < 2:
            continue
        c, d_ = s["strata"]["consensus"], s["strata"]["disputed"]
        ok_raw = c["raw_gap_seed_mean"] > 0 and d_["raw_gap_seed_mean"] > 0
        ok_mat = abs(c["matched_gap_seed_mean"]) < TOL and abs(d_["matched_gap_seed_mean"]) < TOL
        rows[v] = {"raw_positive_in_both": ok_raw, "matched_below_tol_in_both": ok_mat}
        if not (ok_raw and ok_mat):
            fails.append(v)
    sigrows = [v for v in rows if sig[v]]
    p["P5"] = {"rule": f"in both entropy strata the seed-mean raw gap is positive and the absolute seed-mean matched gap is below {TOL} (literal reading: every view with strata)",
               "n_views_with_strata": len(rows), "n_views_both_conditions": len(rows) - len(fails), "violations": fails, "per_view": rows,
               "supported_literal": not fails,
               "n_signal_views_with_strata": len(sigrows), "n_signal_views_both_conditions": sum(1 for v in sigrows if v not in fails),
               "supported_on_signal_views": all(v not in fails for v in sigrows)}
    return p, sig


# ----------------------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------------------
def style(ax):
    ax.grid(True, linewidth=0.6, color=GRID, linestyle="-")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bdbdbd")
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.xaxis.label.set_color(TEXT1)
    ax.axvline(0, color="#8a8a8a", linewidth=1.0, zorder=1)


def layout(s35, names):
    ys, y = OrderedDict(), 0.0
    for base in ORDER:
        vs = [v for v in names if s35[v]["base"] == base]
        if not vs:
            continue
        for v in sorted(vs, key=lambda x: s35[x]["K"]):
            ys[v] = y
            y -= 1.0
        y -= 0.55
    return ys


def draw_gap_panel(ax, s35, ys, key, tol_band):
    style(ax)
    if tol_band:
        ax.axvspan(-TOL, TOL, color="#eef4fd", zorder=0)
    for v, y in ys.items():
        s = s35[v][key]
        col, mk, _ = K_STYLE[s35[v]["K"]]
        ax.plot([s["seed_min"], s["seed_max"]], [y, y], color=col, linewidth=2.0, zorder=3, solid_capstyle="round")
        ax.plot([s["seed_mean"]], [y], marker=mk, markersize=8, color=col, markeredgecolor="white", markeredgewidth=1.5, zorder=4, linestyle="none")
    ax.set_ylim(min(ys.values()) - 0.8, 0.8)
    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels([f"{s35[v]['base']}  K={s35[v]['K']}" for v in ys], fontsize=9, color=TEXT1)


def fig_gap(s35, p, constant_range):
    ys = layout(s35, list(s35))
    fig, (a, b) = plt.subplots(1, 2, figsize=(15.5, 9.4), sharey=True)
    draw_gap_panel(a, s35, ys, "raw_pearson_minus_ce", False)
    draw_gap_panel(b, s35, ys, "matched_pearson_minus_ce", True)
    raw = {v: s["raw_pearson_minus_ce"]["seed_mean"] for v, s in s35.items()}
    mat = {v: s["matched_pearson_minus_ce"]["seed_mean"] for v, s in s35.items()}
    lab = ("Test normalized ranked probability score (RPS) of the Pearson-trained head\nminus test normalized RPS of the CE-trained head\n"
           "(positive = CE-trained head better; lower RPS is better)")
    a.set_xlabel("Raw gap, before any calibration map\n" + lab, fontsize=9.5)
    b.set_xlabel("Matched gap, after each head's mean score passes through its own validation-fitted calibration map\n" + lab, fontsize=9.5)
    npos = sum(x > 0 for x in raw.values())
    a.set_title(f"A. Raw gap: positive on {npos} of {len(raw)} views, up to {max(raw.values()):.3f}", loc="left", fontsize=12, color=TEXT1, fontweight="bold")
    b.set_title(f"B. Matched gap: at most {max(abs(x) for x in mat.values()):.4f} in absolute value", loc="left", fontsize=12, color=TEXT1, fontweight="bold")
    # selective direct labels: the one view where the Pearson-trained head is better raw, and the largest matched gap
    vmin = min(raw, key=raw.get)
    if raw[vmin] < 0:
        a.annotate("Pearson-trained head better here", (s35[vmin]["raw_pearson_minus_ce"]["seed_mean"], ys[vmin]), xytext=(12, 0),
                   textcoords="offset points", va="center", fontsize=8.5, color=TEXT1)
    vmax = max(mat, key=mat.get)
    b.annotate(f"{mat[vmax]:+.4f}", (s35[vmax]["matched_pearson_minus_ce"]["seed_max"], ys[vmax]), xytext=(7, 0),
               textcoords="offset points", va="center", fontsize=8.5, color=TEXT1)
    b.text(TOL * 0.985, 0.45, f"shaded: pre-registered tolerance ±{TOL}", fontsize=8.5, color=TEXT2, va="center", ha="right")
    handles = [plt.Line2D([], [], marker=m, color=c, markersize=8, markeredgecolor="white", markeredgewidth=1.2, linewidth=2.0, label=l) for c, m, l in K_STYLE.values()]
    a.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("Experiment 41 K-sweep: linear heads on hashed text, held-out TEST prompts, five split seeds (dot = mean over seeds, bar = range over seeds)",
                 x=0.01, ha="left", fontsize=11.5, color=TEXT1)
    fig.text(0.01, 0.012, f"Scale: the constant (train-marginal) forecast scores {constant_range[0]:.3f} to {constant_range[1]:.3f} test normalized RPS on these views.\n"
             "Both heads: linear softmax on 24 train-only principal components of a 256-dimensional token hash, ridge penalty fixed at 0.01 (preregistered, not tuned), at most 35 L-BFGS iterations (no head reached the cap). "
             "VeriBench could not be rebuilt on this machine.", fontsize=8.2, color=TEXT2, va="bottom")
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    fig.savefig(FIG / "k_sweep_raw_vs_matched_gap.png", dpi=150)
    plt.close(fig)


def fig_strata(s35):
    names = [v for v, s in s35.items() if len(s["strata"]) == 2]
    ys = layout(s35, names)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 7.4), sharey=True)
    for ax, key, title in ((a, "raw_gap_seed_mean", "A. Raw gap by item entropy"), (b, "matched_gap_seed_mean", "B. Matched gap by item entropy")):
        style(ax)
        if ax is b:
            ax.axvspan(-TOL, TOL, color="#eef4fd", zorder=0)
        for v, y in ys.items():
            for name, col, mk, dy in (("consensus", BLUE, "o", 0.16), ("disputed", ORANGE, "s", -0.16)):
                st = s35[v]["strata"][name]
                if ax is b:
                    ax.plot([st["matched_gap_lo_seed_mean"], st["matched_gap_hi_seed_mean"]], [y + dy, y + dy], color=col, linewidth=2.0, zorder=3)
                ax.plot([st[key]], [y + dy], marker=mk, markersize=8, color=col, markeredgecolor="white", markeredgewidth=1.5, zorder=4, linestyle="none")
        ax.set_ylim(min(ys.values()) - 0.8, 0.8)
        ax.set_yticks(list(ys.values()))
        ax.set_yticklabels([f"{s35[v]['base']}  K={s35[v]['K']}" for v in ys], fontsize=9, color=TEXT1)
        ax.set_title(title, loc="left", fontsize=12, color=TEXT1, fontweight="bold")
        ax.set_xlabel("Test normalized ranked probability score (RPS) of the Pearson-trained head\nminus test normalized RPS of the CE-trained head\n(positive = CE-trained head better; lower RPS is better)"
                      + ("\ndot = mean over five split seeds; bar = mean of the five 95% item-clustered bootstrap intervals" if ax is b else "\ndot = mean over five split seeds; no interval drawn in this panel"), fontsize=9.5)
    handles = [plt.Line2D([], [], marker=m, color=c, markersize=8, markeredgecolor="white", markeredgewidth=1.2, linewidth=2.0, label=l)
               for c, m, l in ((BLUE, "o", "consensus items (half-A entropy at or below the median)"), (ORANGE, "s", "disputed items (above the median)"))]
    a.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("Entropy strata on held-out TEST items with four or more ratings; the stratum is chosen from half A of each item's ratings and scored on half B",
                 x=0.01, ha="left", fontsize=11, color=TEXT1)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.text(0.01, 0.008, "Both heads: linear softmax on 24 train-only principal components of a 256-dimensional token hash, ridge penalty fixed at 0.01 (preregistered, not tuned); matched = each head's mean score "
             "passes through its own validation-fitted calibration map.", fontsize=7.8, color=TEXT2)
    fig.savefig(FIG / "k_sweep_entropy_strata.png", dpi=150)
    plt.close(fig)


def fig_signal(s35, sig):
    ys = layout(s35, list(s35))
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    style(ax)
    for v, y in ys.items():
        s = s35[v]["constant_minus_ce_raw"]
        col = BLUE if sig[v] else GRAY_MARK
        ax.plot([s["seed_min"], s["seed_max"]], [y, y], color=col, linewidth=2.0, zorder=3)
        ax.plot([s["seed_mean"]], [y], marker="o", markersize=8, color=col, markeredgecolor="white", markeredgewidth=1.5, zorder=4, linestyle="none")
    ax.set_ylim(min(ys.values()) - 0.8, 0.8)
    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels([f"{s35[v]['base']}  K={s35[v]['K']}" for v in ys], fontsize=9, color=TEXT1)
    ax.set_xlabel("Test normalized ranked probability score (RPS) of the constant (train-marginal) forecast minus test normalized RPS of the CE-trained head\n"
                  "(positive = CE-trained head beats the constant; lower RPS is better; dot = mean, bar = range over five split seeds)", fontsize=9.5)
    ax.set_title("Which views carry any signal for a 24-component linear head? (blue = yes, gray = no)", loc="left", fontsize=12, color=TEXT1, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.01, 0.008, "CE-trained linear softmax head on 24 train-only principal components of a 256-dimensional token hash, ridge penalty fixed at 0.01 (preregistered, not tuned).",
             fontsize=7.8, color=TEXT2)
    fig.savefig(FIG / "k_sweep_signal_over_constant.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
def table_md(s35, s300, sig):
    L = ["| view | K | carries signal | constant nRPS | CE raw | CE matched | Pearson raw | Pearson matched | MSE raw | MSE matched | raw gap (Pearson minus CE) | matched gap (Pearson minus CE; range over seeds) | matched gap (MSE minus CE) | matched gap (Pearson minus CE) at cap 300 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, s in s35.items():
        lv = s["levels_test_nrps"]
        r, m = s["raw_pearson_minus_ce"], s["matched_pearson_minus_ce"]
        m3 = s300[v]["matched_pearson_minus_ce"]["seed_mean"]
        mm = s["matched_mse_minus_ce"]["seed_mean"]
        L.append(f"| {s['base']} | {s['K']} | {'yes' if sig[v] else 'no'} | {lv['constant']:.4f} | {lv['ce_raw']:.4f} | {lv['ce_matched']:.4f} | {lv['pearson_raw']:.4f} | "
                 f"{lv['pearson_matched']:.4f} | {lv['mse_raw']:.4f} | {lv['mse_matched']:.4f} | {r['seed_mean']:+.4f} | {m['seed_mean']:+.4f} ({m['seed_min']:+.4f} to {m['seed_max']:+.4f}) | {mm:+.4f} | {m3:+.4f} |")
    return "\n".join(L)


def main():
    d35, by35 = load(35)
    d300, by300 = load(300)
    s35, s300 = summarize(by35), summarize(by300)
    p, sig = score(s35, s300)
    consts = [s["levels_test_nrps"]["constant"] for s in s35.values()]
    fig_gap(s35, p, (min(consts), max(consts)))
    fig_strata(s35)
    fig_signal(s35, sig)
    payload = {"unavailable": d35.get("unavailable"), "n_views": len(s35), "signal_views_primary": [v for v, f in sig.items() if f],
               "scores": p, "summary_cap35": s35, "summary_cap300": s300}
    (RES / "k_sweep_summary.json").write_text(json.dumps(jsonable(payload), indent=1))
    (RES / "k_sweep_summary_table.md").write_text(
        "Test normalized ranked probability score (RPS; RPS divided by K - 1; 0 to 1; lower is better), mean over five split seeds, Experiment 41 K-sweep PREREGISTERED run "
        "(ridge penalty fixed at 0.01 for every head; at most 35 iterations). gap = the first-named head minus the CE-trained head (positive = CE-trained head better). "
        "Columns 'matched' pass each head's scalar mean score through its own validation-fitted multinomial-logit calibration map (same family for every head). "
        "MSE = the head trained by mean squared error on the scalar rating (it was trained in the preregistered run). The sensitivity run with the penalty chosen on validation is in "
        "k_sweep_l2_sensitivity_tables.md.\n\n" + table_md(s35, s300, sig) + "\n")
    print("views", len(s35), "unavailable", d35.get("unavailable"))
    for k, v in p.items():
        brief = {kk: vv for kk, vv in v.items() if kk not in ("values", "per_view", "signal_views")}
        print(k, "SUPPORTED" if v.get("supported", v.get("supported_literal")) else "NOT SUPPORTED", json.dumps(jsonable(brief))[:700])
    print("signal views:", [v for v, f in sig.items() if f])
    print("no-signal views:", [v for v, f in sig.items() if not f])


if __name__ == "__main__":
    main()
