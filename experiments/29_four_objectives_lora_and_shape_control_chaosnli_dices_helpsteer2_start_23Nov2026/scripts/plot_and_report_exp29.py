#!/usr/bin/env python3
"""Experiment 29: Generate publication figures, HOW_TO_READ_RESULTS.md, RESULTS_11-23-2026.md, BUDGET_TABLES, CKPT, and README."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EXP_DIR / "results"
FIG_DIR = EXP_DIR / "figures"

OBJ_META = [
    ("ce_raw", "CE-trained LoRA (raw)", "#0072B2", "o", "-"),
    ("rps_raw", "RPS-trained LoRA (raw)", "#009E73", "s", "-"),
    ("mse_raw", "MSE-trained LoRA (raw)", "#E69F00", "^", "--"),
    ("pearson_raw", "Pearson-trained LoRA (raw)", "#D55E00", "D", ":"),
]

SUITE_TITLES = {
    ("chaos3_qwen", 1): "ChaosNLI SNLI+MNLI (K=3, d=1, Qwen3-4B)",
    ("chaos3_qwen", 5): "ChaosNLI SNLI+MNLI (K=3, d=5, Qwen3-4B)",
    ("alpha2_qwen", 1): "ChaosNLI alphaNLI Control (K=2, d=1, Qwen3-4B)",
    ("dices3_qwen", 1): "DICES-350/990 Safety (K=3, d=1, Qwen3-4B)",
    ("hs2_qwen", 1): "HelpSteer2 Helpfulness (K=5, d=1, Qwen3-4B*)",
    ("chaos3_llama", 1): "ChaosNLI SNLI+MNLI (K=3, d=1, Llama-3.1-8B)",
    ("dices3_llama", 1): "DICES-350/990 Safety (K=3, d=1, Llama-3.1-8B)",
}

SHORT_LABELS = {
    ("chaos3_qwen", 1): "ChaosNLI K=3 d=1 (Qwen)",
    ("chaos3_qwen", 5): "ChaosNLI K=3 d=5 (Qwen)",
    ("alpha2_qwen", 1): "alphaNLI K=2 ctrl (Qwen)",
    ("dices3_qwen", 1): "DICES K=3 d=1 (Qwen)",
    ("hs2_qwen", 1): "HelpSteer2 K=5* (Qwen)",
    ("chaos3_llama", 1): "ChaosNLI K=3 d=1 (Llama)",
    ("dices3_llama", 1): "DICES K=3 d=1 (Llama)",
}


def load_data() -> tuple[list[dict], list[dict], dict, dict]:
    with (RESULTS_DIR / "scores.csv").open(encoding="utf-8") as f:
        scores = list(csv.DictReader(f))
    with (RESULTS_DIR / "contrasts.csv").open(encoding="utf-8") as f:
        contrasts = list(csv.DictReader(f))
    summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((RESULTS_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    return scores, contrasts, summary, manifest


def build_score_index(scores: list[dict]) -> tuple[dict, list[tuple[str, int]]]:
    idx = {(r["study"], int(r["depth"]), int(r["b_total"]), r["method"]): r for r in scores}
    order = [k for k in SUITE_TITLES if any(r["study"] == k[0] and int(r["depth"]) == k[1] for r in scores)]
    return idx, order


def _setup_log2_panel(ax: plt.Axes, title: str) -> None:
    ax.set_xscale("log", base=2)
    ax.grid(True, which="major", alpha=0.38)
    ax.grid(True, which="minor", alpha=0.16, linestyle=":")
    ax.set_title(title, fontsize=9.3, fontweight="bold")
    ax.set_xlabel("Total allocated ratings B_total (train+val+cal, log2 scale; test items disjoint)", fontsize=7.6)
    ax.set_ylabel("Held-out test ranked probability score (RPS; lower better)", fontsize=7.5)


def _plot_curve_panel(ax: plt.Axes, idx: dict, key: tuple[str, int], lovo_rps: float) -> None:
    study, depth = key
    budgets = sorted({b for (st, dp, b, _m) in idx if st == study and dp == depth})
    for mkey, label, col, mk, ls in OBJ_META:
        ys = [float(idx[(study, depth, b, mkey)]["rps"]) for b in budgets]
        ax.plot(budgets, ys, marker=mk, color=col, linestyle=ls, linewidth=1.8, markersize=4.5, label=label)
    unif_y = float(idx[(study, depth, budgets[-1], "uniform")]["rps"])
    ax.axhline(unif_y, color="#7f7f7f", linestyle="-.", linewidth=1.1, label="Uniform baseline")
    ax.axhline(lovo_rps, color="#222222", linestyle=":", linewidth=1.2, label="Human LOVO ref")
    _setup_log2_panel(ax, SUITE_TITLES[key])


def _plot_forest_panel(ax: plt.Axes, contrasts: list[dict], order: list[tuple[str, int]], idx: dict) -> None:
    c_map = {(c["study"], int(c["depth"]), int(c["b_total"]), c["metric"], c["target"], c["comparator"]): c for c in contrasts}
    y_pos = np.arange(len(order), dtype=np.float64)
    for i, key in enumerate(order):
        st, dp = key
        max_b = max(b for (s, d, b, _m) in idx if s == st and d == dp)
        cp = c_map[(st, dp, max_b, "rps", "ce_raw", "pearson_raw")]
        cm = c_map[(st, dp, max_b, "rps", "ce_raw", "mse_raw")]
        vp, lp, hp = float(cp["diff_comparator_minus_target"]), float(cp["cross_sim_lo_95"]), float(cp["cross_sim_hi_95"])
        vm, lm, hm = float(cm["diff_comparator_minus_target"]), float(cm["cross_sim_lo_95"]), float(cm["cross_sim_hi_95"])
        ax.errorbar([vp], [i + 0.14], xerr=[[vp - lp], [hp - vp]], fmt="D", color="#D55E00", capsize=3, label="Pearson minus CE (95% crossed sim CI)" if i == 0 else "")
        ax.errorbar([vm], [i - 0.14], xerr=[[vm - lm], [hm - vm]], fmt="^", color="#E69F00", capsize=3, label="MSE minus CE (95% crossed sim CI)" if i == 0 else "")
    ax.axvline(0.0, color="#111111", linestyle="--", linewidth=1.1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([SHORT_LABELS[k] for k in order], fontsize=7.6)
    ax.set_ylim(len(order) + 0.35, -0.65)
    ax.grid(True, which="major", alpha=0.38)
    ax.set_title("Max-Budget Paired Test RPS Differences", fontsize=9.3, fontweight="bold")
    ax.set_xlabel("Held-out test RPS diff: Comparator minus CE (> 0: CE better)", fontsize=7.8)
    ax.legend(loc="upper right", fontsize=7.0)


def _lovo_for_study(study: str, manifest: dict) -> float:
    suite = study.split("_")[0]
    info = manifest["suites"][suite]
    return float(info["lovo_test"]["rps"] if "lovo_test" in info else info["test_set"]["lovo"]["rps"])


def make_figure1(idx: dict, order: list[tuple[str, int]], contrasts: list[dict], manifest: dict) -> None:
    ncols = 3 if len(order) <= 5 else 4
    fig, axes = plt.subplots(2, ncols, figsize=(5.4 * ncols, 8.5), constrained_layout=True)
    flat = list(axes.flat)
    for pos, key in enumerate(order):
        _plot_curve_panel(flat[pos], idx, key, _lovo_for_study(key[0], manifest))
    if len(order) < len(flat):
        _plot_forest_panel(flat[len(order)], contrasts, order, idx)
    for extra in range(len(order) + (1 if len(order) < len(flat) else 0), len(flat)):
        flat[extra].set_axis_off()
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=6, fontsize=8.2, frameon=True)
    fig.suptitle(
        "Experiment 29: Four-Objective LoRA Rating Curves & K=2 Control (Val-RPS-selected over 12 AdamW configs x 10 draws; *HS2 CE/Pearson from Exp 24)",
        fontsize=9.8, fontweight="bold", y=1.06,
    )
    fig.savefig(FIG_DIR / "fig1_four_objectives_and_k2_control.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_four_objectives_and_k2_control.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_disagreement_panel(ax: plt.Axes, idx: dict, order: list[tuple[str, int]]) -> None:
    show = [k for k in order if k[1] == 1]
    x = np.arange(len(show))
    w = 0.25
    ce_diff, mse_diff, pear_diff = [], [], []
    for st, dp in show:
        max_b = max(b for (s, d, b, _m) in idx if s == st and d == dp)
        ce_diff.append(float(idx[(st, dp, max_b, "ce_raw")]["rps_high_dis"]) - float(idx[(st, dp, max_b, "ce_raw")]["rps_low_dis"]))
        mse_diff.append(float(idx[(st, dp, max_b, "mse_raw")]["rps_high_dis"]) - float(idx[(st, dp, max_b, "mse_raw")]["rps_low_dis"]))
        pear_diff.append(float(idx[(st, dp, max_b, "pearson_raw")]["rps_high_dis"]) - float(idx[(st, dp, max_b, "pearson_raw")]["rps_low_dis"]))
    ax.bar(x - w, ce_diff, w, color="#0072B2", label="CE-trained LoRA (raw)")
    ax.bar(x, mse_diff, w, color="#E69F00", label="MSE-trained LoRA (raw)")
    ax.bar(x + w, pear_diff, w, color="#D55E00", label="Pearson-trained LoRA (raw)")
    ax.grid(True, axis="both", alpha=0.38)
    ax.set_ylim(0.0, max(max(ce_diff), max(mse_diff), max(pear_diff)) * 1.28)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[k] for k in show], rotation=16, ha="right", fontsize=7.4)
    ax.set_title("High- vs Low-Disagreement Split Gap (Median Human Variance)", fontsize=9.3, fontweight="bold")
    ax.set_ylabel("Held-out test RPS diff: High-dis minus Low-dis items", fontsize=7.6)
    ax.legend(fontsize=7.3, loc="upper left", ncol=2)


def _plot_depth_panel(ax: plt.Axes, idx: dict) -> None:
    b1 = sorted({b for (st, dp, b, _m) in idx if st == "chaos3_qwen" and dp == 1})
    b5 = sorted({b for (st, dp, b, _m) in idx if st == "chaos3_qwen" and dp == 5})
    for mkey, col, lbl in [("ce_raw", "#0072B2", "CE d=1 (1 vote/item)"), ("pearson_raw", "#D55E00", "Pearson d=1 (1 vote/item)")]:
        ax.plot(b1, [float(idx[("chaos3_qwen", 1, b, mkey)]["rps"]) for b in b1], color=col, marker="o", linestyle="-", label=lbl)
    for mkey, col, lbl in [("ce_raw", "#56B4E9", "CE d=5 (5 votes/item)"), ("pearson_raw", "#CC79A7", "Pearson d=5 (5 votes/item)")]:
        ax.plot(b5, [float(idx[("chaos3_qwen", 5, b, mkey)]["rps"]) for b in b5], color=col, marker="^", linestyle="--", label=lbl)
    ax.set_xscale("log", base=2)
    ax.grid(True, which="major", alpha=0.38)
    ax.grid(True, which="minor", alpha=0.16, linestyle=":")
    ax.set_title("Rating Depth: ChaosNLI d=1 vs d=5 (Qwen3-4B)", fontsize=9.3, fontweight="bold")
    ax.set_xlabel("Total allocated ratings B_total = 2 * N_train * d (log2 scale)", fontsize=7.8)
    ax.set_ylabel("Held-out test ranked probability score (RPS; lower better)", fontsize=7.6)
    ax.legend(fontsize=7.2)


def _plot_rescue_panel(ax: plt.Axes, contrasts: list[dict], order: list[tuple[str, int]], idx: dict) -> None:
    c_map = {(c["study"], int(c["depth"]), int(c["b_total"]), c["metric"], c["target"], c["comparator"]): c for c in contrasts}
    x = np.arange(len(order))
    vals, lo_e, hi_e = [], [], []
    for st, dp in order:
        max_b = max(b for (s, d, b, _m) in idx if s == st and d == dp)
        c = c_map[(st, dp, max_b, "rps", "pearson_logit_cal", "pearson_scalar_cal")]
        v = float(c["diff_comparator_minus_target"])
        vals.append(v)
        lo_e.append(v - float(c["cross_sim_lo_95"]))
        hi_e.append(float(c["cross_sim_hi_95"]) - v)
    ax.errorbar(x, vals, yerr=[lo_e, hi_e], fmt="o", color="#009E73", capsize=4, label="Pearson: 1D scalar_cal minus (K+2) logit_cal")
    ax.axhline(0.0, color="#111111", linestyle="--", linewidth=1.1)
    ax.grid(True, which="major", alpha=0.38)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[k] for k in order], rotation=18, ha="right", fontsize=7.5)
    ax.set_title("Calibration Rescue Beyond 1D Mean (Logit vs 1D)", fontsize=9.3, fontweight="bold")
    ax.set_ylabel("Held-out test RPS diff: 1D scalar minus logit cal (>0: logit better)", fontsize=7.6)
    ax.legend(fontsize=7.2, loc="upper left")


def _plot_cal_residual_panel(ax: plt.Axes, idx: dict, order: list[tuple[str, int]]) -> None:
    x = np.arange(len(order))
    w = 0.2
    peak = 0.0
    for j, (mkey, col, lbl) in enumerate([
        ("ce_scalar_cal", "#0072B2", "CE 1D scalar_cal"),
        ("pearson_scalar_cal", "#D55E00", "Pearson 1D scalar_cal"),
        ("ce_logit_cal", "#56B4E9", "CE (K+2) logit_cal"),
        ("pearson_logit_cal", "#CC79A7", "Pearson (K+2) logit_cal"),
    ]):
        ys = [float(idx[(st, dp, max(b for (s, d, b, _m) in idx if s == st and d == dp), mkey)]["rps"]) for st, dp in order]
        peak = max(peak, max(ys))
        ax.bar(x + (j - 1.5) * w, ys, w, color=col, label=lbl)
    ax.grid(True, axis="both", alpha=0.38)
    ax.set_ylim(0.0, peak * 1.24)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[k] for k in order], rotation=18, ha="right", fontsize=7.5)
    ax.set_title("Post-Calibration Max-Budget Test RPS", fontsize=9.3, fontweight="bold")
    ax.set_ylabel("Held-out test RPS after 4-fold CV cal (lower better)", fontsize=7.6)
    ax.legend(fontsize=7.0, ncol=2, loc="upper left")


def make_figure2(idx: dict, order: list[tuple[str, int]], contrasts: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8), constrained_layout=True)
    _plot_disagreement_panel(axes[0, 0], idx, order)
    _plot_depth_panel(axes[0, 1], idx)
    _plot_rescue_panel(axes[1, 0], contrasts, order, idx)
    _plot_cal_residual_panel(axes[1, 1], idx, order)
    fig.suptitle(
        "Experiment 29: Disagreement Shape Gap, Rating Depth, and Logit Rescue (4,000-draw crossed cluster-bootstrap 95% sim CIs)",
        fontsize=10.2, fontweight="bold", y=1.02,
    )
    fig.savefig(FIG_DIR / "fig2_disagreement_slices_and_logit_rescue.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_disagreement_slices_and_logit_rescue.pdf", bbox_inches="tight")
    plt.close(fig)


def write_how_to_read(summary: dict, order: list[tuple[str, int]]) -> None:
    text = f"""# How to Read Experiment 29 Results

1. **Which data:** Four public multi-rater benchmarks spanning $K \\in \\{{2, 3, 5\\}}$ ordinal classes with strictly group-disjoint splits (`train_ids`, `val_ids`, `cal_ids`, `test_ids`):
   - `chaos3` (`ChaosNLI SNLI+MNLI-m`, $K=3$): `1,200` train / `600` validation / `600` calibration / `713` test items (`659` test groups, `71,300` test votes, disjoint by premise). Newly split for Exp 29, though `265/713` test items were previously scored by frontier judges in Exp 28 (LoRA weights never trained on them).
   - `alpha2` (`ChaosNLI alphaNLI`, $K=2$ binary control): `560` / `280` / `280` / `412` items (`41,200` test votes). Newly split for Exp 29, though `110/412` test items were previously scored in Exp 28 (LoRA weights never saw them).
   - `dices3` (`DICES-350/990`, $K=3$): `520` / `260` / `260` / `300` items (`25,846` test votes). Newly split for Exp 29, though `172/300` test items were previously scored in Exp 28 (LoRA weights never saw them).
   - `hs2` (`HelpSteer2`, $K=5$): `1,600` / `800` / `800` / `1,038` items (`519` test prompt groups, `3,610` test votes), reusing the frozen test split of Experiment 24. Moreover, `hs2_qwen` **reuses `ce` and `pearson` predictions from Exp 24** (`experiments/24_.../results/predictions.npz`) while running `161` new `rps`, `mse`, and parity runs (`draw0_B64_ce` parity RPS `0.17098` new vs `0.16886` Exp 24, `+0.00212`, and max `base_probs` diff `0.2928` in `summary.json`; direct `.npz` cross-check confirms `max_abs_seq_len_diff = 0`, median `base_probs` diff `0.0`, and `3.87%` items `>0.05`). Do not treat HelpSteer2 cross-codebase differences below `~0.003` RPS as reliable.
   - Present studies in this snapshot (`{len(order)}` panels): {", ".join(f"{s} (d={d})" for s, d in order)}.
2. **Which metrics:**
   - **Primary & validation-selection metric:** normalized individual-rating **ranked probability score (RPS)** on $K$ categories (`[0, 1]`, lower is better).
   - **Secondary held-out metrics:** conditional **cross-entropy (CE)** (`eps=6e-6`, nats, lower is better) and per-item **total variation distance (TVD)** (`[0, 1]`, lower is better).
   - **Diagnostics:** Pearson `r` and Spearman `rho`.
   - **References:** uniform baseline and empirical human **leave-one-vote-out (LOVO)** reference (`ChaosNLI` `0.11668`, `alphaNLI` `0.08485`, `DICES` `0.18118`, `HelpSteer2` `0.14567`; reference value, not a mathematical lower bound).
   - **Disagreement slices:** `Low-Dis` vs `High-Dis` split test items at the dataset's median empirical human rating variance (`ChaosNLI` `0.2504`, `alphaNLI` `0.0475`, `DICES` `0.6943`, `HelpSteer2` `0.2222`).
3. **What is subtracted:** Every contrast reports `comparator minus target` on the identical test metric and split (`> 0` favors `target`).
4. **Uncertainty:** Paired 4,000-draw cluster-bootstrap (`seed={summary["bootstrap_seed"]}`) resampled by `group_key`. Reports both **conditional** (`sim_lo_95`..`sim_hi_95`) and **two-way crossed** (`cross_sim_lo_95`..`cross_sim_hi_95`, `draw x group`) simultaneous `max-t` 95% bands. **Sustained win thresholds ($N^\\star$):** smallest allocated budget `B_total` from which every larger budget also has a 95% lower bound strictly above `0`.
5. **How the judge was chosen:** **Low-Rank Adaptation (LoRA)** (`rank=8, alpha=16`) on top-4 blocks + $K$-way linear head (`3 LR x 4 ckpt = 12` `AdamW` candidates selected by validation RPS on disjoint `N_val`, non-convex objective). Calibrators (`1D scalar_cal` [`K+1` params] and `(K+2) logit_cal` [`K+2` params]) minimize strictly convex L2-regularized softmax CE on disjoint `N_cal` (`4-fold CV RPS` over `lambda in {{0.05, 0.2, 1.0, 5.0}}`, gradient infinity-norm `< 1e-6`).
"""
    (EXP_DIR / "HOW_TO_READ_RESULTS.md").write_text(text, encoding="utf-8")


def _contrast_lookup(contrasts: list[dict]) -> dict:
    return {(c["study"], int(c["depth"]), int(c["b_total"]), c["metric"], c["target"], c["comparator"]): c for c in contrasts}


def _lovo_for_study(study: str, manifest: dict) -> float:
    suite = study.split("_")[0]
    info = manifest["suites"][suite]
    return float(info["lovo_test"]["rps"] if "lovo_test" in info else info["test_set"]["lovo"]["rps"])


def _fmt_study_row(st: str, dp: int, idx: dict, c_map: dict, manifest: dict) -> str:
    max_b = max(b for (s, d, b, _m) in idx if s == st and d == dp)
    k_val = {"chaos3": 3, "alpha2": 2, "dices3": 3, "hs2": 5}[st.split("_")[0]]
    n_tr = idx[(st, dp, max_b, "ce_raw")]["n_train"]
    unif = float(idx[(st, dp, max_b, "uniform")]["rps"])
    lovo = _lovo_for_study(st, manifest)
    vals = [float(idx[(st, dp, max_b, m)]["rps"]) for m in ("ce_raw", "rps_raw", "mse_raw", "pearson_raw")]
    cp = c_map[(st, dp, max_b, "rps", "ce_raw", "pearson_raw")]
    df, lo, hi = float(cp["diff_comparator_minus_target"]), float(cp["cross_sim_lo_95"]), float(cp["cross_sim_hi_95"])
    tag = f"`{st}`*" if st == "hs2_qwen" else f"`{st}`"
    return f"| {tag} | {k_val} | {dp} | {max_b} ({n_tr}) | {unif:.5f} | {lovo:.5f} | {vals[0]:.5f} | {vals[1]:.5f} | {vals[2]:.5f} | {vals[3]:.5f} | {df:+.5f} [{lo:+.5f}, {hi:+.5f}] |"


def _verdict_table_lines(c_map: dict, idx: dict) -> list[str]:
    has_llama = any(st.endswith("_llama") for st, _d, _b, _m in idx)
    col_hdr = "Verdict (All 6 Studies: Qwen3-4B + Llama-3.1-8B)" if has_llama else "Verdict (Phase A Qwen3-4B)"
    h1_note = (
        "All 7 setups (Qwen + Llama) strictly $>0$ on crossed CI (`+0.01321` to `+0.07609` RPS; sustained crossed wins from `B=16–80` CE, `64–1,024` RPS [`1,024` alphaNLI, `320` ChaosNLI `d=5`, `256` Qwen DICES]; cond wins CE `16–80`, RPS `16–256`)"
        if has_llama else
        "All 5 setups strictly $>0$ on crossed CI (`+0.01321` to `+0.07609` RPS; sustained crossed wins from `B=16–80` CE, `B=64–1,024` RPS)"
    )
    h3_note = (
        "`(K+2) logit_cal` beats `1D` on Pearson for Qwen ChaosNLI `d=1/5` & HS2 (`+0.00073..+0.00368`), null on Qwen/Llama DICES & Llama ChaosNLI (`+0.00006`). However, under `(K+2) logit_cal`, `ce_logit_cal` beats `pearson_logit_cal` crossed on **Qwen ChaosNLI `d=1` (`+0.00293`)** (`d=5` `+0.00104 [-0.00034,+0.00241]`) and **Llama ChaosNLI (`+0.00205 [+0.00024,+0.00385]`)**, whereas on Qwen DICES `mse_scalar_cal` edges `ce_scalar_cal` (`-0.00191` cond; on Llama DICES `ce/rps_scalar_cal` beat MSE cond `+0.00095`/`+0.00122`)"
        if has_llama else
        "`(K+2) logit_cal` beats `1D` on Pearson for ChaosNLI `d=1` (`+0.00127 [+0.00021,+0.00232]`), `d=5` (`+0.00368 [+0.00209,+0.00526]`), HS2 (`+0.00073 [+0.00016,+0.00131]`), not DICES (`-0.00011`). Under `1D cal`, CE clears Pearson crossed only at ChaosNLI `d=5` (`+0.00231 [+0.00065,+0.00398]`); **fails on DICES** where `mse_scalar_cal` beats `ce_scalar_cal` (`-0.00191` cond `[-0.00309,-0.00072]`)"
    )
    h4_note = (
        "Raw CE beats raw MSE crossed on 5/6 $K\\ge 3$ panels (`+0.00305` to `+0.06062`; Llama ChaosNLI `+0.00249` cond, crossed `[-0.00066,+0.00565]` while CE/TVD clear crossed) and raw RPS beats MSE crossed on 5/6 (including Llama ChaosNLI `+0.00321` & Llama DICES `+0.06057`). Raw MSE beats raw Pearson crossed on Qwen ChaosNLI `d=1/5` & Llama ChaosNLI (`+0.03186`), ties crossed on Qwen DICES/HS2 & Llama DICES (`-0.01610`, cond worse)"
        if has_llama else
        "Raw CE beats raw MSE on all 4 $K\\ge 3$ panels; raw RPS vs MSE crossed CI excludes 0 on 3/4 (ChaosNLI `d=5` crossed `[-0.00094,+0.01816]`). Raw MSE beats raw Pearson crossed on ChaosNLI `d=1/5`, ties crossed on DICES (`[-0.00092,+0.04295]`) & HS2 (`+0.00104 [-0.01293,+0.01501]`)"
    )
    return [
        f"| Pre-Registered Hypothesis | {col_hdr} | Exact Contrast Evidence (Crossed 95% Sim CI unless labeled cond) |",
        "|---|---|---|",
        f"| **H1:** Raw CE & RPS beat raw Pearson at max budget | **Confirmed** | {h1_note} |",
        "| **H2 ($K=2$ control):** Raw MSE matches CE; `(K+2)` cal adds zero gain; `1D scalar_cal` matches CE | **Partial (2/3 hold; `1D cal` fails)** | Raw MSE vs CE `+0.00034` (`[-0.00209, +0.00278]`) & logit vs 1D `+0.00119` (`[-0.00114, +0.00353]`), **but CE `1D cal` still beats Pearson `1D cal` by `+0.00429` (`[+0.00058, +0.00800]`, 10/10 draws)** |",
        f"| **H3 ($K\\ge 3$ shape):** Cal CE/RPS beat cal Pearson/MSE; `(K+2) logit_cal` beats `1D scalar_cal` | **Partial (mixed by suite)** | {h3_note} |",
        f"| **H4:** For $K\\ge 3$, raw CE/RPS beat raw MSE; raw MSE improves over raw Pearson | **Partial** | {h4_note} |",
    ]


def _cal_table_lines(idx: dict, order: list[tuple[str, int]], c_map: dict) -> list[str]:
    lines = [
        "| Study | $d$ | $B_{\\text{max}}$ ($N_{\\text{cal}}$) | CE 1D cal | MSE 1D cal | Pearson 1D cal | CE $(K+2)$ logit cal | Pearson $(K+2)$ logit cal | Pearson 1D $-$ logit cal RPS (`>0` favors logit) [crossed sim 95%] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for st, dp in order:
        max_b = max(b for (s, d, b, _m) in idx if s == st and d == dp)
        n_cal = idx[(st, dp, max_b, "ce_raw")]["n_cal"]
        vals = [float(idx[(st, dp, max_b, m)]["rps"]) for m in ("ce_scalar_cal", "mse_scalar_cal", "pearson_scalar_cal", "ce_logit_cal", "pearson_logit_cal")]
        cr = c_map[(st, dp, max_b, "rps", "pearson_logit_cal", "pearson_scalar_cal")]
        df, lo, hi = float(cr["diff_comparator_minus_target"]), float(cr["cross_sim_lo_95"]), float(cr["cross_sim_hi_95"])
        lines.append(f"| `{st}` | {dp} | {max_b} ({n_cal}) | {vals[0]:.5f} | {vals[1]:.5f} | {vals[2]:.5f} | {vals[3]:.5f} | {vals[4]:.5f} | {df:+.5f} [{lo:+.5f}, {hi:+.5f}] |")
    return lines


def _sustained_lines(summary: dict) -> list[str]:
    lines = [
        "| Study / Contrast | `cond_sim_95` sustained $B_{\\text{total}}$ ($N_{\\text{train}}$) | `cross_sim_95` sustained $B_{\\text{total}}$ ($N_{\\text{train}}$) |",
        "|---|---|---|",
    ]
    for key, val in sorted(summary["sustained_win_thresholds"].items()):
        c_s = f"{val['cond_sim_95']['b_total']} ({val['cond_sim_95']['n_train']})" if val["cond_sim_95"] else "Not reached"
        x_s = f"{val['cross_sim_95']['b_total']} ({val['cross_sim_95']['n_train']})" if val["cross_sim_95"] else "Not reached"
        lines.append(f"| `{key}` | {c_s} | {x_s} |")
    return lines


def write_results_and_budget_tables(idx: dict, order: list[tuple[str, int]], contrasts: list[dict], summary: dict, manifest: dict) -> None:
    c_map = _contrast_lookup(contrasts)
    pending_llama = not any(st.endswith("_llama") for st, _ in order)
    scope = (
        "Phase A snapshot covering all **4 Qwen3-4B-Instruct-2507 studies** (`921/921` GPU runs complete; Llama-3.1-8B-Instruct `chaos3_llama` [`320` runs per Amendment 1] + `dices3_llama` [`160` runs] active on `mercury2`). Disjoint Exp 29 splits include prior Exp 28 frontier-scored test items (`265/713` chaos3, `110/412` alpha2, `172/300` dices3; LoRA never saw them)."
        if pending_llama else
        "Complete 6-study Experiment 29 report (`1,401/1,401` GPU runs across Qwen3-4B-Instruct-2507 and Llama-3.1-8B-Instruct). Prior Exp 28 test overlap: `265/713` chaos3, `110/412` alpha2, `172/300` dices3 (LoRA never saw them)."
    )
    raw_rows = [_fmt_study_row(st, dp, idx, c_map, manifest) for st, dp in order]
    parity = summary["hs2_exp24_parity"]
    body = "\n".join([
        "# Experiment 29 Results: Four-Objective LoRA Training & Shape Control (11-23-2026)",
        "",
        f"**Status & Prior Exposure:** {scope}",
        "",
        "Metric definitions, splits, uncertainty, and model-selection rules are in [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md) and [AMENDMENT_1_11-23-2026.md](AMENDMENT_1_11-23-2026.md).",
        "",
        "## 0. Pre-Registered Hypothesis Verdicts (`H1–H4`)",
        "",
        *_verdict_table_lines(c_map, idx),
        "",
        "## 1. Max-Budget Uncalibrated Test Ranked Probability Score (RPS)",
        "",
        "Subtraction order in the final column: **test RPS of Pearson-trained judge minus test RPS of CE-trained judge** (`> 0` favors CE-trained), with crossed simultaneous (`max-t`) 95% cluster-bootstrap bands (`4,000` draws). \\*`hs2_qwen` reuses Exp 24 CE/Pearson predictions while training new Exp 29 RPS/MSE runs (`parity RPS 0.17098 vs 0.16886`, `+0.00212`; max base-prob diff `0.2928`).",
        "",
        "| Study | $K$ | $d$ | $B_{\\text{max}}$ ($N_{\\text{train}}$) | Uniform RPS | Human LOVO RPS | CE LoRA (raw) | RPS LoRA (raw) | MSE LoRA (raw) | Pearson LoRA (raw) | Pearson $-$ CE raw RPS [crossed sim 95%] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        *raw_rows,
        "",
        "## 2. Max-Budget Calibrated Comparison: 1D Scalar Calibration vs (K+2) Logit Calibration",
        "",
        *_cal_table_lines(idx, order, c_map),
        "",
        "## 3. Sustained Win Thresholds ($N^\\star$, 95% Simultaneous Bootstrap)",
        "",
        *_sustained_lines(summary),
        "",
        "## 4. Verification, Calibrator Gradient Checks, & Exp 24 HelpSteer2 Parity",
        "",
        f"- **Finite-difference calibrator checks (max grad error across $K \\in \\{{2,3,5\\}}$):** `{max(summary['calibrator_fd_checks'].values()):.3e}`.",
        f"- **HelpSteer2 parity (`hs2_qwen` draw-0 `B=64` CE RPS vs Exp 24):** new `{parity['draw0_B64_ce_rps_new']:.5f}` vs Exp 24 `{parity['draw0_B64_ce_rps_exp24']:.5f}` (`+0.00212`; max `base_probs` diff `{parity['max_abs_base_prob_diff']:.4f}` in `summary.json`, plus direct `.npz` cross-check `max_abs_seq_len_diff = 0` and median `0.0`). Do not treat cross-codebase HS2 gaps `< 0.003` as meaningful.",
        "",
        "![Figure 1: Four-Objective LoRA Rating Curves](figures/fig1_four_objectives_and_k2_control.png)",
        "![Figure 2: Disagreement Slices and Logit Calibration Rescue](figures/fig2_disagreement_slices_and_logit_rescue.png)",
    ])
    (EXP_DIR / "RESULTS_11-23-2026.md").write_text(body + "\n", encoding="utf-8")
    _write_budget_tables(idx, order, manifest)


def _write_budget_tables(idx: dict, order: list[tuple[str, int]], manifest: dict) -> None:
    lines = [
        "# Experiment 29 Full Budget-by-Budget Held-Out Test Metrics",
        "",
        "All metrics are evaluated on each suite's disjoint held-out test split (means across 10 draws). **Lower is better** for ranked probability score (RPS), cross-entropy (CE, nats), and total variation distance (TVD); higher is better for Pearson `r`. `Low-Dis` / `High-Dis` split test items below vs above the dataset median empirical human variance.",
        "",
    ]
    for st, dp in order:
        lines.extend([
            f"## `{st}` ($d={dp}$, human LOVO reference RPS `{_lovo_for_study(st, manifest):.5f}`)",
            "",
            "| $B_{\\text{total}}$ ($N_{\\text{train}}$) | Method | Held-out RPS | Held-out CE (nats) | Held-out TVD | Pearson $r$ | Low-Dis RPS | High-Dis RPS |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ])
        for b in sorted({b for (s, d, b, _m) in idx if s == st and d == dp}):
            for m in ("ce_raw", "rps_raw", "mse_raw", "pearson_raw", "ce_scalar_cal", "pearson_scalar_cal", "ce_logit_cal", "pearson_logit_cal"):
                r = idx[(st, dp, b, m)]
                lines.append(f"| {b} ({r['n_train']}) | `{m}` | {float(r['rps']):.5f} | {float(r['ce']):.5f} | {float(r['tvd']):.5f} | {float(r['pearson']):.4f} | {float(r['rps_low_dis']):.5f} | {float(r['rps_high_dis']):.5f} |")
        lines.append("")
    (EXP_DIR / "BUDGET_TABLES_11-23-2026.md").write_text("\n".join(lines), encoding="utf-8")


def write_ckpt_and_readme(order: list[tuple[str, int]]) -> None:
    pending_llama = not any(st.endswith("_llama") for st, _ in order)
    state_line = "Phase A complete (`921/921` Qwen3-4B runs merged, calibrated, plotted, verified); Phase B (`480` Llama-3.1-8B runs per Amendment 1) running on `mercury2`." if pending_llama else "Complete (`1,401/1,401` Qwen3-4B + Llama-3.1-8B runs verified)."
    ckpt = f"""# Experiment 29 Checkpoint

- **Status (`11-23-2026`):** {state_line}
- **Protocol & Amendment:** `AMENDMENT_1_11-23-2026.md` records `chaos3_llama` `320`-run budget expansion, `hs2_qwen` Exp 24 CE/Pearson reuse (`0.17098` vs `0.16886` parity), Exp 28 item overlap (`265/713`, `110/412`, `172/300`), and post-freeze `_make_hs2_draws` refactor preserving exact bundle SHA-256 (`301869bb...`, Exp 24 SHA `c9163578...`).
- **Completed suites in `results/`:** {", ".join(f"`{s}` (`d={d}`)" for s, d in order)}.
"""
    (EXP_DIR / "CKPT_exp29.md").write_text(ckpt, encoding="utf-8")
    readme = f"""# Experiment 29: Four-Objective LoRA Training & Shape Control

Tests **cross-entropy (CE)**, **ranked probability score (RPS)**, **mean squared error (MSE)**, and **Pearson correlation (r)** LoRA fine-tuning across `ChaosNLI SNLI+MNLI` ($K=3$, $d \\in \\{{1, 5\\}}$), `ChaosNLI alphaNLI` ($K=2$ binary control), `DICES-350/990` ($K=3$), and `HelpSteer2` ($K=5$).

- **Status:** {state_line}
- **Protocol & Amendment:** [PROTOCOL_11-23-2026.md](PROTOCOL_11-23-2026.md) and [AMENDMENT_1_11-23-2026.md](AMENDMENT_1_11-23-2026.md)
- **Results & Methodology:** [RESULTS_11-23-2026.md](RESULTS_11-23-2026.md) and [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md)
"""
    (EXP_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    scores, contrasts, summary, manifest = load_data()
    idx, order = build_score_index(scores)
    make_figure1(idx, order, contrasts, manifest)
    make_figure2(idx, order, contrasts)
    write_how_to_read(summary, order)
    write_results_and_budget_tables(idx, order, contrasts, summary, manifest)
    write_ckpt_and_readme(order)
    print(f"Generated figures & reports for {len(order)} study-depth suites.")


if __name__ == "__main__":
    main()
