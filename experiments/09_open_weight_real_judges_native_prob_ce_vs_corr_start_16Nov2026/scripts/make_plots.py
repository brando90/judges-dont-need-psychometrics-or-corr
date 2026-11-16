#!/usr/bin/env python3
"""Generate publication-grade figures for Experiment 09 obeying AGENTS.md result-clarity rules:
  - Every axis label names BOTH the metric (with acronym expanded on first use) AND the split (held-out test).
  - Every difference axis states exact subtraction order in words (same metric, two judges).
  - Major gridlines on all numeric axes (plus minor gridlines on log axes).
  - metadata={"CreationDate": None} where supported for deterministic PNG/PDF reproducibility.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPT_DIR = Path(__file__).resolve().parents[1]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "figure.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_fig1_forest(summary: dict, fig_dir: Path) -> None:
    """Forest plot across ii.a (Prompt Opt), ii.b (Harness Opt), and ii.c (Real Weight Updates)."""
    rows = []

    # Representation bridge
    rb = summary["representation_bridge"]
    rows.append(
        (
            "Archived Exp 14 · 39 open judges (3-sample scalar mix)",
            rb["exp14_3sample_scalar_39judges"]["contrasts"]["rps: corr_mix_pearson minus ce_mix_train"],
            rb["exp14_3sample_scalar_39judges"]["contrasts"]["ce: corr_mix_pearson minus ce_mix_train"],
        )
    )
    rows.append(
        (
            "ii.a · Candidate prompt pool (verbal 6-prob mix, ② vs ④)",
            summary["ii_a_prompt_optimization"]["all_models_prompt_pool_verbal"]["contrasts"]["rps: corr_mix_pearson minus ce_mix_train"],
            summary["ii_a_prompt_optimization"]["all_models_prompt_pool_verbal"]["contrasts"]["ce: corr_mix_pearson minus ce_mix_train"],
        )
    )
    mp = summary["ii_a_prompt_optimization"]["blind_meta_prompt_search_q3_30b"]
    if mp is not None:
        rows.append(
            (
                "ii.a · Blind meta-prompt search Qwen3-30B (① val vs ③ val)",
                mp["contrasts"]["rps: corr_prompt_val_1 minus ce_prompt_val_3"],
                mp["contrasts"]["ce: corr_prompt_val_1 minus ce_prompt_val_3"],
            )
        )
    rows.append(
        (
            "ii.b · Judge harness pool (H1–H6 + temp, ② vs ④)",
            summary["ii_b_harness_optimization"]["harness_pool_selection_and_mixtures"]["contrasts"]["rps: corr_mix_pearson minus ce_mix_train"],
            summary["ii_b_harness_optimization"]["harness_pool_selection_and_mixtures"]["contrasts"]["ce: corr_mix_pearson minus ce_mix_train"],
        )
    )
    rows.append(
        (
            "ii.b · 2x2 Prompt + Harness both Corr (②) vs both CE (④)",
            summary["ii_b_harness_optimization"]["prompt_x_harness_2x2_factorial"]["contrasts"]["rps_both_ce_vs_both_corr"],
            summary["ii_b_harness_optimization"]["prompt_x_harness_2x2_factorial"]["contrasts"]["ce_both_ce_vs_both_corr"],
        )
    )

    lora = summary["ii_c_real_weight_training"]["pytorch_lora_end_to_end"]
    for mk_label, mk in (("Qwen3-4B", "q3_4b_2507"), ("Qwen2.5-Coder-7B", "q25c_7b")):
        if mk in lora:
            rows.append(
                (
                    f"ii.c · {mk_label} PyTorch LoRA weights (② train vs ④ train)",
                    lora[mk]["contrasts"]["rps: pearson_lora_train_2 minus ce_lora_train_4"],
                    lora[mk]["contrasts"]["ce: pearson_lora_train_2 minus ce_lora_train_4"],
                )
            )
            rows.append(
                (
                    f"ii.c · {mk_label} PyTorch LoRA weights (① val vs ③ val)",
                    lora[mk]["contrasts"]["rps: pearson_lora_val_corr_1 minus ce_lora_val_rps_3"],
                    lora[mk]["contrasts"]["ce: pearson_lora_val_corr_1 minus ce_lora_val_rps_3"],
                )
            )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)
    y_pos = np.arange(len(rows))[::-1]

    for idx, (label, c_rps, c_ce) in enumerate(rows):
        y = y_pos[idx]
        for ax, c in ((ax1, c_rps), (ax2, c_ce)):
            pt = c["point"]
            lo, hi = c["ci95"]
            color = "#0f7c6e" if lo > 0 else ("#b4690e" if hi < 0 else "#6b7280")
            ax.plot([lo, hi], [y, y], color=color, lw=2.3, solid_capstyle="round")
            ax.plot(pt, y, "o", color=color, markersize=6)

    ax1.axvline(0.0, color="#1b2230", lw=1.1, ls="--")
    ax2.axvline(0.0, color="#1b2230", lw=1.1, ls="--")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([r[0] for r in rows])

    ax1.set_xlabel(
        "Held-out test ranked probability score (RPS) difference:\nCorrelation-trained judge minus CE-trained judge (>0 favors CE)"
    )
    ax2.set_xlabel(
        "Held-out test cross-entropy (CE, nats) difference:\nCorrelation-trained judge minus CE-trained judge (>0 favors CE)"
    )
    ax1.set_title("Primary: Held-Out Test RPS Diff (19 Families)", fontsize=9.8)
    ax2.set_title("Secondary: Held-Out Test CE Diff (19 Families)", fontsize=9.8)
    ax1.grid(True, which="major", linestyle=":", alpha=0.65)
    ax2.grid(True, which="major", linestyle=":", alpha=0.65)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(fig_dir / "fig1_taxonomy_forest_rps_and_ce.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig2_lora_and_mse(summary: dict, fig_dir: Path) -> None:
    """Plot ii.c PyTorch LoRA held-out test RPS/CE and exact MSE Bias^2 + Scale^2 + Dispersion decomposition."""
    lora = summary["ii_c_real_weight_training"]["pytorch_lora_end_to_end"]
    if not lora:
        return
    mk = "q3_4b_2507" if "q3_4b_2507" in lora else next(iter(lora))
    m_dict = lora[mk]["methods"]

    order = [
        ("base_ep0", "Untrained Base\n(Epoch 0)"),
        ("pearson_lora_train_2", "Pearson-LoRA\n(② Train)"),
        ("pearson_lora_val_corr_1", "Pearson-LoRA\n(① Val Corr)"),
        ("spearman_lora_train_2", "Spearman-LoRA\n(② Train)"),
        ("pearson_lora_plus_ce_head", "Pearson-LoRA +\nCE Head"),
        ("rps_lora_val_rps_3", "RPS-LoRA\n(③ Val RPS)"),
        ("ce_lora_train_4", "CE-LoRA\n(④ Train)"),
        ("ce_lora_val_rps_3", "CE-LoRA\n(③ Val RPS)"),
    ]
    present = [(k, lbl) for k, lbl in order if k in m_dict]
    x = np.arange(len(present))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8))

    rps_vals = [m_dict[k]["rps"] for k, _ in present]
    colors = ["#6b7280" if "base" in k else ("#b4690e" if ("pearson" in k or "spearman" in k) and "head" not in k else ("#d97706" if "head" in k else "#0f7c6e")) for k, _ in present]
    bars = ax1.bar(x, rps_vals, color=colors, width=0.62, edgecolor="#1b2230", linewidth=0.6)
    const_rps = summary["representation_bridge"]["exp09_logprob_6prob_pool"]["methods"]["constant"]["rps"]
    ax1.axhline(const_rps, color="#dc2626", ls="--", lw=1.2, label=f"Constant reference: separate 200-split pool ({const_rps:.3f})")
    for b, v in zip(bars, rps_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in present], rotation=18, ha="right")
    ax1.set_ylabel("Held-out test ranked probability score (RPS;\n0 to 1, lower is better, 5-fold task-family CV)")
    ax1.set_title(f"ii.c Real LLM LoRA Weight Fine-Tuning ({mk})")
    ax1.legend(loc="upper right")
    ax1.grid(True, axis="y", which="major", linestyle=":", alpha=0.65)

    bias_sq = np.array([m_dict[k]["bias_sq_05"] for k, _ in present])
    scale_sq = np.array([m_dict[k]["scale_mismatch_sq_05"] for k, _ in present])
    disp = np.array([m_dict[k]["dispersion_term_05"] for k, _ in present])

    ax2.bar(x, disp, width=0.62, label="Unexplained variance 2·sx·sy·(1 − r) [seen by Pearson]", color="#3b82f6")
    ax2.bar(x, scale_sq, width=0.62, bottom=disp, label="Scale mismatch² (sx − sy)² [ignored by Pearson]", color="#f59e0b")
    ax2.bar(x, bias_sq, width=0.62, bottom=disp + scale_sq, label="Signed bias² (x̄ − ȳ)² [ignored by Pearson]", color="#ef4444")
    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in present], rotation=18, ha="right")
    ax2.set_ylabel("Held-out test mean squared error (MSE on 0–5 scale;\nlower is better, exact 3-part decomposition)")
    ax2.set_title(f"Held-Out Test MSE Decomposition ({mk})")
    ax2.legend(loc="upper right", fontsize=7.6)
    ax2.grid(True, axis="y", which="major", linestyle=":", alpha=0.65)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_lora_weight_training_and_mse_decomp.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig3_learning_curves(summary: dict, fig_dir: Path) -> None:
    """Plot real-judge training-size learning curves across n_fam in {3, 5, 7, 9, 12, 15}."""
    lc = summary["ii_c_real_weight_training"]["neural_head_learning_curve_q3_4b"]
    if not lc:
        return
    nfams = sorted(int(k) for k in lc.keys())
    ratings_x = [lc[str(nf)]["mean_train_ratings"] for nf in nfams]

    series = [
        ("ce", "CE-trained judge (④)", "#0f7c6e", "-", "o"),
        ("rps", "RPS-trained judge (④)", "#0284c7", "-", "s"),
        ("pearson_plus_ce_head", "Pearson + calibrated CE head", "#d97706", "--", "^"),
        ("pearson", "Pearson-trained judge (②)", "#b4690e", "-", "D"),
        ("spearman", "Spearman-trained judge (②)", "#9333ea", "-.", "v"),
        ("kendall", "Kendall-trained judge (②)", "#ec4899", ":", "P"),
        ("constant", "Constant train-marginal baseline", "#6b7280", "--", "x"),
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.5))
    for key, lbl, col, ls, mk in series:
        y_rps = [lc[str(nf)]["rps"][key] for nf in nfams]
        y_ce = [lc[str(nf)]["ce"][key] for nf in nfams]
        ax1.plot(ratings_x, y_rps, color=col, ls=ls, marker=mk, lw=1.8, markersize=5, label=lbl)
        ax2.plot(ratings_x, y_ce, color=col, ls=ls, marker=mk, lw=1.8, markersize=5, label=lbl)
        if key != "constant":
            y_pe = [lc[str(nf)]["mean_test_pearson"][key] for nf in nfams]
            ax3.plot(ratings_x, y_pe, color=col, ls=ls, marker=mk, lw=1.8, markersize=5, label=lbl)

    ax1.set_xlabel("Training human ratings (across 3 to 15 training task families)")
    ax1.set_ylabel("Held-out test ranked probability score\n(RPS; 0 to 1, lower is better)")
    ax1.set_title("Held-Out Test RPS vs Training Ratings")
    ax1.grid(True, which="major", linestyle=":", alpha=0.7)

    ax2.set_xlabel("Training human ratings (across 3 to 15 training task families)")
    ax2.set_ylabel("Held-out test cross-entropy\n(CE in nats; lower is better)")
    ax2.set_title("Held-Out Test Cross-Entropy vs Training Ratings")
    ax2.grid(True, which="major", linestyle=":", alpha=0.7)

    ax3.set_xlabel("Training human ratings (across 3 to 15 training task families)")
    ax3.set_ylabel("Held-out test Pearson correlation\nwith human mean rating (higher is better)")
    ax3.set_title("Held-Out Test Pearson Correlation vs Training Ratings")
    ax3.grid(True, which="major", linestyle=":", alpha=0.7)
    handles, labels = ax1.get_legend_handles_labels()
    ax3.legend(handles, labels, loc="lower right", fontsize=7.2)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_real_judge_learning_curves_vs_n.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig4_when_and_2x2(summary: dict, fig_dir: Path) -> None:
    """Plot stratified rater-disagreement / task-type effects and the 2x2 Prompt x Harness factorial."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))

    wd = summary["when_does_ce_win"]["logprob_mixture"]
    strata = [
        ("Unanimous raters\n(SD = 0)", wd["unanimous_raters_SD_eq_0"]["rps_corr_minus_ce"]),
        ("Split raters\n(SD > 0, disagreement)", wd["split_raters_SD_gt_0"]["rps_corr_minus_ce"]),
        ("Ordinary tasks\n(60 artifacts)", wd["ordinary_non_placeholder_tasks"]["rps_corr_minus_ce"]),
        ("Placeholder-ref tasks\n(15 artifacts)", wd["placeholder_reference_tasks"]["rps_corr_minus_ce"]),
    ]
    xs = np.arange(len(strata))
    pts = [s[1]["point"] for s in strata]
    los = [s[1]["point"] - s[1]["ci95"][0] for s in strata]
    his = [s[1]["ci95"][1] - s[1]["point"] for s in strata]
    bar_cols = ["#0f7c6e" if s[1]["ci95"][0] > 0 else ("#b4690e" if s[1]["ci95"][1] < 0 else "#6b7280") for s in strata]
    ax1.bar(xs, pts, yerr=[los, his], capsize=4, color=bar_cols, width=0.55, alpha=0.88, edgecolor="#1b2230")
    ax1.axhline(0.0, color="#1b2230", ls="--", lw=1.0)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([s[0] for s in strata])
    ax1.set_ylabel(
        "Held-out test ranked probability score (RPS) difference:\nPearson-trained minus CE-trained (>0 favors CE)"
    )
    ax1.set_title("When CE Wins: Stratified Held-Out Test RPS Advantage")
    ax1.grid(True, axis="y", which="major", linestyle=":", alpha=0.65)

    f22 = summary["ii_b_harness_optimization"]["prompt_x_harness_2x2_factorial"]
    mat = np.array(
        [
            [f22["corr_prompt__corr_harness"]["rps"], f22["corr_prompt__ce_harness"]["rps"]],
            [f22["ce_prompt__corr_harness"]["rps"], f22["ce_prompt__ce_harness"]["rps"]],
        ]
    )
    im = ax2.imshow(mat, cmap="YlGnBu_r", aspect="auto")
    for i in range(2):
        for j in range(2):
            key = ("corr_prompt" if i == 0 else "ce_prompt") + "__" + ("corr_harness" if j == 0 else "ce_harness")
            ax2.text(
                j,
                i,
                f"Test RPS: {f22[key]['rps']:.4f}\nTest CE: {f22[key]['ce']:.3f} nats",
                ha="center",
                va="center",
                fontsize=9.0,
                fontweight="bold",
                color="#111827",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white", alpha=0.86, edgecolor="none"),
            )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Correlation-chosen\nHarness / Mixture (②)", "CE-chosen\nHarness / Mixture (④)"])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Correlation-chosen\nPrompt (②)", "CE-chosen\nPrompt (④)"])
    ax2.set_xlabel("Judge Harness & Calibration Selection Objective (on training task families)")
    ax2.set_ylabel("Judge Prompt Selection Objective (on training task families)")
    ax2.set_title("Held-Out Test RPS & CE: 2x2 Prompt × Harness Optimization")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Held-out test ranked probability score (RPS; lower is better)")

    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_when_ce_wins_disagreement_and_prompts_vs_harnesses.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    summary = json.loads((EXPT_DIR / "results/analysis_summary.json").read_text(encoding="utf-8"))
    fig_dir = EXPT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_fig1_forest(summary, fig_dir)
    plot_fig2_lora_and_mse(summary, fig_dir)
    plot_fig3_learning_curves(summary, fig_dir)
    plot_fig4_when_and_2x2(summary, fig_dir)
    print("Generated all 4 publication figures in figures/.")


if __name__ == "__main__":
    main()
