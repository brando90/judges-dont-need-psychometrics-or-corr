"""Publication figures. Every numeric axis has gridlines. Axis text names the metric and the split."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

CROWD_USD = 0.05
EXPERT_USD = 5.0
METRICS = [
    ("ce", "Held-out test cross-entropy (nats)", True),
    ("sqrt_rps", "Held-out test sqrt(ranked probability score)", True),
    ("brier", "Held-out test Brier score", True),
    ("pearson_r", "Held-out test Pearson r", False),
    ("spearman_rho", "Held-out test Spearman rho", False),
    ("kendall_tau", "Held-out test Kendall tau", False),
]
ROWS = ["ce", "brier", "rps", "pearson", "mse"]


def _style(ax):
    ax.grid(True, which="major", linewidth=0.6, alpha=0.7)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.35)
    ax.set_axisbelow(True)


def _finite(v):
    if v is None:
        return np.nan
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(v):
        return np.nan
    return v


def make_all(payload, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    names_main = [n for n in ("summeval_expert", "summeval_expert_binary", "chaosnli_alpha", "dices990") if n in payload["matrices"]]
    if not names_main:
        names_main = list(payload["matrices"])[:2]
    _matrix(payload, names_main, fig_dir / "wrong_objective_matrix.png")
    names_new = [
        n for n in ("popquorn_offensiveness", "popquorn_politeness", "veribench_fsc_rv1", "veribench_fsc_native_main", "veribench_fsc_pooled")
        if n in payload["matrices"]
    ]
    if names_new:
        _matrix(payload, names_new, fig_dir / "wrong_objective_matrix_popquorn_veribench.png")
    _crossover(payload, fig_dir / "finite_sample_crossover_binary_vs_ordinal.png")
    _high_k(payload, fig_dir / "high_k_crossover_dices_chaos.png")
    _crossover_extra(payload, fig_dir / "crossover_popquorn_veribench.png")
    _certjudge_panel(payload, fig_dir / "certjudge_real_panels_proper_vs_corr.png")
    _rollout(payload, fig_dir / "single_pass_vs_rollout.png")
    _iso(payload, fig_dir / "iso_budget_chaosnli.png")


def _matrix(payload, names, path):
    fig, axes = plt.subplots(1, len(names), figsize=(7.5 * len(names), 5.0), squeeze=False)
    for ax, name in zip(axes[0], names):
        block = payload["matrices"][name]
        objs = [o for o in ROWS if o in block["rows"]]
        grid = np.full((len(objs), len(METRICS)), np.nan)
        text = [["" for _ in METRICS] for _ in objs]
        for i, obj in enumerate(objs):
            point = block["rows"][obj]["point"]
            ci = block["rows"][obj]["ci"]
            for j, (key, _label, _low) in enumerate(METRICS):
                val = _finite(point.get(key))
                grid[i, j] = val
                lo = _finite(ci.get(key, {}).get("lo"))
                hi = _finite(ci.get(key, {}).get("hi"))
                raw = point.get(key)
                if raw == "inf" or raw == float("inf"):
                    text[i][j] = "inf"
                elif np.isfinite(val):
                    text[i][j] = f"{val:.3f}\n[{lo:.2f},{hi:.2f}]" if np.isfinite(lo) else f"{val:.3f}"
                else:
                    text[i][j] = "undef"
        # Color by within-column rank. Rank 0 is best.
        rank = np.zeros_like(grid)
        for j, (_k, _l, lower_better) in enumerate(METRICS):
            col = grid[:, j]
            order = np.argsort(np.where(np.isfinite(col), col, np.inf if lower_better else -np.inf))
            if not lower_better:
                order = order[::-1]
            for r, i in enumerate(order):
                if np.isfinite(col[i]):
                    rank[i, j] = r
                else:
                    rank[i, j] = np.nan
        im = ax.imshow(rank, cmap="viridis", aspect="auto", vmin=0, vmax=max(len(objs) - 1, 1))
        ax.set_xticks(range(len(METRICS)))
        ax.set_xticklabels([m[1].replace("Held-out test ", "") for m in METRICS], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(objs)))
        ax.set_yticklabels([f"{o}-trained head" for o in objs])
        ax.set_xlabel("Held-out test metric (proper scores: lower better; correlations: higher better)", fontsize=8)
        ax.set_ylabel("Training objective (linear head)", fontsize=8)
        midpoint = max(len(objs) - 1, 1) / 2
        for i in range(len(objs)):
            for j in range(len(METRICS)):
                # viridis paints rank 0 (best) dark purple and higher ranks (worse) yellow; NaN cells are white.
                ink = "black" if (not np.isfinite(rank[i, j]) or rank[i, j] >= midpoint) else "white"
                ax.text(j, i, text[i][j], ha="center", va="center", fontsize=5.8, color=ink)
        ax.set_title(
            f"{name}\nheld-out test prompts, item-clustered 95% interval\n"
            f"color is within-column rank (darker is better)",
            fontsize=9,
        )
        ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(objs), 1), minor=True)
        ax.tick_params(which="minor", length=0)
        ax.grid(True, which="major", linewidth=0.3, alpha=0.18)
        ax.grid(True, which="minor", color="0.5", linewidth=0.6, alpha=0.55)
        ax.set_axisbelow(True)
        fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03, label="Rank in column (0 = best)")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _dollar_axis(ax):
    top = ax.secondary_xaxis("top", functions=(lambda k: k, lambda d: d))
    # Tick labels already carry both prices. A second title collides with them.
    ticks = ax.get_xticks()
    ticks = [t for t in ticks if t >= 1]
    top.set_xticks(ticks)
    top.set_xticklabels([f"${CROWD_USD * t:.2f}\n${EXPERT_USD * t:.0f}" for t in ticks], fontsize=7)
    top.xaxis.set_minor_formatter(NullFormatter())
    _style(top)
    return top


def _plot_curve(ax, curves, title):
    colors = {"ce": "#0072B2", "brier": "#E69F00", "sqrt_rps": "#009E73"}
    labels = {
        "ce": "Held-out test cross-entropy (nats)",
        "brier": "Held-out test Brier score",
        "sqrt_rps": "Held-out test sqrt(ranked probability score)",
    }
    for obj, style in (("ce", "-"), ("pearson", "--")):
        series = curves["curves"][obj]
        ks = [row["k"] for row in series]
        for metric in ("ce", "brier", "sqrt_rps"):
            y = np.array([_finite(row["point"][metric]) for row in series])
            lo = np.array([_finite(row["ci"][metric]["lo"]) for row in series])
            hi = np.array([_finite(row["ci"][metric]["hi"]) for row in series])
            name = f"{obj}-trained, {labels[metric]}"
            ax.plot(ks, y, style, color=colors[metric], label=name, linewidth=1.6)
            ax.fill_between(ks, lo, hi, color=colors[metric], alpha=0.12)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(
        "Train ratings per item k (top ticks: USD 0.05 crowd / USD 5 expert)\nHeld-out test uses every test rating",
        fontsize=8.5,
    )
    ax.set_ylabel("Held-out test score (lower is better)")
    ax.set_title(title, fontsize=10, pad=22)
    xs = np.concatenate([line.get_xdata() for line in ax.get_lines()])
    kmin, kmax = float(np.min(xs)), float(np.max(xs))
    if kmax > 4:
        ax.axvspan(4, min(8.0, kmax), color="0.75", alpha=0.25, zorder=0)
    ax.set_xlim(kmin / 1.15, kmax * 1.15)
    _style(ax)
    _dollar_axis(ax)


def _crossover(payload, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    left = payload["crossover"].get("binary_alpha")
    right = payload["crossover"].get("ordinal_summeval_crowd") or payload["crossover"].get("ordinal_summeval")
    if left:
        _plot_curve(axes[0], left, "Binary: ChaosNLI abductive NLI (native 2-way)")
    if right:
        _plot_curve(axes[1], right, "Ordinal Likert: SummEval crowd grades 1–5 (not a 0–5 scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=True,
               bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _high_k(payload, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    a = payload["crossover"].get("ordinal_dices990")
    b = payload["crossover"].get("chaos_3way")
    if a:
        _plot_curve(axes[0], a, "Ordinal 3-level: DICES-990 Q_overall (No < Unsure < Yes)")
    if b:
        _plot_curve(axes[1], b, "3-class: ChaosNLI SNLI+MNLI (imposed order entailment < neutral < contradiction)")
    handles, labels = axes[1].get_legend_handles_labels() if b else axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=True,
               bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _crossover_extra(payload, path):
    mats = payload["matrices"]
    specs = [
        ("ordinal_popquorn_offensiveness", "popquorn_offensiveness", "POPQUORN offensiveness (1–5 Likert, N_train={n})"),
        ("ordinal_popquorn_politeness", "popquorn_politeness", "POPQUORN politeness (1–5 Likert, N_train={n})"),
        ("ordinal_veribench_fsc_rv1", "veribench_fsc_rv1", "VeriBench FSC R_v1 (0–5, N_train={n}, 15 tasks)"),
        ("ordinal_veribench_fsc_pooled", "veribench_fsc_pooled", "VeriBench FSC pooled (0–5, N_train={n}, 30 tasks)"),
    ]
    avail = [
        (payload["crossover"][ck], fmt.format(n=mats[mk]["n_train"]))
        for ck, mk, fmt in specs
        if ck in payload["crossover"] and mk in mats
    ]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(5.8 * len(avail), 5.0), squeeze=False)
    for ax, (curve, title) in zip(axes[0], avail):
        _plot_curve(ax, curve, title)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=True, bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _certjudge_panel(payload, path):
    cj = payload.get("certjudge_real_panels")
    if not cj:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    # Left panel: E63 Qwen3-8B training arms on VeriBench FSC R_v1 held-out test (all5)
    ax = axes[0]
    arms = cj["e63_study_a"]["all5"]["arms"]
    order = ["zero_shot_base", "prompt_opt_arm1", "rl_grpo", "teacher_only_sft", "human_mix_sft", "shuffled_control"]
    present = [k for k in order if k in arms]
    xs = np.arange(len(present))
    ces = [arms[k]["ce"] for k in present]
    sps = [arms[k]["spearman"] for k in present]
    ax.bar(xs - 0.18, ces, width=0.36, color="#0072B2", label="Held-out test CE (nats, lower better)")
    ax.set_xticks(xs)
    ax.set_xticklabels(present, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Held-out test cross-entropy (nats, lower better)", color="#0072B2")
    ax.set_xlabel("CertJudge E63 Qwen3-8B training arm (VeriBench FSC R_v1 all5, 30 test items / 10 tasks)")
    _style(ax)
    ax2 = ax.twinx()
    ax2.plot(xs + 0.18, sps, "o-", color="#D55E00", label="Held-out test Spearman rho (higher better)")
    ax2.set_ylabel("Held-out test Spearman rho (higher better)", color="#D55E00")
    ax2.set_ylim(-0.4, 1.0)
    _style(ax2)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7)
    rl = arms.get("rl_grpo", {})
    ax.set_title(
        f"CertJudge E63: RL-GRPO raises Spearman ({rl.get('spearman', float('nan')):.3f}) "
        f"yet blows up CE ({rl.get('ce', float('nan')):.3f})",
        fontsize=9,
    )

    # Right panel: E39 LLM configs + Exp 67 headline families (CE-fit vs Pearson-fit)
    ax = axes[1]
    e39_test = cj["e39_study_b"]["test_llm"]
    e39_all = cj["e39_study_b"]["all_llm"]
    n_native = cj["exp67_native_interim"]["n_files"]
    exp67_lvl = [r for r in cj["exp67_native_interim"]["calibrated_new"] if "levels" in r]
    labels = [
        f"E39 test_llm\n({e39_test['n_configs']} configs)",
        f"E39 all_llm\n({e39_all['n_configs']} configs)",
    ] + [f"Exp67\n{r['family']}\n({n_native} native)" for r in exp67_lvl]
    ce_fit = [e39_test["ce_cal_ce_mean"], e39_all["ce_cal_ce_mean"]] + [r["levels"]["CE_fit_CE"] for r in exp67_lvl]
    pe_fit = [e39_test["pe_cal_ce_mean"], e39_all["pe_cal_ce_mean"]] + [r["levels"]["Pearson_fit_CE"] for r in exp67_lvl]
    x2 = np.arange(len(labels))
    ax.bar(x2 - 0.18, ce_fit, width=0.36, color="#0072B2", label="CE-calibrated held-out CE (nats, lower better)")
    ax.bar(x2 + 0.18, pe_fit, width=0.36, color="#D55E00", label="Pearson-calibrated held-out CE (nats, lower better)")
    ax.set_xticks(x2)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Held-out test cross-entropy (nats, lower better)")
    ax.set_xlabel("Real LLM judge calibration panel (VeriBench FSC R_v1 & native_main)")
    ax.set_title(f"CertJudge E39 (LOTO/held-out) & Exp 67 ({n_native} native): CE-fit beats Pearson-fit", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _rollout(payload, path):
    rows = payload["rollout"]["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = [r["readout"] for r in rows]
    ces = [_finite(r.get("ce_mean_over_reps", r["ce"])) for r in rows]
    ax.bar(range(len(rows)), ces, color="#0072B2")
    for idx, val in enumerate(ces):
        if np.isfinite(val):
            ax.text(idx, val + 0.15, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    finite_ces = [v for v in ces if np.isfinite(v)]
    if finite_ces:
        ax.set_ylim(0, max(finite_ces) * 1.12)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Held-out test cross-entropy (nats, lower is better)")
    ax.set_xlabel(f"Readout of the CE-trained linear head on {payload['rollout']['dataset']}")
    ax.set_title("Single linear pass vs M multinomial samples (1e-6 floor on MC histograms)")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _iso(payload, path):
    rows = payload["iso_budget"]["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if not rows:
        ax.set_title("Iso-budget not run")
        _style(ax)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return
    for obj, color in (("ce", "#0072B2"), ("pearson", "#D55E00")):
        sub = [r for r in rows if r["objective"] == obj]
        xs = [f"N={r['N']}\nk={r['k']}" for r in sub]
        ys = [_finite(r["point"]["ce"]) for r in sub]
        ax.plot(range(len(sub)), ys, "o-", color=color, label=f"{obj}-trained held-out test CE")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(xs)
    ax.set_ylabel("Held-out test cross-entropy (nats, lower is better)")
    ax.set_xlabel(f"Train allocation at fixed rating budget B≈{payload['iso_budget']['budget_target']}")
    ax.set_title("Iso-budget on ChaosNLI SNLI+MNLI (same test prompts)")
    ax.legend(fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
