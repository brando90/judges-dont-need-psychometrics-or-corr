"""Publication figures. Every numeric axis has gridlines. Axis text names the metric and the split."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import percentile_interval  # noqa: F401  (kept so a reader sees the CI helper lives with the metrics)

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
    _matrix(payload, fig_dir / "wrong_objective_matrix.png")
    _crossover(payload, fig_dir / "finite_sample_crossover_binary_vs_ordinal.png")
    _high_k(payload, fig_dir / "high_k_crossover_dices_chaos.png")
    _rollout(payload, fig_dir / "single_pass_vs_rollout.png")
    _iso(payload, fig_dir / "iso_budget_chaosnli.png")


def _matrix(payload, path):
    names = [n for n in ("summeval_expert", "summeval_expert_binary", "chaosnli_alpha", "dices990") if n in payload["matrices"]]
    if not names:
        names = list(payload["matrices"])[:2]
    fig, axes = plt.subplots(1, len(names), figsize=(6.2 * len(names), 4.8), squeeze=False)
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
        im = ax.imshow(rank, cmap="viridis_r", aspect="auto", vmin=0, vmax=max(len(objs) - 1, 1))
        ax.set_xticks(range(len(METRICS)))
        ax.set_xticklabels([m[1].replace("Held-out test ", "") for m in METRICS], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(objs)))
        ax.set_yticklabels([f"{o}-trained head" for o in objs])
        midpoint = max(len(objs) - 1, 1) / 2
        for i in range(len(objs)):
            for j in range(len(METRICS)):
                # viridis_r paints a good rank dark and a bad rank yellow.
                ink = "black" if np.isfinite(rank[i, j]) and rank[i, j] >= midpoint else "white"
                ax.text(j, i, text[i][j], ha="center", va="center", fontsize=6.5, color=ink)
        ax.set_title(
            f"{name}\nheld-out test prompts, item-clustered 95% interval\n"
            f"color is within-column rank (darker is better)",
            fontsize=9,
        )
        _style(ax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Rank in column (0 = best)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _dollar_axis(ax):
    top = ax.secondary_xaxis("top", functions=(lambda k: k, lambda d: d))
    # Tick labels already carry both prices. A second title collides with them.
    ticks = ax.get_xticks()
    ticks = [t for t in ticks if t >= 1]
    top.set_xticks(ticks)
    top.set_xticklabels([f"${CROWD_USD * t:.2f}\n${EXPERT_USD * t:.0f}" for t in ticks], fontsize=7)
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
    ax.set_xlabel("Ratings per item k. Each top tick is crowd $0.05 over expert $5. Test uses every test rating.")
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


def _rollout(payload, path):
    rows = payload["rollout"]["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = [r["readout"] for r in rows]
    ces = [_finite(r["ce"]) for r in rows]
    ax.bar(range(len(rows)), ces, color="#0072B2")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Held-out test cross-entropy (nats, lower is better)")
    ax.set_xlabel(f"Readout of the CE-trained linear head on {payload['rollout']['dataset']}")
    ax.set_title("Single linear pass versus M multinomial samples of that same distribution")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _iso(payload, path):
    rows = payload["iso_budget"]["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if not rows:
        ax.set_title("Iso-budget not run")
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
