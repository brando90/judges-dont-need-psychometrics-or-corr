"""Experiment 11 report generator: computes summary.json, results/tables.md, and fig_mse_and_tuned_head_vs_n.png."""
import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "results", "raw.jsonl")
SC = "ABCD"
SCEN = {
    "A": "A: mean nearly constant, disagreement varies",
    "B": "B: nonlinear mean, disagreement varies",
    "C": "C: nonlinear mean, constant concentration κ",
    "D": "D: linear mean, constant concentration κ",
}
JUDGES = [
    "constant",
    "ce_fixed",
    "ce_valrps",
    "pearson_fixed",
    "pearson_head_naive",
    "pearson_head_in_tuned",
    "pearson_head_tuned",
    "mse_fixed",
    "mse_valrps",
    "mse_head_tuned",
]
NAMES = {
    "constant": "Constant (train histogram)",
    "ce_fixed": "CE-trained, ridge 1 (④)",
    "ce_valrps": "CE-trained, val RPS ridge (③)",
    "pearson_fixed": "Pearson-trained, ridge 1 (②)",
    "pearson_head_naive": "Pearson + naive head (1/n ridge)",
    "pearson_head_in_tuned": "Pearson + tuned in-sample head",
    "pearson_head_tuned": "Pearson + cross-fitted tuned head",
    "mse_fixed": "MSE-trained, ridge 1",
    "mse_valrps": "MSE-trained, val RPS ridge",
    "mse_head_tuned": "MSE + cross-fitted tuned head",
}
PLOT_JUDGES = [
    ("constant", "gray", "-", None),
    ("ce_valrps", "#1d4ed8", "-", "o"),
    ("pearson_fixed", "#dc2626", "-", "s"),
    ("pearson_head_naive", "#f97316", ":", "^"),
    ("pearson_head_tuned", "#d97706", "--", "D"),
    ("mse_valrps", "#15803d", "-.", "v"),
    ("mse_head_tuned", "#0f766e", "--", "P"),
]
METRICS = {
    "rps_excess": "Held-out test RPS excess over truth\n(0 optimal; lower is better)",
    "ce_excess": "Held-out test CE excess = KL(true || judge)\n(nats, 0 optimal; lower is better)",
}


def load_cells():
    cells = {}
    with open(RAW) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cells.setdefault((r["scenario"], r["n"]), {})[r["seed"]] = r["fits"]
    ns = sorted({n for (_, n) in cells if all(len(cells.get((s, n), {})) == 100 for s in SC)})
    return cells, ns


def vals(cells, sc, n, j, key):
    c = cells[(sc, n)]
    if j == "true_rps":
        return np.array([c[s]["true_rps"] for s in sorted(c)], dtype=float)
    return np.array([c[s][j][key] for s in sorted(c)], dtype=float)


def mean_ci(v, rng, draws=10000):
    idx = rng.integers(0, len(v), (draws, len(v)))
    boot = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def diff_ci(cells, sc, n, a, b, key, rng):
    x, y = vals(cells, sc, n, a, key), vals(cells, sc, n, b, key)
    if key in ("pearson", "spearman"):
        x, y = np.nan_to_num(x, nan=0.0), np.nan_to_num(y, nan=0.0)
    return mean_ci(x - y, rng)


def crossover(cells, ns, sc, key, better_j, baseline_j, rng):
    ok = [diff_ci(cells, sc, n, baseline_j, better_j, key, rng)[1] > 0 for n in ns]
    return next((ns[k] for k in range(len(ns)) if all(ok[k:])), None)


def main():
    cells, ns = load_cells()
    rng = np.random.default_rng(0)
    summary = {"n_values": ns, "seeds_per_cell": 100, "test_tasks_per_seed": 500, "scenarios": {}}
    lines = [
        "# Generated results: experiment 11\n",
        f"Cells with all 100 seeds in every scenario: n = {ns} (3 ratings/task, 500 held-out test tasks/seed).\n",
        "True distribution test RPS baselines (empirical truth on test tasks): checked per scenario below.\n",
        "All pairwise comparisons use 10,000 paired seed bootstrap draws (95% percentile CIs). "
        "CE uses exact unclipped log-sum-exp loss/gradient; Stage-2 heads tune slope ridge c_b in {0.01, 0.1, 1, 10, 100, 1e4, 1e8} by 4-fold CV RPS.\n",
    ]

    for sc in SC:
        true_rps_mean = float(np.mean([vals(cells, sc, n, "true_rps", None).mean() for n in ns]))
        sc_sum = {"true_rps": true_rps_mean, "cells": {}, "crossovers_ce_valrps_beats": {}, "small_n_max_excess_over_const": {}}
        lines.append(f"\n## Scenario {SCEN[sc]} (true test RPS ≈ {true_rps_mean:.4f})\n")
        lines.append("### Held-out test RPS excess over truth (mean over 100 seeds; 0 optimal, lower is better)\n")
        lines.append("| n (ratings) | true RPS | " + " | ".join(NAMES[j] for j in JUDGES) + " |")
        lines.append("|---" * (len(JUDGES) + 2) + "|")

        for n in ns:
            t_rps = vals(cells, sc, n, "true_rps", None).mean()
            row = [f"{vals(cells, sc, n, j, 'rps_excess').mean():.4f}" for j in JUDGES]
            lines.append(f"| {n} ({3 * n}) | {t_rps:.4f} | " + " | ".join(row) + " |")
            cell_d = {"true_rps": float(t_rps)}
            for j in JUDGES:
                m_rps, lo_rps, hi_rps = mean_ci(vals(cells, sc, n, j, "rps_excess"), rng)
                m_raw, _, _ = mean_ci(vals(cells, sc, n, j, "rps"), rng)
                m_ce, _, _ = mean_ci(vals(cells, sc, n, j, "ce_excess"), rng)
                p_vals = vals(cells, sc, n, j, "pearson")
                undef_cnt = int(np.isnan(p_vals).sum()) if j != "constant" else 100
                m_p = float(np.nan_to_num(p_vals, nan=0.0).mean()) if j != "constant" else 0.0
                d_c_m, d_c_lo, d_c_hi = diff_ci(cells, sc, n, j, "constant", "rps_excess", rng) if j != "constant" else (0.0, 0.0, 0.0)
                d_ce_m, d_ce_lo, d_ce_hi = diff_ci(cells, sc, n, j, "ce_valrps", "rps_excess", rng) if j != "ce_valrps" else (0.0, 0.0, 0.0)
                cell_d[j] = {
                    "rps_raw": m_raw,
                    "rps_excess": m_rps,
                    "rps_excess_ci": [lo_rps, hi_rps],
                    "ce_excess": m_ce,
                    "pearson_mean_zero_undef": m_p,
                    "pearson_undef_count": undef_cnt,
                    "minus_constant_rps": [d_c_m, d_c_lo, d_c_hi],
                    "minus_ce_valrps_rps": [d_ce_m, d_ce_lo, d_ce_hi],
                }
            sc_sum["cells"][str(n)] = cell_d

        lines.append("\n### Held-out test RPS relative to Constant and Val-CE (n = 8, 32, 256, 2048)\n")
        lines.append("| n | Judge | Raw RPS | RPS excess | Judge − Constant [95% CI] | Judge − ③ Val-CE [95% CI] | CE excess (nats) |")
        lines.append("|---|---|---|---|---|---|---|")
        for n in (8, 32, 256, 2048):
            c = sc_sum["cells"][str(n)]
            for j in JUDGES:
                cj = c[j]
                dc = cj["minus_constant_rps"]
                dce = cj["minus_ce_valrps_rps"]
                lines.append(
                    f"| {n} | {NAMES[j]} | {cj['rps_raw']:.4f} | {cj['rps_excess']:.4f} | "
                    f"{dc[0]:+.4f} [{dc[1]:+.4f}, {dc[2]:+.4f}] | "
                    f"{dce[0]:+.4f} [{dce[1]:+.4f}, {dce[2]:+.4f}] | {cj['ce_excess']:.4f} |"
                )

        lines.append("\n### Crossovers & small-n stability (n <= 32)\n")
        for j in ("pearson_fixed", "pearson_head_naive", "pearson_head_in_tuned", "pearson_head_tuned", "mse_fixed", "mse_valrps", "mse_head_tuned", "constant"):
            cr = crossover(cells, ns, sc, "rps_excess", "ce_valrps", j, rng)
            sc_sum["crossovers_ce_valrps_beats"][j] = cr
            msg = f"{cr} tasks ({3 * cr} ratings)" if cr else "not at every larger n"
            lines.append(f"- ③ Val-CE significantly beats {NAMES[j]} at every larger n from: {msg}")
        for j in ("pearson_head_naive", "pearson_head_in_tuned", "pearson_head_tuned", "mse_fixed", "mse_valrps", "mse_head_tuned"):
            max_ex = max(sc_sum["cells"][str(n)][j]["minus_constant_rps"][0] for n in ns if n <= 32)
            sc_sum["small_n_max_excess_over_const"][j] = float(max_ex)
            lines.append(f"- Max small-n (n <= 32) test RPS minus constant for {NAMES[j]}: {max_ex:+.4f}")

        summary["scenarios"][sc] = sc_sum

    with open(os.path.join(HERE, "results", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(HERE, "results", "tables.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Render figure across Scenarios A-D (RPS excess left, CE excess right)
    fig, axs = plt.subplots(4, 2, figsize=(11.5, 14.5), sharex=True)
    for i, sc in enumerate(SC):
        for col_i, key in enumerate(("rps_excess", "ce_excess")):
            ax = axs[i, col_i]
            for j, color, ls, mk in PLOT_JUDGES:
                ys = np.array([mean_ci(vals(cells, sc, n, j, key), rng) for n in ns])
                ax.plot(ns, ys[:, 0], linestyle=ls, marker=mk, markersize=3.5, lw=1.6, color=color, label=NAMES[j])
                ax.fill_between(ns, np.maximum(ys[:, 1], 1e-6), np.maximum(ys[:, 2], 1e-6), color=color, alpha=0.09)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="major", alpha=0.4)
            ax.grid(True, which="minor", linestyle=":", alpha=0.22)
            ax.set_title(f"Scenario {SCEN[sc]}", fontsize=9.5)
            ax.set_ylabel(METRICS[key], fontsize=8.5)
            if i == 3:
                ax.set_xlabel("Training tasks n (3 ratings each, so 3n training ratings)", fontsize=8.5)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.992))
    fig.text(
        0.5,
        0.01,
        "How chosen: 78-wt cosine/sine softmax + 12-param CE head (Exp 11, 100 seeds, 500 test tasks/seed). "
        "Fixed L-BFGS-B <=400 iters; Stage-2 heads tune slope ridge c_b in {0.01..1e8} via 4-fold CV RPS.\n"
        "Exact log-sum-exp CE is convex; Pearson/MSE have no global optimality certificate. "
        "Bands: 95% percentile bootstrap over 100 seeds (10,000 draws).",
        ha="center",
        va="bottom",
        fontsize=7.2,
    )
    fig.tight_layout(rect=(0, 0.048, 1, 0.935))
    fig.savefig(os.path.join(HERE, "fig_mse_and_tuned_head_vs_n.png"), dpi=140)
    print("Wrote results/summary.json, results/tables.md, and fig_mse_and_tuned_head_vs_n.png")


if __name__ == "__main__":
    main()
