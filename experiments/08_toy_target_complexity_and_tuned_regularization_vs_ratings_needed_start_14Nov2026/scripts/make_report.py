"""Experiment 08 report: tests H1 (Brando's low-data conjecture) and H2 (alternative) from the README.

Reads results/raw.jsonl (no new fits); uses (scenario, n) cells with all 100 seeds in every scenario.
Writes fig_complexity_vs_n.png, fig_complexity_correlation_vs_n.png and results/tables.md; prints the tables.
Paired differences are per seed with 95% bootstrap intervals over seeds (10,000 draws).
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cells = {}
for line in open(os.path.join(HERE, "results", "raw.jsonl")):
    r = json.loads(line)
    cells.setdefault((r["scenario"], r["n"]), {})[r["seed"]] = r["fits"]
SC = "ABCD"
NS = sorted({n for (_, n) in cells if all(len(cells.get((s, n), {})) == 100 for s in SC)})
SCEN = {"A": "A: mean nearly constant, disagreement varies", "B": "B: nonlinear mean, disagreement varies",
        "C": "C: nonlinear mean, constant disagreement", "D": "D: linear mean, constant disagreement"}
JUDGES = ["constant", "ce_fixed", "ce_valrps", "rps_valrps", "pearson_fixed", "pearson_valcorr", "pearson_valrps"]
NAMES = {"constant": "constant (training histogram)", "ce_fixed": "CE-trained, ridge 1 (④)", "ce_valrps": "CE-trained, ridge by validation RPS (③)",
         "rps_valrps": "RPS-trained, ridge by validation RPS", "pearson_fixed": "Pearson-trained, ridge 1 (②)",
         "pearson_valcorr": "Pearson-trained, ridge by validation Pearson (①)", "pearson_valrps": "Pearson-trained, ridge by validation RPS"}
STYLE = {"constant": ("gray", "-"), "ce_fixed": ("tab:blue", "-"), "ce_valrps": ("tab:blue", "--"), "rps_valrps": ("tab:green", "--"),
         "pearson_fixed": ("tab:red", "-"), "pearson_valcorr": ("tab:red", "--"), "pearson_valrps": ("tab:red", ":")}
METRICS = {"rps_excess": "Held-out test RPS excess\nover the true distribution", "ce_excess": "Held-out test CE excess = KL(true ‖ judge)\nover the true distribution (nats)"}
rng = np.random.default_rng(0)


def vals(sc, n, j, key):
    c = cells[(sc, n)]
    return np.array([c[s][j][key] for s in sorted(c)], dtype=float)


def mean_ci(v, draws=10000):
    boot = v[rng.integers(0, len(v), (draws, len(v)))].mean(1)
    return v.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def diff_ci(sc, n, a, b, key):
    """Paired difference a minus b; for correlations, undefined (nan) counts as 0."""
    x, y = vals(sc, n, a, key), vals(sc, n, b, key)
    if key in ("pearson", "spearman"):
        x, y = np.nan_to_num(x, nan=0.0), np.nan_to_num(y, nan=0.0)
    return mean_ci(x - y)


def crossover(sc, key, loss_judge, corr_judge):
    """Smallest n from which loss_judge is significantly better (lower) at every larger n."""
    better = [diff_ci(sc, n, corr_judge, loss_judge, key)[1] > 0 for n in NS]
    return next((NS[k] for k in range(len(NS)) if all(better[k:])), None)


fmt = lambda v: f"{v} tasks ({3 * v} ratings)" if v else "not at every larger n"
L = [f"Cells with all 100 seeds in every scenario: n = {NS}.\n"]
for sc in SC:
    L.append(f"\n## Scenario {SCEN[sc]}\n\nHeld-out test RPS excess (mean over 100 seeds):\n")
    L.append("| n (ratings) | " + " | ".join(NAMES[j] for j in JUDGES) + " |")
    L.append("|---" * (len(JUDGES) + 1) + "|")
    for n in NS:
        L.append(f"| {n} ({3 * n}) | " + " | ".join(f"{vals(sc, n, j, 'rps_excess').mean():.4f}" for j in JUDGES) + " |")
    L.append("\nCrossovers on held-out test RPS (smallest n from which the first judge is significantly better at every larger n):")
    for a, b in (("ce_fixed", "pearson_fixed"), ("ce_valrps", "pearson_valcorr"), ("ce_valrps", "pearson_fixed"), ("ce_valrps", "pearson_valrps"), ("ce_valrps", "constant"), ("pearson_fixed", "constant")):
        L.append(f"- {NAMES[a]} vs {NAMES[b]}: {fmt(crossover(sc, 'rps_excess', a, b))}")
    worse = {b: [n for n in NS if diff_ci(sc, n, "ce_valrps", b, "rps_excess")[1] > 0] for b in ("pearson_fixed", "pearson_valcorr", "pearson_valrps")}
    L.append("\nH2 check, sizes where the validation-tuned CE judge (③) is significantly WORSE on test RPS than: " + "; ".join(f"{NAMES[b]}: {v if v else 'none'}" for b, v in worse.items()))
    h1 = [n for n in NS if diff_ci(sc, n, "ce_fixed", "pearson_fixed", "rps_excess")[1] > 0]
    L.append(f"H1 check, sizes where the Pearson-trained judge (②) is significantly BETTER on test RPS than the CE-trained judge (④): {h1 if h1 else 'none'}")
    corrw = {b: [n for n in NS if diff_ci(sc, n, "ce_fixed", b, "pearson")[2] < 0] for b in ("pearson_fixed", "pearson_valcorr")}
    L.append("H2 check, sizes where the CE-trained judge (④) has significantly LOWER test Pearson correlation with the true mean than: " + "; ".join(f"{NAMES[b]}: {v if v else 'none'}" for b, v in corrw.items()))
    L.append("\nTest Pearson correlation with the true mean (mean over 100 seeds; undefined counted as 0):\n")
    L.append("| n | " + " | ".join(NAMES[j] for j in JUDGES[1:]) + " |")
    L.append("|---" * len(JUDGES) + "|")
    for n in NS:
        L.append(f"| {n} | " + " | ".join(f"{np.nan_to_num(vals(sc, n, j, 'pearson'), nan=0.0).mean():.3f}" for j in JUDGES[1:]) + " |")
    L.append("\nDistance from the constant forecast (mean test total variation distance), small n:\n")
    L.append("| n | " + " | ".join(NAMES[j] for j in JUDGES[1:]) + " |")
    L.append("|---" * len(JUDGES) + "|")
    for n in NS[:6]:
        L.append(f"| {n} | " + " | ".join(f"{vals(sc, n, j, 'tvd_from_constant').mean():.3f}" for j in JUDGES[1:]) + " |")
    L.append("\nChosen ridge c (most common value; share of seeds), and Pearson (①) fallbacks to c = 1:\n")
    L.append("| n | CE ③ | RPS (val RPS) | Pearson ① | Pearson (val RPS) | ① fallbacks |")
    L.append("|---|---|---|---|---|---|")
    for n in NS:
        def mode(j):
            v = vals(sc, n, j, "c"); u, cnt = np.unique(v, return_counts=True)
            return f"{u[cnt.argmax()]:g} ({cnt.max()}%)"
        fb = int(sum(cells[(sc, n)][s]["pearson_valcorr"].get("fallback", False) for s in cells[(sc, n)]))
        L.append(f"| {n} | {mode('ce_valrps')} | {mode('rps_valrps')} | {mode('pearson_valcorr')} | {mode('pearson_valrps')} | {fb} |")

for fname, keys in (("fig_complexity_vs_n.png", list(METRICS)), ("fig_complexity_correlation_vs_n.png", ["pearson"])):
    fig, axs = plt.subplots(4, len(keys), figsize=(5.6 * len(keys) + 0.4, 15), sharex=True, squeeze=False)
    for i, sc in enumerate(SC):
        for j, key in enumerate(keys):
            ax = axs[i, j]
            for jd in JUDGES:
                if key == "pearson" and jd == "constant":
                    continue
                col, ls = STYLE[jd]
                v = [mean_ci(np.nan_to_num(vals(sc, n, jd, key), nan=0.0)) for n in NS]
                v = np.array(v)
                ax.plot(NS, v[:, 0], "o" + ls, color=col, ms=3, label=NAMES[jd])
                ax.fill_between(NS, v[:, 1], v[:, 2], color=col, alpha=0.1)
            ax.set_xscale("log")
            if key != "pearson":
                ax.set_yscale("log")
                for q, (a, b, ls2) in enumerate((("ce_valrps", "pearson_valcorr", "--"), ("ce_fixed", "pearson_fixed", ":"))):
                    s = crossover(sc, key, a, b)
                    if s is not None:
                        ax.axvline(s, color="black", ls=ls2, lw=1.2)
                        lab = "③ beats ①" if a == "ce_valrps" else "④ beats ②"
                        ax.annotate(f"{lab} from n = {s}\n({3 * s} ratings)", xy=(s, 0.03 + 0.13 * q), xycoords=("data", "axes fraction"),
                                    xytext=(4, 0), textcoords="offset points", fontsize=7, va="bottom")
                ax.set_ylabel(METRICS[key] + " (lower is better)", fontsize=8.5)
            else:
                # Data-needed lines on test correlation: dashed = the CE-trained judge significantly higher at every larger n;
                # dotted = the Pearson-trained judge significantly higher at every larger n (paired 95% bootstrap over seeds).
                q = 0
                for a, b, la, lb in (("ce_fixed", "pearson_fixed", "④", "②"), ("ce_valrps", "pearson_valcorr", "③", "①")):
                    d = [diff_ci(sc, n, a, b, "pearson") for n in NS]
                    s1 = next((NS[k] for k in range(len(NS)) if all(x[1] > 0 for x in d[k:])), None)
                    s2 = next((NS[k] for k in range(len(NS)) if all(x[2] < 0 for x in d[k:])), None)
                    for s, txt, ls2, ha, xo, yfrac in ((s1, f"{la} higher than {lb}", "--", "left", 4, 0.62), (s2, f"{lb} higher than {la}", ":", "right", -4, 0.10)):
                        if s is not None:
                            ax.axvline(s, color="black", ls=ls2, lw=1.2)
                            ax.annotate(f"{txt} at every n\nfrom {s} ({3 * s} ratings)", xy=(s, yfrac + 0.13 * q), xycoords=("data", "axes fraction"),
                                        xytext=(xo, 0), textcoords="offset points", fontsize=7, va="bottom", ha=ha)
                    q += 1
                ax.set_ylabel("Held-out test Pearson correlation\n(judge's expected rating vs true mean) (higher is better)", fontsize=8.5)
            ax.grid(True, which="major", alpha=0.4); ax.grid(True, which="minor", ls=":", alpha=0.2)
            ax.set_title(f"Scenario {SCEN[sc]}", fontsize=9.5)
            ax.set_xlabel("training tasks n (3 ratings each, so 3n training ratings)", fontsize=8.5)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2 if len(keys) == 1 else 4, fontsize=8, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.95 if len(keys) == 1 else 0.96))
    fig.savefig(os.path.join(HERE, fname), dpi=120)
open(os.path.join(HERE, "results", "tables.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
