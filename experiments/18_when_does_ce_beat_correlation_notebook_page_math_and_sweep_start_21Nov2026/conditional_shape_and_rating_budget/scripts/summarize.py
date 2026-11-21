"""Report paired simulation uncertainty, absolute metrics, and matched annotation budgets."""
import csv
import gzip
import itertools
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
import numpy as np

from run import N_TRAIN, OUTPUT, REPEATS, REPLICATIONS, ROOT

DRAW_COUNT = 2000
BASE = ("https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/"
        "18_when_does_ce_beat_correlation_notebook_page_math_and_sweep_start_21Nov2026/"
        "conditional_shape_and_rating_budget/")
METHODS = ("ce", "pearson", "ce_val_rps", "pearson_val_rps", "pearson_oracle_shape", "oracle", "constant")
CONTRASTS = (("pearson", "ce"), ("pearson_val_rps", "ce_val_rps"), ("pearson_oracle_shape", "ce"))
METRICS = ("rps", "ce", "empirical_tvd", "truth_tvd", "pearson_empirical", "spearman_empirical",
           "kendall_empirical", "pearson_truth", "spearman_truth", "mean_absolute_error",
           "signed_bias", "forecast_mean", "observed_mean", "curve_recovery", "shape_recovery")


def load():
    groups = defaultdict(dict)
    undefined, missing = defaultdict(int), defaultdict(int)
    with gzip.open(OUTPUT / "raw.csv.gz", "rt") as file:
        for row in csv.DictReader(file):
            key = (row["scenario"], int(row["n"]), int(row["k"]), row["method"])
            rep = int(row["rep"])
            fields = [float(row[name]) for name in METRICS[:-2]]
            for selected, truth in (("selected_curve", "true_curve"), ("selected_shape", "true_shape")):
                fields.append(float(row[selected] == row[truth]) if row[selected] else np.nan)
            if rep in groups[key]:
                raise ValueError(f"Duplicate {key}, {rep}")
            groups[key][rep] = fields
            undefined[key] += int(row["train_pearson_undefined"])
            missing[key] += int(not np.isfinite(fields[4]))
    return {key: np.array([rows[i] for i in range(REPLICATIONS)]) for key, rows in groups.items()}, undefined, missing


def bootstrap_weights():
    draws = np.random.default_rng(20261122).integers(REPLICATIONS, size=(DRAW_COUNT, REPLICATIONS))
    return np.array([np.bincount(d, minlength=REPLICATIONS) for d in draws], dtype=float)


def summarize(array, weights):
    finite = np.isfinite(array)
    totals = np.nan_to_num(array, nan=0)
    count = finite.sum(axis=0)
    output = []
    for i in range(array.shape[1]):
        if not count[i]:
            output.append((np.nan, np.nan, np.nan, 0))
            continue
        boot_counts = weights @ finite[:, i]
        boot_sums = weights @ totals[:, i]
        selected = boot_counts > 0
        lo, hi = np.percentile(boot_sums[selected]/boot_counts[selected], [2.5, 97.5])
        output.append((totals[:, i].sum()/count[i], lo, hi, int(count[i])))
    return output


def make_tables(groups, weights):
    stats = {}
    with (OUTPUT / "summary.csv").open("w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["scenario", "n", "k", "method", "subtract_method", "metric",
                         "mean", "lower_95", "upper_95", "replications_defined"])
        for sc, n, k in itertools.product(("neutral", "varying"), N_TRAIN, REPEATS):
            for method, subtract in [(name, "") for name in METHODS] + list(CONTRASTS):
                data = groups[sc, n, k, method]
                if subtract:
                    data = data - groups[sc, n, k, subtract]
                for metric, estimate in zip(METRICS, summarize(data, weights), strict=True):
                    stats[sc, n, k, method, subtract, metric] = estimate
                    writer.writerow([sc, n, k, method, subtract, metric, *estimate])
    alloc = {}
    with (OUTPUT / "allocation.csv").open("w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["scenario", "method", "metric_128x1_minus_8x16", "mean",
                         "lower_95", "upper_95", "replications_defined"])
        for sc, method in itertools.product(("neutral", "varying"), METHODS):
            gap = groups[sc, 128, 1, method][:, :1] - groups[sc, 8, 16, method][:, :1]
            alloc[sc, method] = summarize(gap, weights)[0]
            writer.writerow([sc, method, "rps", *alloc[sc, method]])
    return stats, alloc


def estimate(data):
    value, lo, hi, _ = data
    return f"{value:+.5f} [{lo:+.5f}, {hi:+.5f}]"


def plot(stats):
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9.2, 4.8))
    for ax, sc, title in zip(axes, ("neutral", "varying"), ("Neutral shape", "Feature-dependent spread")):
        for k, color, marker in zip(REPEATS, ("#196387", "#9A4A18", "#4D7131"), ("o", "s", "^")):
            values = np.array([stats[sc, n, k, "pearson", "ce", "rps"][:3] for n in N_TRAIN])
            ax.plot(N_TRAIN, values[:, 0], marker=marker, color=color, label=f"{k} per artifact")
            ax.fill_between(N_TRAIN, values[:, 1], values[:, 2], color=color, alpha=.10)
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_TRAIN, labels=[str(n) for n in N_TRAIN])
        ax.xaxis.set_minor_locator(FixedLocator([16, 64]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Distinct training artifacts")
        ax.grid(True, which="major", alpha=.28)
        ax.grid(True, which="minor", alpha=.12)
        ax.axhline(0, color="#484848", lw=.8)
    axes[0].set_ylabel("Held-out test ranked probability score (RPS)\nPearson-trained minus cross-entropy-trained")
    axes[1].legend(title="Training ratings", fontsize=8, loc="lower right")
    fig.suptitle("When a rating distribution can change at a fixed mean", fontsize=14)
    fig.text(.5, .055, "Positive favors cross-entropy. RPS ranges from 0 to 1 (lower is better). Shading: pointwise 95% paired bootstrap intervals.\n"
             "Both judges exhaustively search the same 12 candidates on training ratings, with a neutral-shape tie rule; 256 independent simulations.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .125, 1, .95))
    fig.savefig(OUTPUT / "held_out_rps.png", dpi=180)
    plt.close(fig)


def write_report(stats, alloc, undefined, missing):
    title = "# When does cross-entropy win as rating budgets change? | 11-21-2026"
    report = [title, f"**Doc link:** <{BASE}results.md>", "Date: 11-21-2026",
              "**TLDR:** With eight training ratings, changing only the true shape flipped which objective had lower held-out ranked probability score (RPS). "
              "At the largest budget both methods recovered a neutral-shape truth, but only the cross-entropy (CE) fit recovered a feature-dependent shape under the default Pearson tie rule. "
              "Independent distributional validation could recover it for the Pearson fit as well.",
              "## Primary comparison", "A simulated judge predicts probabilities for integer ratings 0–5. Normalized individual-rating RPS is the mean squared error of five cumulative rating thresholds, ranges from 0 to 1 and is lower-is-better. "
              "The oracle knows the generating probabilities. Constant predicts the smoothed training histogram for every artifact. "
              "Within each replication, CE and Pearson exhaustively search the same 12 forecasts on the same individual training ratings, with the neutral shape first in a tie. "
              "Both achieve a global optimum over that finite set up to the declared tie tolerance. "
              "Every cell averages 256 independent simulations, each scoring the same held-out 384 synthetic artifacts × three fresh ratings across methods. "
              "The grid reuses held-out artifacts across budgets within a simulation. There are no human ratings or previously exposed ratings in this sweep.",
              "The difference subtracts test RPS of the CE-trained judge from test RPS of the Pearson-trained judge. "
              "Positive favors CE. Brackets show pointwise 95% paired percentile bootstrap intervals over complete simulations, 2,000 resamples; p-val=n/a. "
              "The neutral and varying conditions have exactly the same population mean curve in each simulation. Only the latter uses the independent feature z to change rating dispersion.",
              "| True shape | Train artifacts | Ratings per artifact | Oracle | Constant | CE-trained | Pearson-trained | Test RPS difference [95% interval] |\n"
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    body = []
    for sc, n, k in itertools.product(("neutral", "varying"), N_TRAIN, REPEATS):
        vals = [stats[sc, n, k, method, "", "rps"][0] for method in ("oracle", "constant", "ce", "pearson")]
        body.append(f"| {sc} | {n} | {k} | " + " | ".join(f"{v:.5f}" for v in vals)
                    + f" | {estimate(stats[sc,n,k,'pearson','ce','rps'])} |")
    report[-1] += "\n" + "\n".join(body)
    report.extend(["![Paired difference in held-out ranked probability score across all training budgets](results/held_out_rps.png)",
                   "## What the tie rule contributes",
                   "Every trained mean curve has three candidate distributions with exactly the same mean at every input. "
                   "A Pearson fit selecting the correct curve receives exactly the same correlation for all three candidates, even with unlimited ratings. "
                   "Here a preregistered neutral-first rule resolves the ambiguity. A separate fit uses the true shape to break each Pearson tie (an oracle sensitivity). "
                   "A practical alternative chooses among equally optimal training fits by normalized RPS on independent validation artifacts, n/4 artifacts × k ratings per artifact, identically available to each objective. "
                   "Validation data thus augment the training budget only for these explicitly named procedures.",
                   "| True shape | Train budget | Extra validation ratings for validation tie rule | Default Pearson minus CE test RPS | Oracle-shape Pearson minus ordinary CE test RPS | Validation-tied Pearson minus validation-tied CE test RPS |\n"
                   "|---|---:|---:|---:|---:|---:|"])
    rows = []
    for sc, n, k in itertools.product(("neutral", "varying"), N_TRAIN, REPEATS):
        vals = [stats[sc,n,k,a,b,"rps"] for a,b in (CONTRASTS[0], CONTRASTS[2], CONTRASTS[1])]
        rows.append(f"| {sc} | {n} × {k} | {(n//4)*k} | " + " | ".join(estimate(v) for v in vals) + " |")
    report[-1] += "\n" + "\n".join(rows)
    report.extend(["## At a fixed total of 128 training ratings",
                   "For each method, the entry is its held-out test RPS at 128 artifacts × one rating minus its test RPS at eight artifacts × 16 ratings. Negative favors spreading the same 128 ratings across more artifacts. "
                   "The 95% intervals use the paired replicate bootstrap above.",
                   "| True shape | CE-trained | Pearson-trained |\n|---|---:|---:|\n" +
                   "\n".join(f"| {sc} | {estimate(alloc[sc,'ce'])} | {estimate(alloc[sc,'pearson'])} |" for sc in ("neutral", "varying")),
                   "## Secondary held-out metrics at the largest budget",
                   "All entries use 128 training artifacts × 16 ratings. CE is the average negative natural log probability per individual test rating in nats (lower is better). "
                   "Total variation distance (TVD) is half the summed absolute probability difference ([0,1], lower is better). Empirical TVD uses the three-rating test histogram; population TVD uses the generator probabilities. "
                   "Pearson and midrank Spearman measure linear and rank association between forecast and observed artifact means (−1 to 1, higher is better). "
                   "Across every budget, [summary.csv](results/summary.csv) provides absolute estimates, paired contrasts, uncertainty and valid counts for these metrics, Kendall tau-b, numeric agreement and the true-mean diagnostics.",
                   "| True shape | Method | CE (nats) | Empirical TVD | Population TVD | Pearson | Spearman |\n|---|---|---:|---:|---:|---:|---:|"])
    rows = []
    for sc, method in itertools.product(("neutral", "varying"), ("ce", "pearson", "pearson_val_rps", "oracle", "constant")):
        vals = [stats[sc,128,16,method,"",m][0] for m in ("ce", "empirical_tvd", "truth_tvd", "pearson_empirical", "spearman_empirical")]
        label = {"ce":"CE-trained","pearson":"Pearson-trained","pearson_val_rps":"Pearson, validation RPS tie","oracle":"Oracle","constant":"Constant"}[method]
        rows.append(f"| {sc} | {label} | " + " | ".join(f"{v:.5f}" if np.isfinite(v) else "undefined" for v in vals) + " |")
    report[-1] += "\n" + "\n".join(rows)
    undefined_count = sum(value for key,value in undefined.items() if key[3] == "pearson")
    constant_missing = sum(value for key,value in missing.items() if key[3] == "constant")
    other_missing = sum(value for key,value in missing.items() if key[3] != "constant")
    report.extend(["## Checks, applicability and reproduction",
                   f"All {len(METHODS)*REPLICATIONS*2*len(N_TRAIN)*len(REPEATS):,} records were retained. Undefined training-target fallback occurred in {undefined_count:,} of {REPLICATIONS*2*len(N_TRAIN)*len(REPEATS):,} Pearson training instances. "
                   f"The constant comparator has {constant_missing:,} undefined held-out Pearson values. Across the six nonconstant procedures, {other_missing:,} held-out Pearson values were undefined.",
                   "When the predicted curve is correct and the true shape varies, the population RPS penalty for fixing the shape to neutral is exactly 0.00512. "
                   "This follows by averaging squared cumulative-probability differences across the five thresholds. It is a consequence of the chosen shape amplitude. "
                   "Population TVD is zero for the oracle; its empirical TVD is positive because three ratings give a noisy target. This is why empirical TVD alone should not decide which distribution forecasts best.",
                   "The generator deliberately contains a shape feature orthogonal to the mean. All candidate distributions have known form, no separate rater identities or severity, and include the true conditional distribution. "
                   "A judge can predict a nonlinear mean under Pearson training. Observed finite-budget crossovers and ranking performance depend on the family, strength of the shape effect, tie convention and use of validation labels. "
                   "The human evidence needs its own comparison, including the strong constant baseline. See [how to read these results](HOW_TO_READ_RESULTS.md) for the metric dictionary, budgets, and reproduction commands."])
    (ROOT / "results.md").write_text("\n\n".join(report) + "\n")


if __name__ == "__main__":
    groups, undefined, missing = load()
    stats, alloc = make_tables(groups, bootstrap_weights())
    plot(stats)
    write_report(stats, alloc, undefined, missing)
    print("Wrote results.md, summary.csv, allocation.csv and the held-out test RPS plot.")
