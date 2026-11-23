"""Experiment 28 analysis (11-23-2026). Joins the frozen forecasts to the human counts and writes results/*.csv.
Usage: python3 run_analysis.py FORECASTS.csv LABELS.json OUT_DIR
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from lib28 import *

COMPARATORS = ("pearson", "uniform", "hist", "pcal", "rps")


def main(fc_path, lab_path, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    labels = json.loads(Path(lab_path).read_text())
    fc = {}
    for r in csv.DictReader(open(fc_path)):
        fc.setdefault((r["id"].split("/")[0], r["judge"]), {})[r["id"]] = (r["status"], [float(r[f"p{j}"]) for j in range(3) if r[f"p{j}"] != ""])
    judges = sorted({j for _, j in fc})
    abs_rows, con_rows, cor_rows, status_rows, raw_rows = [], [], [], [], []
    rng = np.random.default_rng(np.random.SeedSequence([20261123, 999]))
    rng_raw = np.random.default_rng(np.random.SeedSequence([20261123, 998]))   # separate stream: adding the raw comparator leaves every other interval unchanged
    for ds, k in DATASETS.items():
        ids = [i for i in labels if i.startswith(ds + "/")]
        counts = np.array([labels[i]["counts"] for i in ids], float)
        groups = [labels[i]["group"] for i in ids]
        K = counts.shape[1]
        scores = np.arange(K, dtype=float)
        for judge in judges:
            f = fc[(ds, judge)]
            p = np.array([f[i][1] for i in ids])
            st = [f[i][0] for i in ids]
            status_rows.append([ds, judge, len(ids), st.count("valid"), len(ids) - st.count("valid")])
            for d in (1, 5):
                budgets = BUDGETS[d]
                out, raw, cor, fold = run_set(p, counts, groups, k, d, scores)
                for mi, m in enumerate(METHODS):
                    for bi, B in enumerate(budgets):
                        abs_rows.append([ds, judge, d, B, m] + [f"{out[:, bi, mi, j].mean():.6f}" for j in range(4)])
                        cor_rows.append([ds, judge, d, B, m, f"{cor[:, bi, mi, 0].mean():.4f}", f"{cor[:, bi, mi, 1].mean():.4f}"])
                if d == 1:
                    raw_rows.append([ds, judge] + [f"{raw[:, j].mean():.6f}" for j in range(4)])
                ce = out[:, :, METHODS.index("ce"), :]
                for comp in COMPARATORS:
                    cm = out[:, :, METHODS.index(comp), :]
                    for j, metric in enumerate(METRICS[:3]):
                        delta = cm[:, :, j] - ce[:, :, j]                # positive: CE better
                        mean, lower, se = simultaneous_lower(delta, 4000, rng)
                        for bi, B in enumerate(budgets):
                            con_rows.append([ds, judge, d, B, metric, comp, f"{mean[bi]:.6f}", f"{lower[bi]:.6f}", f"{se[bi]:.6f}", int(lower[bi] > 0)])
                        # reversed: is comp (e.g. Pearson) significantly better than CE?
                        m2, l2, _ = simultaneous_lower(-delta, 4000, rng)
                        for bi, B in enumerate(budgets):
                            con_rows.append([ds, judge, d, B, metric, comp + "_reverse", f"{m2[bi]:.6f}", f"{l2[bi]:.6f}", "", int(l2[bi] > 0)])
                # raw judge forecast (no fitting) as comparator, own random stream
                rawm = np.repeat(raw[:, None, :3], len(budgets), axis=1)
                for j, metric in enumerate(METRICS[:3]):
                    delta = rawm[:, :, j] - ce[:, :, j]
                    mean, lower, se = simultaneous_lower(delta, 4000, rng_raw)
                    for bi, B in enumerate(budgets):
                        con_rows.append([ds, judge, d, B, metric, "raw", f"{mean[bi]:.6f}", f"{lower[bi]:.6f}", f"{se[bi]:.6f}", int(lower[bi] > 0)])
                    m2, l2, _ = simultaneous_lower(-delta, 4000, rng_raw)
                    for bi, B in enumerate(budgets):
                        con_rows.append([ds, judge, d, B, metric, "raw_reverse", f"{m2[bi]:.6f}", f"{l2[bi]:.6f}", "", int(l2[bi] > 0)])
            print(ds, judge, "done", flush=True)

    def w(name, header, rows):
        with open(out_dir / name, "w", newline="") as fh:
            c = csv.writer(fh); c.writerow(header); c.writerows(rows)
    w("scores_absolute.csv", ["dataset", "judge", "d", "B", "method", "rps", "ce", "tvd", "top1"], abs_rows)
    w("contrasts.csv", ["dataset", "judge", "d", "B", "metric", "comparator", "mean_comparator_minus_ce", "simultaneous_lower_975", "bootstrap_se", "lower_gt_0"], con_rows)
    w("correlations.csv", ["dataset", "judge", "d", "B", "method", "pearson_expected_vs_human_mean", "spearman_expected_vs_human_mean"], cor_rows)
    w("raw_judge_scores.csv", ["dataset", "judge", "rps", "ce", "tvd", "top1"], raw_rows)
    w("forecast_status.csv", ["dataset", "judge", "items", "valid", "uniform_fallback"], status_rows)


if __name__ == "__main__":
    main(*sys.argv[1:4])
