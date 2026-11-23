"""Break-even tables for experiment 28 (11-23-2026). Usage: python3 summarize.py RESULTS_DIR
Writes breakeven_summary.csv (formal N*, sustained lower bound > 0 on RPS), pearson_wins.csv (every cell where a comparator is
significantly better than CE-fit) and metric_agreement.csv (the same N* on CE and TVD)."""
import sys
from pathlib import Path
import pandas as pd

BUD = {1: [4, 8, 16, 32, 64, 128, 256], 5: [10, 20, 40, 80, 160, 320]}
COMPS = ("pearson", "uniform", "both", "hist", "pcal", "raw", "rps")


def nstar(g):
    g = g.sort_values("B"); ok = g.lower_gt_0.values == 1
    for i in range(len(ok)):
        if ok[i:].all():
            return int(g.B.values[i])
    return None


def table(c, metric):
    rows = []
    for (ds, j, d), g in c[c.metric == metric].groupby(["dataset", "judge", "d"]):
        r = {"dataset": ds, "judge": j, "d": d}
        for comp in ("pearson", "uniform", "hist", "pcal", "raw", "rps"):
            r["N_" + comp] = nstar(g[g.comparator == comp])
        both = g[g.comparator.isin(["pearson", "uniform"])].groupby("B").lower_gt_0.min().reset_index()
        r["N_both"] = nstar(both)
        rows.append(r)
    return pd.DataFrame(rows)


def main(res):
    res = Path(res); c = pd.read_csv(res / "contrasts.csv")
    t = table(c, "rps"); t.to_csv(res / "breakeven_summary.csv", index=False)
    pd.concat([table(c, m).assign(metric=m) for m in ("rps", "ce", "tvd")]).to_csv(res / "metric_agreement.csv", index=False)
    w = c[c.comparator.str.endswith("_reverse") & (c.lower_gt_0 == 1)].copy()
    w["comparator"] = w.comparator.str.replace("_reverse", "")
    w = w.rename(columns={"mean_comparator_minus_ce": "mean_ce_minus_comparator", "simultaneous_lower_975": "lower_975_comparator_better"})
    w.drop(columns=["bootstrap_se", "lower_gt_0"]).to_csv(res / "comparator_wins.csv", index=False)
    print(t.to_string()); print(len(w), "comparator-significantly-better cells")
    print(w[w.metric == "rps"].groupby(["comparator", "dataset"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main(sys.argv[1])
