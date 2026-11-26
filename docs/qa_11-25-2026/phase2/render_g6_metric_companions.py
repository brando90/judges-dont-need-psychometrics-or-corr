"""Write dated metric-first companions without changing scored sources, 11-26-2026.

Old correlation-only scored tables remain preserved. Every join requires an
exact metadata match; missing values raise rather than being invented.
"""
import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[3]
E = {n: next((ROOT / "experiments").glob(f"{n}_*")) for n in (28, 29, 31)}
read = lambda p: list(csv.DictReader(p.open()))
scores28 = read(E[28] / "results/scores_absolute.csv")
cor28 = read(E[28] / "results/correlations.csv")
key = lambda r: tuple(r[k] for k in ("dataset", "judge", "d", "B", "method"))
score_lookup = {key(r): r for r in scores28}
assert len(score_lookup) == len(scores28)
header = """# Experiment 28: complete metric-first companion (11-26-2026)

As of 11-26-2026. Finding g6-L1. This is a new presentation of the 1,950 historical correlation rows; the original scored CSVs remain unchanged.

Held-out ranked probability score (RPS, normalized 0–1), cross-entropy (CE, nats), and total variation distance (TVD, 0–1 against the empirical human distribution) come before the secondary Spearman and Pearson correlations of predicted and human means. Lower RPS/CE/TVD is better; larger correlation means stronger association. RPS and CE are proper scores; TVD is a distribution diagnostic. `B` is total training ratings; `d` is ratings per item. These saved point estimates average five held-out folds and 200 training draws. The same fixed judge is transformed by an exhaustive 27-setting head search for CE, RPS or Pearson objectives; `pcal` adds distribution calibration to the Pearson-selected head. The uniform reference is RPS = 0.25 for binary alphaNLI and CE = log(K); other reference rows are in `scores_absolute.csv`.

Source: [scores_absolute.csv](results/scores_absolute.csv) joined to the preserved [correlations.csv](results/correlations.csv) on dataset, judge, d, B and method. No new uncertainty estimate was computed; original simultaneous bands remain in [contrasts.csv](results/contrasts.csv). Raw-label verification remains limited as explained in the [dated QA note](QA_CORRECTIONS_11-26-2026.md).

| Dataset | Judge | d | B | Method | RPS | CE (nats) | TVD | Spearman | Pearson |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
"""
lines = []
for r in sorted(cor28, key=lambda r: (r["dataset"], r["judge"], int(r["d"]), int(r["B"]), r["method"])):
    s = score_lookup[key(r)]
    vals = [*key(r), s["rps"], s["ce"], s["tvd"], r["spearman_expected_vs_human_mean"], r["pearson_expected_vs_human_mean"]]
    lines.append("| " + " | ".join(vals) + " |")
assert len(lines) == 1950
(E[28] / "QA_METRIC_PANEL_11-26-2026.md").write_text(header + "\n".join(lines) + "\n")

header31 = """# Experiment 31: metric-first companion for historical Table 5 (11-26-2026)

As of 11-26-2026. Finding g6-L2. The old 12-row correlation-only table remains preserved; this companion displays both methods in each panel with all three evaluation metrics first.

Held-out ranked probability score (RPS, normalized 0–1), cross-entropy (CE, nats) and total variation distance (TVD, 0–1 against empirical human ratings) are lower-is-better. Spearman and Pearson measure secondary association of predicted and human means. RPS and CE are proper scores; TVD is a distribution diagnostic. Values are saved point estimates: Exp 28 is the mean of five fixed frontier judges, exhaustive 27-transform head search, five held-out folds and 200 training draws; Exp 29 is raw LoRA training over ten nested draws, as described in its frozen protocol. `B` is the largest available total rating budget for that panel, and `d` is ratings per item. Exp 28 uniform CE is log(K) (alphaNLI RPS 0.25); Exp 29 uniform RPS ranges from 0.20388 to 0.26936. These are descriptive means with no newly computed confidence interval.

Sources: Exp 28 `results/scores_absolute.csv` and `results/correlations.csv`; Exp 29 `results/scores.csv`. Each method is joined by its exact panel and budget. Raw-label evidence limitations remain in the [dated QA note](QA_CORRECTIONS_11-26-2026.md).

| Source / panel | d | B | Method | RPS | CE (nats) | TVD | Spearman | Pearson |
|---|---:|---:|---|---:|---:|---:|---:|---:|
"""
cor_lookup = {key(r): r for r in cor28}
lines31 = []
for ds in ("chaosnli_snli", "chaosnli_mnli", "chaosnli_alphanli", "dices350", "dices990"):
    for method in ("ce", "pearson"):
        ss = [r for r in scores28 if r["dataset"] == ds and r["d"] == "1" and r["B"] == "256" and r["method"] == method]
        assert len(ss) == 5
        vals = [f"Exp 28 / {ds} / five-judge mean", "1", "256", method]
        vals += [f"{mean(float(r[k]) for r in ss):.10f}" for k in ("rps", "ce", "tvd")]
        vals += [f"{mean(float(cor_lookup[key(r)][k]) for r in ss):.10f}" for k in ("spearman_expected_vs_human_mean", "pearson_expected_vs_human_mean")]
        lines31.append("| " + " | ".join(vals) + " |")
scores29 = read(E[29] / "results/scores.csv")
for study, depth in (("chaos3_qwen", "1"), ("chaos3_qwen", "5"), ("alpha2_qwen", "1"), ("dices3_qwen", "1"), ("hs2_qwen", "1"), ("chaos3_llama", "1"), ("dices3_llama", "1")):
    for method in ("ce_raw", "pearson_raw"):
        r = max((r for r in scores29 if r["study"] == study and r["depth"] == depth and r["method"] == method), key=lambda r: int(r["b_total"]))
        vals = [f"Exp 29 / {study}", depth, r["b_total"], method]
        vals += [r[k] for k in ("rps", "ce", "tvd", "spearman", "pearson")]
        lines31.append("| " + " | ".join(vals) + " |")
assert len(lines31) == 24
(E[31] / "QA_METRIC_PANEL_11-26-2026.md").write_text(header31 + "\n".join(lines31) + "\n")
print("Metric companions (11-26-2026): 1,950 Exp 28 rows and 24 Exp 31 rows; all source scores preserved.")
