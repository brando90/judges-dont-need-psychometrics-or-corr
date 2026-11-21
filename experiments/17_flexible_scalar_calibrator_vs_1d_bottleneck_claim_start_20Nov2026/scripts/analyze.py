"""Experiment 17 summary: mean test RPS excess over the true RPS (smaller is better) per scenario, n and arm, with seed standard errors, plus the paired
gaps used by the pre-registered predictions. Usage: python3 analyze.py ../results/raw.jsonl > ../results/summary.md"""
import json, sys
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ARMS = ["constant", "ce_fixed", "ce_valrps", "pearson_fixed", "pearson_head_tuned", "pearson_head_flex", "mse_head_tuned", "mse_head_flex", "truemean_head_affine", "truemean_head_flex"]
NS = sorted({r["n"] for r in rows}); SC = "ABCD"
def col(sc, n, arm, key="rps_excess"):
    return np.array([r["fits"][arm][key] for r in rows if r["scenario"] == sc and r["n"] == n])
def paired(sc, n, a, b, key="rps_excess"):
    d = {}
    for r in rows:
        if r["scenario"] == sc and r["n"] == n: d[r["seed"]] = r["fits"][a][key] - r["fits"][b][key]
    v = np.array(list(d.values())); return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v)
print(f"Cells read: {len(rows)} of 4400.\n")
for sc in SC:
    print(f"## Scenario {sc}: mean test RPS excess over truth (smaller is better; seed SE in the paired table)\n")
    print("| n | " + " | ".join(ARMS) + " |"); print("|---|" + "---|" * len(ARMS))
    for n in NS:
        if len(col(sc, n, "constant")) == 0: continue
        print(f"| {n} | " + " | ".join(f"{col(sc,n,a).mean():.4f}" for a in ARMS) + " |")
    print()
print("## Paired gaps (arm minus end-to-end CE tuned `ce_valrps`; negative = arm better), mean [mean +/- 2 SE], n >= 512\n")
print("| Scenario | n | pearson_head_flex | mse_head_flex | truemean_head_flex | truemean_head_affine | pearson_head_tuned |"); print("|---|---|---|---|---|---|---|")
for sc in SC:
    for n in [512, 1024, 2048]:
        cells = []
        for a in ["pearson_head_flex", "mse_head_flex", "truemean_head_flex", "truemean_head_affine", "pearson_head_tuned"]:
            m, se, k = paired(sc, n, a, "ce_valrps"); cells.append(f"{m:+.4f} [{m-2*se:+.4f}, {m+2*se:+.4f}]")
        print(f"| {sc} | {n} | " + " | ".join(cells) + " |")
print("\n## Paired gaps, flexible minus affine head (negative = flexible better)\n")
print("| Scenario | n | oracle score | Pearson score | MSE score |"); print("|---|---|---|---|---|")
for sc in SC:
    for n in [16, 32, 64, 256, 1024, 2048]:
        cells = []
        for a, b in (("truemean_head_flex", "truemean_head_affine"), ("pearson_head_flex", "pearson_head_tuned"), ("mse_head_flex", "mse_head_tuned")):
            m, se, k = paired(sc, n, a, b); cells.append(f"{m:+.4f} [{m-2*se:+.4f}, {m+2*se:+.4f}]")
        print(f"| {sc} | {n} | " + " | ".join(cells) + " |")
print("\n## Secondary columns at n = 2048 (mean test Spearman / Pearson with the true mean; CE excess)\n")
print("| Scenario | arm | Spearman | Pearson | CE excess |"); print("|---|---|---|---|---|")
for sc in SC:
    for a in ["ce_valrps", "pearson_head_flex", "truemean_head_flex", "truemean_head_affine"]:
        print(f"| {sc} | {a} | {col(sc,2048,a,'spearman').mean():.3f} | {col(sc,2048,a,'pearson').mean():.3f} | {col(sc,2048,a,'ce_excess').mean():.4f} |")
