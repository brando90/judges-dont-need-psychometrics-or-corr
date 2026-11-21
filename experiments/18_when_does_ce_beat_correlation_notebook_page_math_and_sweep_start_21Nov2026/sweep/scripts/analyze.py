"""Exp 18 analysis of sweeps A and B (paired differences, bootstrap over seeds, crossover n, the 6 pre-registered predictions). Date 11-21-2026.
Run: python3 analyze.py --rawA ../results/rawA.jsonl --rawB ../results/rawB.jsonl --out ../results
Writes summary_A.csv, summary_B.csv, tables.md, predictions.json. A difference is "other minus CE" (positive = CE better, lower metric is better)."""
import argparse, json, collections, csv, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--rawA"); ap.add_argument("--rawB"); ap.add_argument("--out", required=True)
a = ap.parse_args(); rng = np.random.default_rng(0); NB = 2000

def boot_ci(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    if len(d) == 0: return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(d), (NB, len(d))); m = d[idx].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

def verdict(dr, dc):
    """dr, dc: other-minus-CE paired differences on RPS and CE excess. CE wins only if both CIs are above 0."""
    r, c = boot_ci(dr), boot_ci(dc)
    if r[1] > 0 and c[1] > 0: v = "CE"
    elif r[2] < 0 and c[2] < 0: v = "other"
    elif r[1] <= 0 <= r[2] and c[1] <= 0 <= c[2]: v = "tie"
    else: v = "mixed"
    return v, r, c

def load(path, keyf):
    cells = collections.defaultdict(dict)
    for l in open(path):
        if not l.strip(): continue
        r = json.loads(l); cells[keyf(r)][r["seed"]] = r["fits"]
    return cells

def arr(seeds, arm, met):
    return np.array([seeds[s][arm][met] if arm in seeds[s] else np.nan for s in sorted(seeds)], float)

out = {}; md = []
# ---------------- Sweep A
if a.rawA and os.path.exists(a.rawA):
    A = load(a.rawA, lambda r: (r["scenario"], r["n"], r["k"], r["J"]))
    rows = []
    for (s, n, k, J), sd in sorted(A.items()):
        row = dict(scenario=s, n=n, k=k, J=J, seeds=len(sd), true_rps=float(np.nanmean(arr(sd, "ce_tuned", "rps")) * 0 + np.nanmean([sd[x]["true_rps"] for x in sd])))
        for arm in ["constant", "ce_tuned", "rps_tuned", "mse_tuned", "pearson_raw", "pearson_head_tuned"]:
            for m in ["rps", "ce_excess", "tvd", "spearman", "pearson"]:
                row[f"{arm}.{m}"] = float(np.nanmean(arr(sd, arm, m))) if not np.all(np.isnan(arr(sd, arm, m))) else np.nan
        for other in ["pearson_raw", "pearson_head_tuned", "mse_tuned", "rps_tuned", "constant"]:
            dr = arr(sd, other, "rps") - arr(sd, "ce_tuned", "rps"); dc = arr(sd, other, "ce_excess") - arr(sd, "ce_tuned", "ce_excess")
            v, r, c = verdict(dr, dc); dt = arr(sd, other, "tvd") - arr(sd, "ce_tuned", "tvd")
            row[f"CEvs_{other}.verdict"] = v; row[f"CEvs_{other}.dRPS"] = r[0]; row[f"CEvs_{other}.dRPS_lo"] = r[1]; row[f"CEvs_{other}.dRPS_hi"] = r[2]
            row[f"CEvs_{other}.dCE"] = c[0]; row[f"CEvs_{other}.dTVD"] = float(np.nanmean(dt)); row[f"CEvs_{other}.winrate_rps"] = float(np.nanmean(dr > 0))
        rows.append(row)
    with open(os.path.join(a.out, "summary_A.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    R = {(r["scenario"], r["n"], r["k"], r["J"]): r for r in rows}
    ncell = len(rows); out["A_cells"] = ncell; out["A_seeds_min"] = min(r["seeds"] for r in rows)
    # P1
    lose = [r["pearson_raw.rps"] > r["constant.rps"] for r in rows]; out["P1_pearson_loses_to_constant_frac"] = float(np.mean(lose))
    # P2
    def sub(sc, cond): return [r for r in rows if r["scenario"] in sc and cond(r)]
    for nm, rs in [("AB_J6_n>=64", sub("AB", lambda r: r["J"] == 6 and r["n"] >= 64)), ("AB_J6_n<=16", sub("AB", lambda r: r["J"] == 6 and r["n"] <= 16)),
                   ("CD_J1_all_n", sub("CD", lambda r: r["J"] == 1))]:
        vs = collections.Counter(r["CEvs_pearson_head_tuned.verdict"] for r in rs)
        out[f"P2_{nm}"] = dict(verdicts=dict(vs), mean_dRPS=float(np.mean([r["CEvs_pearson_head_tuned.dRPS"] for r in rs])),
                               max_abs_dRPS=float(np.max([abs(r["CEvs_pearson_head_tuned.dRPS"]) for r in rs])))
    # P3 crossover
    def crossover(s, k, J, other):
        ns = sorted(n for (ss, n, kk, jj) in R if (ss, kk, jj) == (s, k, J))
        for i, n in enumerate(ns):
            if all(R[(s, m, k, J)][f"CEvs_{other}.verdict"] == "CE" for m in ns[i:]): return n
        return None
    for other in ["pearson_head_tuned", "pearson_raw"]:
        co = {(s, k, J): crossover(s, k, J, other) for s in "ABCD" for k in [1, 3, 10] for J in [1, 3, 6]}
        out[f"P3_crossover_{other}"] = {f"{s}_k{k}_J{J}": v for (s, k, J), v in co.items()}
        for k in [1, 3, 10]:
            vals = [v for (s, kk, J), v in co.items() if kk == k]
            out[f"P3_{other}_k{k}_never"] = sum(v is None for v in vals); out[f"P3_{other}_k{k}_median_n"] = float(np.median([v for v in vals if v is not None])) if any(v is not None for v in vals) else None
    # P4
    ok = [r["ce_tuned.spearman"] >= r["pearson_raw.spearman"] - 0.02 for r in rows if not np.isnan(r["ce_tuned.spearman"]) and not np.isnan(r["pearson_raw.spearman"])]
    out["P4_ce_spearman_within_0.02_of_pearson_trained_frac"] = float(np.mean(ok))
    ok2 = [r["ce_tuned.spearman"] >= r["pearson_head_tuned.spearman"] - 0.02 for r in rows if not np.isnan(r["ce_tuned.spearman"]) and not np.isnan(r["pearson_head_tuned.spearman"])]
    out["P4b_ce_spearman_within_0.02_of_pearson_head_frac"] = float(np.mean(ok2))
    # tables
    md.append("## Sweep A, verdicts CE-tuned vs each other arm (cells = scenario x n x k x J; CE wins = both RPS and CE-excess intervals above 0)\n")
    for other in ["pearson_raw", "pearson_head_tuned", "mse_tuned", "rps_tuned", "constant"]:
        md.append(f"\n### CE-tuned vs {other}: counts by n (CE / tie / other / mixed)\n\n| n | CE | tie | other | mixed |\n|---|---|---|---|---|")
        for n in sorted({r["n"] for r in rows}):
            c = collections.Counter(r[f"CEvs_{other}.verdict"] for r in rows if r["n"] == n)
            md.append(f"| {n} | {c['CE']} | {c['tie']} | {c['other']} | {c['mixed']} |")
    md.append("\n### Mean test RPS excess over true-distribution RPS floor (all scenarios, k=3, J=3), by n\n\n| n | constant | pearson_raw | pearson_head | mse | ce | rps |\n|---|---|---|---|---|---|---|")
    for n in sorted({r["n"] for r in rows}):
        rs = [r for r in rows if r["n"] == n and r["k"] == 3 and r["J"] == 3]
        f = lambda arm: np.mean([r[f"{arm}.rps"] - r["true_rps"] for r in rs])
        md.append(f"| {n} | {f('constant'):.4f} | {f('pearson_raw'):.4f} | {f('pearson_head_tuned'):.4f} | {f('mse_tuned'):.4f} | {f('ce_tuned'):.4f} | {f('rps_tuned'):.4f} |")
    md.append("\n### Crossover n (smallest n from which CE-tuned beats the Pearson-selected+calibrated arm at every larger n; '-' = never)\n\n| scenario | k | J=1 | J=3 | J=6 |\n|---|---|---|---|---|")
    co = out["P3_crossover_pearson_head_tuned"]
    for s in "ABCD":
        for k in [1, 3, 10]: md.append(f"| {s} | {k} | " + " | ".join(str(co[f'{s}_k{k}_J{J}'] or '-') for J in [1, 3, 6]) + " |")
# ---------------- Sweep B
if a.rawB and os.path.exists(a.rawB):
    B = load(a.rawB, lambda r: (r["scenario"], r["pool"], r["k"], r["n"]))
    rows = []
    for (s, p, k, n), sd in sorted(B.items()):
        row = dict(scenario=s, pool=p, k=k, n=n, seeds=len(sd))
        for arm in ["constant", "corr_pearson", "corr_spearman", "ce", "rps", "oracle_rps"]:
            for m in ["rps", "ce_excess", "tvd", "spearman", "pearson"]:
                x = arr(sd, arm, m); row[f"{arm}.{m}"] = float(np.nanmean(x)) if not np.all(np.isnan(x)) else np.nan
        for sel in ["ce", "rps", "corr_spearman"]:
            dr = arr(sd, "corr_pearson", "rps") - arr(sd, sel, "rps"); dc = arr(sd, "corr_pearson", "ce_excess") - arr(sd, sel, "ce_excess")
            v, r, c = verdict(dr, dc); row[f"{sel}_vs_pearson.verdict"] = v; row[f"{sel}_vs_pearson.dRPS"] = r[0]; row[f"{sel}_vs_pearson.dCE"] = c[0]
        rows.append(row)
    with open(os.path.join(a.out, "summary_B.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    out["B_cells"] = len(rows)
    def cnt(cond, sel, v): rs = [r for r in rows if cond(r)]; return dict(collections.Counter(r[f"{sel}_vs_pearson.verdict"] for r in rs)), len(rs)
    for pool in ["prob", "votes3", "votes3_f5e-2"]:
        for sel in ["ce", "rps"]:
            for nm, cond in [("n>=16", lambda r: r["pool"] == pool and r["n"] >= 16), ("n<=8", lambda r: r["pool"] == pool and r["n"] <= 8)]:
                vs, tot = cnt(cond, sel, None)
                rs = [r for r in rows if cond(r)]
                out[f"P5_{pool}_{sel}_{nm}"] = dict(verdicts=vs, cells=tot, mean_dRPS=float(np.mean([r[f"{sel}_vs_pearson.dRPS"] for r in rs])))
    md.append("\n## Sweep B, selection of one of 24 candidate judges: CE-selection vs Pearson-selection, counts (CE / tie / other / mixed) by pool and n; cells = scenario x k\n\n| pool | n | CE-sel better | tie | Pearson better | mixed | mean dRPS (Pearson-sel minus CE-sel) |\n|---|---|---|---|---|---|---|")
    for pool in ["prob", "votes3", "votes3_f5e-2"]:
        for n in sorted({r["n"] for r in rows}):
            rs = [r for r in rows if r["pool"] == pool and r["n"] == n]; c = collections.Counter(r["ce_vs_pearson.verdict"] for r in rs)
            md.append(f"| {pool} | {n} | {c['CE']} | {c['tie']} | {c['other']} | {c['mixed']} | {np.mean([r['ce_vs_pearson.dRPS'] for r in rs]):+.4f} |")
    md.append("\n### Sweep B, same for RPS-selection vs Pearson-selection\n\n| pool | n | RPS-sel better | tie | Pearson better | mixed | mean dRPS |\n|---|---|---|---|---|---|---|")
    for pool in ["prob", "votes3", "votes3_f5e-2"]:
        for n in sorted({r["n"] for r in rows}):
            rs = [r for r in rows if r["pool"] == pool and r["n"] == n]; c = collections.Counter(r["rps_vs_pearson.verdict"] for r in rs)
            md.append(f"| {pool} | {n} | {c['CE']} | {c['tie']} | {c['other']} | {c['mixed']} | {np.mean([r['rps_vs_pearson.dRPS'] for r in rs]):+.4f} |")
json.dump(out, open(os.path.join(a.out, "predictions.json"), "w"), indent=1, default=float)
open(os.path.join(a.out, "tables.md"), "w").write("\n".join(md) + "\n")
print(json.dumps(out, indent=1, default=float)[:6000])
