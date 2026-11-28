#!/usr/bin/env python3
"""Analysis of Study A v2 (see PROTOCOL_V2_11-27-2026.md section 6). CPU only.

Reads results/study_a_v2_replications.csv and the data bundle; writes results/analysis_v2.json,
results/cells_v2.csv and results/tables_v2.md.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT39 = V2.parent
sys.path.insert(0, str(ROOT39 / "src"))
import study_a as A  # noqa: E402

RES = Path(os.environ.get("V2_RES_DIR", V2 / "results"))  # override only for smoke tests
MASTER = A.MASTER
COMP_B = (64, 128, 256, 512)
N_BOOT = 10_000
DS_LABEL = {"chaosnli_snli": "ChaosNLI SNLI", "chaosnli_mnli": "ChaosNLI MNLI-m",
            "chaosnli_alphanli": "ChaosNLI alphaNLI", "dices350": "DICES-350", "dices990": "DICES-990"}
PILOT_COST = A.N_PILOT_ITEMS * A.N_PILOT_RATERS
# (id, first, second): gain of first over second = loss(second) - loss(first)
FAMILIES = [("F1", "fix_items", "fix_raters"), ("F2", "ps", "fix_raters"), ("F3", "ps", "fix_items"),
            ("F4", "ds_rel80", "fix_raters"), ("F5", "ds_rel80", "fix_items"), ("F6", "ps", "ds_rel80"),
            ("F7", "ds_rel90", "fix_items")]
MIX_FAMILIES = [("M1", "ps_mix", "fix_raters"), ("M2", "ps_mix", "fix_items")]
PRIMARY_METRICS = ("rps", "ce", "tvd")
SELECTORS = ["fix_items", "fix_raters", "ds_u1", "ds_u2", "ds_rel80", "ds_rel90", "ps", "ps_mix"]


def holm(p):
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


def twoway_bootstrap(R, rng, n_boot=N_BOOT):
    """R [n_rep, n_item] per-(replication, item) paired gains. Resample both margins independently."""
    nr, ni = R.shape
    W = rng.multinomial(nr, np.full(nr, 1.0 / nr), size=n_boot).astype(float)
    V = rng.multinomial(ni, np.full(ni, 1.0 / ni), size=n_boot).astype(float)
    s = ((W @ R) * V).sum(1) / (nr * ni)
    lo, hi = np.percentile(s, [2.5, 97.5])
    p_le = (1 + np.sum(s <= 0)) / (n_boot + 1)
    p_ge = (1 + np.sum(s >= 0)) / (n_boot + 1)
    return {"mean": float(R.mean()), "ci95": [float(lo), float(hi)], "p": float(min(1.0, 2 * min(p_le, p_ge)))}


def chosen_values(sub, B, sel, col_suffix):
    """Per-replication value of a metric for the design chosen by `sel` at budget B."""
    chosen = sub[f"B{B}_{sel}"].to_numpy()
    out = np.empty(len(sub))
    for d in np.unique(chosen):
        m = chosen == d
        out[m] = sub[f"cand_{d}_{col_suffix}"].to_numpy()[m]
    return out


def item_matrices(bundle, name):
    d = bundle[name]
    h = d["counts"] / d["counts"].sum(1, keepdims=True)
    Q = A.transforms(d["p_mix"])
    test_idx = np.asarray(d["test_idx"], int)
    dev_idx = np.asarray(d["dev_idx"], int)
    test_h = h[test_idx]
    rps, ce, tvd = A.rps_tvd_ce(Q[test_idx], test_h[:, None, :])
    K = int(d["K"])
    g = test_h.cumsum(-1)[..., :-1]
    floor = float((g * (1 - g)).mean(-1).mean())
    tot = d["counts"][dev_idx].sum(0)
    pi = (tot + 0.5) / (tot.sum() + 0.5 * K)
    base_rps, base_ce, base_tvd = A.rps_tvd_ce(np.tile(pi, (len(test_idx), 1)), test_h)
    return {"rps": rps, "ce": ce, "tvd": tvd, "floor": floor,
            "base": {"rps": float(base_rps.mean()), "ce": float(base_ce.mean()), "tvd": float(base_tvd.mean())},
            "n_test": int(len(test_idx))}


def transform_idx(sub, B, sel):
    return chosen_values(sub, B, sel, "transform").astype(int)


def main():
    df = pd.read_csv(RES / "study_a_v2_replications.csv")
    bundle = A.load_bundle()
    out = {"protocol": "PROTOCOL_V2_11-27-2026.md", "n_boot": N_BOOT, "pilot_cost_ratings": PILOT_COST,
           "comparable_budgets": list(COMP_B), "datasets": {}, "families": {}, "mix_families": {}, "checks": {}}
    rows = []
    fam_cells = {f[0]: [] for f in FAMILIES}
    fam_cells_mix = {f[0]: [] for f in MIX_FAMILIES}
    max_rebuild_err = 0.0
    for di, name in enumerate(A.DATASET_ORDER):
        sub = df[df.dataset == name].reset_index(drop=True)
        mats = item_matrices(bundle, name)
        ds = {"floor_rps": mats["floor"], "base_rate": mats["base"], "n_test": mats["n_test"], "cells": {}}
        for B in COMP_B:
            cands = [f"{a}x{b}" for a, b in A.designs_for_budget(B)]
            cell = {"candidates": cands, "pilot_cost": PILOT_COST, "pilot_over_B": PILOT_COST / B,
                    "selection": {}, "metrics": {}, "tie": {}, "agreement": {}}
            ix = {s: transform_idx(sub, B, s) for s in SELECTORS + ["or_rps"]}
            # (1) rebuild check: per-item matrices reproduce the stored replication scores
            for s in SELECTORS + ["or_rps"]:
                for m in PRIMARY_METRICS:
                    rebuilt = mats[m][:, ix[s]].mean(0)
                    stored = chosen_values(sub, B, s, m)
                    max_rebuild_err = max(max_rebuild_err, float(np.abs(rebuilt - stored).max()))
            # (2) selection distributions and tie structure
            for s in SELECTORS + ["or_rps", "or_item"]:
                vc = sub[f"B{B}_{s}"].value_counts()
                cell["selection"][s] = {"mode": str(vc.index[0]), "counts": {str(k): int(v) for k, v in vc.items()}}
            nt = sub[f"B{B}_ntie_or_rps"].to_numpy()
            cell["tie"] = {"unique_best": float(np.mean(nt == 1)), "two_tied": float(np.mean(nt == 2)),
                           "three_tied": float(np.mean(nt >= 3)), "n_reps": int(len(nt))}
            sets = [set(x.split("|")) for x in sub[f"B{B}_set_or_rps"]]
            for s in ("fix_items", "fix_raters", "ds_rel80", "ps"):
                pick = sub[f"B{B}_{s}"].to_numpy()
                inc = np.array([p in st for p, st in zip(pick, sets)], float)
                frac = np.array([(p in st) / len(st) for p, st in zip(pick, sets)], float)
                neutral = (pick == sub[f"B{B}_or_rps"].to_numpy()).astype(float)
                cell["agreement"][s] = {"neutral_pick": float(neutral.mean()), "tie_inclusive": float(inc.mean()),
                                        "fractional_credit": float(frac.mean())}
            # (3) mean losses per selector (and oracle) with replication-level t intervals
            for m in PRIMARY_METRICS + ("rps_mix", "item_mse", "grand_mse"):
                cell["metrics"][m] = {}
                for s in SELECTORS + ["or_rps", "or_item", "or_grand", "or_rps_mix"]:
                    if m == "item_mse" and s not in ("fix_items", "fix_raters", "or_item"):
                        continue
                    if m == "grand_mse" and s not in ("fix_items", "fix_raters", "or_grand"):
                        continue
                    cell["metrics"][m][s] = A.tci(chosen_values(sub, B, s, m))
            floor = mats["floor"]
            for s in ("fix_raters", "fix_items", "ds_rel80", "ps", "or_rps"):
                v = cell["metrics"]["rps"][s]["mean"]
                cell["metrics"]["rps"][s]["excess_over_floor"] = v - floor
            # (4) primary families with two-way bootstrap and replication-only t
            cell["families"] = {}
            for fi, (fid, a, b) in enumerate(FAMILIES):
                cell["families"][fid] = {"first": a, "second": b}
                for mi, m in enumerate(PRIMARY_METRICS):
                    R = mats[m][:, ix[b]].T - mats[m][:, ix[a]].T  # [rep, item]: loss(second) - loss(first)
                    rng = np.random.default_rng(np.random.SeedSequence([MASTER, 9, di, B, fi, mi]))
                    tw = twoway_bootstrap(R, rng)
                    drep = R.mean(1)
                    tt = A.tci(drep)
                    p_t = float(stats.ttest_1samp(drep, 0.0).pvalue) if np.isfinite(drep).all() and drep.std() > 0 else 1.0
                    tw.update({"t_ci95": tt["ci95"], "t_p": p_t,
                               "pct_of_total_rps": None, "pct_of_excess": None})
                    if m == "rps":
                        la = cell["metrics"]["rps"][a]["mean"]
                        lb = cell["metrics"]["rps"][b]["mean"]
                        tw["pct_of_total_rps"] = 100 * tw["mean"] / lb
                        tw["pct_of_excess"] = 100 * tw["mean"] / (lb - floor)
                    cell["families"][fid][m] = tw
                fam_cells[fid].append((name, B, cell["families"][fid]))
            # (5) secondary mixture families (replication-level only)
            cell["mix_families"] = {}
            for fid, a, b in MIX_FAMILIES:
                d = chosen_values(sub, B, b, "rps_mix") - chosen_values(sub, B, a, "rps_mix")
                tt = A.tci(d)
                p_t = float(stats.ttest_1samp(d, 0.0).pvalue) if d.std() > 0 else 1.0
                cell["mix_families"][fid] = {"first": a, "second": b, "mean": float(d.mean()), "t_ci95": tt["ci95"], "t_p": p_t}
                fam_cells_mix[fid].append((name, B, cell["mix_families"][fid]))
            # (6) analytic sanity rows: mean-estimation targets (replication-level only)
            cell["analytic"] = {}
            for m in ("item_mse", "grand_mse"):
                d = chosen_values(sub, B, "fix_items", m) - chosen_values(sub, B, "fix_raters", m)
                tt = A.tci(d)
                p_t = float(stats.ttest_1samp(d, 0.0).pvalue) if d.std() > 0 else 1.0
                cell["analytic"][m] = {"fix_items_minus_fix_raters": float(d.mean()), "t_ci95": tt["ci95"], "t_p": p_t}
            # (7) analytic-equality checks
            cell["ds_u1_equals_fix_raters"] = bool((sub[f"B{B}_ds_u1"] == sub[f"B{B}_fix_raters"]).all())
            cell["ds_u2_equals_fix_items"] = bool((sub[f"B{B}_ds_u2"] == sub[f"B{B}_fix_items"]).all())
            ds["cells"][str(B)] = cell
        out["datasets"][name] = ds
    # Holm within family across the 20 comparable cells
    for fid, a, b in FAMILIES:
        out["families"][fid] = {"first": a, "second": b, "metrics": {}}
        for m in PRIMARY_METRICS:
            ps = np.array([c[m]["p"] for _, _, c in fam_cells[fid]])
            pt = np.array([c[m]["t_p"] for _, _, c in fam_cells[fid]])
            adj, adj_t = holm(ps), holm(pt)
            means = np.array([c[m]["mean"] for _, _, c in fam_cells[fid]])
            for (name, B, c), pa, pta in zip(fam_cells[fid], adj, adj_t):
                c[m]["p_holm"] = float(pa)
                c[m]["t_p_holm"] = float(pta)
                c[m]["first_beats_second_holm"] = bool(pa < 0.05 and c[m]["mean"] > 0)
                c[m]["second_beats_first_holm"] = bool(pa < 0.05 and c[m]["mean"] < 0)
            out["families"][fid]["metrics"][m] = {
                "cells": len(ps),
                "mean_gain_positive_cells": int((means > 0).sum()),
                "first_beats_second_raw": int(((ps < 0.05) & (means > 0)).sum()),
                "second_beats_first_raw": int(((ps < 0.05) & (means < 0)).sum()),
                "first_beats_second_holm": int(((adj < 0.05) & (means > 0)).sum()),
                "second_beats_first_holm": int(((adj < 0.05) & (means < 0)).sum()),
                "first_beats_second_t_holm": int(((adj_t < 0.05) & (means > 0)).sum()),
                "second_beats_first_t_holm": int(((adj_t < 0.05) & (means < 0)).sum()),
            }
    for fid, a, b in MIX_FAMILIES:
        pt = np.array([c["t_p"] for _, _, c in fam_cells_mix[fid]])
        adj = holm(pt)
        means = np.array([c["mean"] for _, _, c in fam_cells_mix[fid]])
        for (_, _, c), pa in zip(fam_cells_mix[fid], adj):
            c["t_p_holm"] = float(pa)
        out["mix_families"][fid] = {"first": a, "second": b,
                                    "first_beats_second_raw": int(((pt < 0.05) & (means > 0)).sum()),
                                    "first_beats_second_holm": int(((adj < 0.05) & (means > 0)).sum()),
                                    "second_beats_first_holm": int(((adj < 0.05) & (means < 0)).sum())}
    # analytic sanity: Holm on the mean-estimation rows
    for m in ("item_mse", "grand_mse"):
        pt, cells = [], []
        for name in A.DATASET_ORDER:
            for B in COMP_B:
                c = out["datasets"][name]["cells"][str(B)]["analytic"][m]
                pt.append(c["t_p"])
                cells.append(c)
        adj = holm(pt)
        for c, pa in zip(cells, adj):
            c["t_p_holm"] = float(pa)
        means = np.array([c["fix_items_minus_fix_raters"] for c in cells])
        out["checks"][f"analytic_{m}"] = {"fix_raters_better_holm": int(((adj < 0.05) & (means > 0)).sum()),
                                           "fix_items_better_holm": int(((adj < 0.05) & (means < 0)).sum()),
                                           "cells": len(cells)}
    out["checks"]["max_abs_error_rebuilding_replication_scores_from_per_item_matrices"] = max_rebuild_err
    out["checks"]["ds_u1_equals_fix_raters_all_cells"] = all(
        out["datasets"][n]["cells"][str(B)]["ds_u1_equals_fix_raters"] for n in A.DATASET_ORDER for B in COMP_B)
    out["checks"]["ds_u2_equals_fix_items_all_cells"] = all(
        out["datasets"][n]["cells"][str(B)]["ds_u2_equals_fix_items"] for n in A.DATASET_ORDER for B in COMP_B)
    ties = [out["datasets"][n]["cells"][str(B)]["tie"]["unique_best"] for n in A.DATASET_ORDER for B in COMP_B]
    out["checks"]["mean_unique_best_fraction_or_rps"] = float(np.mean(ties))
    (RES / "analysis_v2.json").write_text(json.dumps(out, indent=2) + "\n")

    # flat cell table
    for name in A.DATASET_ORDER:
        for B in COMP_B:
            c = out["datasets"][name]["cells"][str(B)]
            r = {"dataset": name, "B": B, "pilot_cost": PILOT_COST, "P_over_B": PILOT_COST / B,
                 "floor_rps": out["datasets"][name]["floor_rps"], "base_rate_rps": out["datasets"][name]["base_rate"]["rps"]}
            for s in ("fix_raters", "fix_items", "ds_rel80", "ps", "or_rps"):
                r[f"rps_{s}"] = c["metrics"]["rps"][s]["mean"]
            for s in ("ds_rel80", "ps", "fix_items", "fix_raters"):
                r[f"mode_{s}"] = c["selection"][s]["mode"]
                r[f"agree_incl_{s}"] = c["agreement"][s]["tie_inclusive"]
            r["unique_best_or_rps"] = c["tie"]["unique_best"]
            for fid, _, _ in FAMILIES:
                f = c["families"][fid]["rps"]
                r[f"{fid}_gain"] = f["mean"]
                r[f"{fid}_ci_lo"], r[f"{fid}_ci_hi"] = f["ci95"]
                r[f"{fid}_p"] = f["p"]
                r[f"{fid}_p_holm"] = f["p_holm"]
            rows.append(r)
    pd.DataFrame(rows).to_csv(RES / "cells_v2.csv", index=False)
    write_tables(out)
    print(json.dumps(out["checks"], indent=2))


def fmt_p(p):
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def write_tables(out):
    L = []
    L.append("# Tables for Study A v2 (generated by `src/analyze_v2.py`; do not edit by hand)\n")
    L.append(f"Pilot cost $P={PILOT_COST}$ ratings is charged to every pilot-using selector. Gains are loss(second) minus loss(first); positive means the first selector is better. Intervals and $p$ are from the replication $\\times$ test-item two-way bootstrap ({N_BOOT:,} draws); Holm adjusts across the 20 cells within a family.\n")
    L.append("## Table 1. Mean held-out test RPS by selector (lower is better); floor and base-rate references\n")
    L.append("| Dataset | $B$ | $P/B$ | floor | base rate | `fix_raters` | `fix_items` | `ds_rel80` | `ps` | neutral RPS-oracle |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in A.DATASET_ORDER:
        d = out["datasets"][name]
        for B in COMP_B:
            c = d["cells"][str(B)]
            m = c["metrics"]["rps"]
            L.append(f"| {DS_LABEL[name]} | {B} | {PILOT_COST / B:.1f} | {d['floor_rps']:.4f} | {d['base_rate']['rps']:.4f} | "
                     f"{m['fix_raters']['mean']:.4f} | {m['fix_items']['mean']:.4f} | {m['ds_rel80']['mean']:.4f} | {m['ps']['mean']:.4f} | {m['or_rps']['mean']:.4f} |")
    for fid, a, b in FAMILIES:
        L.append(f"\n## Table {fid}. Gain of `{a}` over `{b}` in test RPS (positive: `{a}` better)\n")
        L.append("| Dataset | $B$ | gain [95% CI] | % of total RPS | % of reducible excess | $p$ | Holm $p$ | CE gain Holm $p$ | TVD gain Holm $p$ | verdict |")
        L.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
        for name in A.DATASET_ORDER:
            for B in COMP_B:
                f = out["datasets"][name]["cells"][str(B)]["families"][fid]
                r = f["rps"]
                v = f"`{a}` beats" if r["first_beats_second_holm"] else (f"`{b}` beats" if r["second_beats_first_holm"] else "not significant after Holm")
                L.append(f"| {DS_LABEL[name]} | {B} | {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] | {r['pct_of_total_rps']:+.2f} | {r['pct_of_excess']:+.1f} | "
                         f"{fmt_p(r['p'])} | {fmt_p(r['p_holm'])} | {fmt_p(f['ce']['p_holm'])} | {fmt_p(f['tvd']['p_holm'])} | {v} |")
        s = out["families"][fid]["metrics"]["rps"]
        L.append(f"\nCounts over 20 cells: first better by mean {s['mean_gain_positive_cells']}; first beats second raw {s['first_beats_second_raw']}, after Holm {s['first_beats_second_holm']}; second beats first raw {s['second_beats_first_raw']}, after Holm {s['second_beats_first_holm']}. (CE: first beats after Holm {out['families'][fid]['metrics']['ce']['first_beats_second_holm']}; TVD: {out['families'][fid]['metrics']['tvd']['first_beats_second_holm']}.)\n")
    L.append("\n## Table S. Selected designs (mode over 150 replications) and oracle tie structure\n")
    L.append("| Dataset | $B$ | `fix_raters` | `fix_items` | `ds_rel80` mode (share) | `ds_rel90` mode | `ps` mode (share) | unique-best share of RPS-oracle | tie-inclusive agreement: `ds_rel80` / `ps` / `fix_items` / `fix_raters` |")
    L.append("|---|---:|---|---|---|---|---|---:|---|")
    for name in A.DATASET_ORDER:
        for B in COMP_B:
            c = out["datasets"][name]["cells"][str(B)]
            def mode(s):
                sel = c["selection"][s]
                return f"{sel['mode']} ({sel['counts'][sel['mode']] / c['tie']['n_reps']:.2f})"
            ag = c["agreement"]
            L.append(f"| {DS_LABEL[name]} | {B} | {c['selection']['fix_raters']['mode']} | {c['selection']['fix_items']['mode']} | {mode('ds_rel80')} | {c['selection']['ds_rel90']['mode']} | {mode('ps')} | "
                     f"{c['tie']['unique_best']:.2f} | {ag['ds_rel80']['tie_inclusive']:.2f} / {ag['ps']['tie_inclusive']:.2f} / {ag['fix_items']['tie_inclusive']:.2f} / {ag['fix_raters']['tie_inclusive']:.2f} |")
    L.append("\n## Table A. Analytic sanity rows (mean-estimation targets; replication-only $t$; not RPS)\n")
    L.append("| Dataset | $B$ | item-mean MSE: `fix_items` minus `fix_raters` [95% CI] | Holm $p$ | grand-mean MSE: `fix_items` minus `fix_raters` [95% CI] | Holm $p$ |")
    L.append("|---|---:|---|---:|---|---:|")
    for name in A.DATASET_ORDER:
        for B in COMP_B:
            a = out["datasets"][name]["cells"][str(B)]["analytic"]
            L.append(f"| {DS_LABEL[name]} | {B} | {a['item_mse']['fix_items_minus_fix_raters']:+.5f} [{a['item_mse']['t_ci95'][0]:+.5f}, {a['item_mse']['t_ci95'][1]:+.5f}] | {fmt_p(a['item_mse']['t_p_holm'])} | "
                     f"{a['grand_mse']['fix_items_minus_fix_raters']:+.5f} [{a['grand_mse']['t_ci95'][0]:+.5f}, {a['grand_mse']['t_ci95'][1]:+.5f}] | {fmt_p(a['grand_mse']['t_p_holm'])} |")
    (RES / "tables_v2.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
