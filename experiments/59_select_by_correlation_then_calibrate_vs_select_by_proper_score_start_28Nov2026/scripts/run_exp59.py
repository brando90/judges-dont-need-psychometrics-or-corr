"""Experiment 59 analysis (protocol and amendment A1: ../PROTOCOL.md, frozen in commits ae58f5c and 973b0a4 before this script was run on real data).

Run:  python experiments/59_*/scripts/run_exp59.py [--workers 2]     (CPU, a few minutes, no model calls)
Writes ../results/{draws.csv, regret.csv, hypotheses.json, paths_summary.md, regret_figure.png, regret_figure_values.csv}.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next((HERE.parents[1]).glob("57_*/scripts"))))
import frozen as Z  # noqa: E402
from judge_scores import (apply_calibrator, base_rate_forecast, boot_weights, fit_calibrator, item_scores, pct_ci, risk)  # noqa: E402

B = 4000
R_DRAWS = 200
GRID = [25, 50, 100, 200]
MARGIN = 0.005
OUT = HERE.parent / "results"
PATHS = ("A_corr_then_calibrate", "B_proper_score", "C_raw_proper_score")
_D = None


def _data():
    global _D
    if _D is None:
        _D = Z.load()[0]
    return _D


def pearson_or_neg_inf(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return -np.inf
    return float(np.corrcoef(x, y)[0, 1])


def one_draw(name, ni, draw):
    """One labeled pool and test split; returns the per-draw record. All randomness comes from SeedSequence([20261128, 59, set_index, n_index, draw])."""
    d = _data()[name]
    si, K, counts, P, groups = d["index"], d["K"], d["counts"], d["P"], d["groups"]
    rule = Z.PRIMARY[name]
    rng = np.random.default_rng(Z.seed(59, si, ni, draw))
    ug = sorted(set(groups))
    perm = rng.permutation(len(ug))
    pool = {ug[j] for j in perm[: len(ug) // 2]}
    in_pool = np.array([g in pool for g in groups])
    idxP, idxT = np.where(in_pool)[0], np.where(~in_pool)[0]
    n = min(GRID[ni], len(idxP))
    L = rng.choice(idxP, size=n, replace=False)
    y = np.array([rng.choice(Z.valid_ratings(d, i)) for i in L])
    onehot = np.eye(K)[y]
    folds = rng.permutation(n) % 3
    labels = np.arange(K)
    nj = len(Z.JUDGES)
    corr, raw_l, cv, cal_t = np.zeros(nj), np.zeros(nj), np.zeros(nj), np.zeros(nj)
    for j in range(nj):
        qL = P[L, j, :]
        corr[j] = pearson_or_neg_inf(qL @ labels, y.astype(float))
        raw_l[j] = risk(qL, onehot, rule)
        cal_t[j] = risk(apply_calibrator(fit_calibrator(qL, onehot), P[idxT, j, :]), counts[idxT], rule)
        s = np.zeros(n)
        for f in range(3):
            te = folds == f
            s[te] = item_scores(apply_calibrator(fit_calibrator(qL[~te], onehot[~te]), qL[te]), onehot[te], rule)
        cv[j] = s.mean()
    base = base_rate_forecast(counts[idxP])
    base_t = risk(np.tile(base, (len(idxT), 1)), counts[idxT], rule)
    pick = {"A": int(np.argmax(corr)), "B": int(np.argmin(cv)), "C": int(np.argmin(raw_l)), "O": int(np.argmin(cal_t))}
    reg = {k: float((cal_t[pick[k]] - cal_t[pick["O"]]) / base_t) for k in "ABC"}
    return dict(dataset=name, n_grid=GRID[ni], n_used=int(n), draw=draw, n_pool=int(len(idxP)), n_test=int(len(idxT)), pick_A=pick["A"], pick_B=pick["B"], pick_C=pick["C"],
                pick_oracle=pick["O"], regret_A=reg["A"], regret_B=reg["B"], regret_C=reg["C"], oracle_skill=float(1 - cal_t[pick["O"]] / base_t))


def _task(args):
    return one_draw(*args)


def summarize(rows):
    out, diffs = [], {}
    for si, name in enumerate(Z.DATASETS):
        for ni, n in enumerate(GRID):
            r = [x for x in rows if x["dataset"] == name and x["n_grid"] == n]
            r.sort(key=lambda x: x["draw"])
            W = boot_weights(len(r), B, seed=Z.seed(59, 99, si, ni))
            a, b, c = (np.array([x[f"regret_{k}"] for x in r]) for k in "ABC")
            def bm(v):
                return (W @ v) / W.sum(1)
            dAB = a - b
            lo, hi = pct_ci(bm(dAB))
            diffs[(name, n)] = dict(diff=float(dAB.mean()), lo=lo, hi=hi)
            dAC, dBC = a - c, b - c  # SENSITIVITY columns, not pre-registered as decisions
            aclo, achi = pct_ci(bm(dAC))
            bclo, bchi = pct_ci(bm(dBC))
            for tag, v in (("A_corr_then_calibrate", a), ("B_proper_score", b), ("C_raw_proper_score", c)):
                k = tag[0]
                vlo, vhi = pct_ci(bm(v))
                out.append(dict(dataset=name, n_grid=n, n_used=r[0]["n_used"], path=tag, draws=len(r), mean_regret=float(v.mean()), regret_lo=vlo, regret_hi=vhi,
                                share_picks_oracle_judge=float(np.mean([x[f"pick_{k}"] == x["pick_oracle"] for x in r])),
                                mean_oracle_skill=float(np.mean([x["oracle_skill"] for x in r])),
                                mean_A_minus_B=float(dAB.mean()), A_minus_B_lo=lo, A_minus_B_hi=hi,
                                sens_mean_A_minus_C=float(dAC.mean()), sens_A_minus_C_lo=aclo, sens_A_minus_C_hi=achi,
                                sens_mean_B_minus_C=float(dBC.mean()), sens_B_minus_C_lo=bclo, sens_B_minus_C_hi=bchi))
    return out, diffs


def decide(diffs):
    h1_rows = {name: diffs[(name, 25)] for name in Z.DATASETS}
    b_better = [n for n, v in h1_rows.items() if v["diff"] > 0]
    b_sig = [n for n in b_better if h1_rows[n]["lo"] > 0]
    h1 = dict(rule="at n = 25, proper-score selection (B) has lower mean regret than correlation selection (A) in >= 3 of 5 sets and the paired 95% interval excludes 0 in >= 2 of those sets",
              sets_B_lower=b_better, sets_with_interval_excluding_zero=b_sig, decision="supported" if len(b_better) >= 3 and len(b_sig) >= 2 else "falsified")
    h2_rows = {name: abs(diffs[(name, 200)]["diff"]) for name in Z.DATASETS}
    h2 = dict(rule=f"at n = 200 the absolute difference in mean regret between A and B is below {MARGIN} skill points in all 5 sets", abs_difference=h2_rows,
              decision="supported" if all(v < MARGIN for v in h2_rows.values()) else "falsified")
    smallest = next((n for n in GRID if all(abs(diffs[(name, n)]["diff"]) < MARGIN for name in Z.DATASETS)), None)
    h3 = dict(rule=f"smallest n on the grid at which |A minus B| < {MARGIN} in every set", value=smallest if smallest is not None else "none")
    return {"H1": h1, "H2": h2, "H3": h3}


def make_figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    fig, axes = plt.subplots(2, 5, figsize=(18, 7.4))
    cols = {"A_corr_then_calibrate": "#0072B2", "B_proper_score": "#D55E00", "C_raw_proper_score": "0.45"}
    names = {"A_corr_then_calibrate": "A: select by correlation, then calibrate", "B_proper_score": "B: select by cross-validated proper score",
             "C_raw_proper_score": "C: select by raw proper score (sensitivity)"}
    vals = []
    for c, name in enumerate(Z.DATASETS):
        ax, bx = axes[0, c], axes[1, c]
        for p in PATHS:
            rows = [r for r in summary if r["dataset"] == name and r["path"] == p]
            x = [r["n_used"] for r in rows]
            ax.plot(x, [100 * r["mean_regret"] for r in rows], marker="o" if p != "C_raw_proper_score" else "s", color=cols[p], ls="-" if p != "C_raw_proper_score" else "--",
                    label=names[p] if c == 0 else None)
            for r in rows:
                vals.append(dict(dataset=name, n_used=r["n_used"], path=p, mean_regret_points=100 * r["mean_regret"], lo=100 * r["regret_lo"], hi=100 * r["regret_hi"]))
        rows = [r for r in summary if r["dataset"] == name and r["path"] == "A_corr_then_calibrate"]
        x = np.array([r["n_used"] for r in rows])
        d = np.array([100 * r["mean_A_minus_B"] for r in rows])
        lo = d - np.array([100 * r["A_minus_B_lo"] for r in rows])
        hi = np.array([100 * r["A_minus_B_hi"] for r in rows]) - d
        bx.errorbar(x, d, yerr=[lo, hi], fmt="o-", color="k", capsize=3)
        bx.axhline(0, color="0.3", lw=1)
        for m in (-100 * MARGIN, 100 * MARGIN):
            bx.axhline(m, color="0.5", lw=1, ls=":")
        ax.set_title(f"{Z.LABELS[name]} ({Z.PRIMARY[name].upper()})", fontsize=10)
        ax.set_ylabel("Mean regret on held-out test items\n(skill points; lower is better; 0 = oracle judge)", fontsize=8)
        bx.set_ylabel("Regret of A minus regret of B (skill points)\npositive = proper-score selection better;\ndotted = equivalence margin", fontsize=8)
        for a in (ax, bx):
            a.set_xscale("log")
            a.set_xticks([25, 50, 100, 200])
            a.set_xticklabels(["25", "50", "100", "200"])
            a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            a.set_xlabel("Labeled items n (one human rating each; log scale)" + ("; last point is 175" if name == "dices350" else ""), fontsize=8)
            a.grid(True, which="both")
            a.set_axisbelow(True)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Experiment 59: regret of choosing a judge (calibrated test risk of the chosen judge minus that of the best judge, as skill points); 200 random pool/test splits per point; bottom row = paired difference with 95% bootstrap interval", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "regret_figure.png", dpi=110)
    plt.close(fig)
    Z.write_csv(OUT / "regret_figure_values.csv", vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--draws", type=int, default=R_DRAWS)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    tasks = [(name, ni, r) for name in Z.DATASETS for ni in range(len(GRID)) for r in range(args.draws)]
    with Pool(args.workers) as pool:
        rows = pool.map(_task, tasks, chunksize=20)
    summary, diffs = summarize(rows)
    dec = decide(diffs)
    dec.update(draws_per_cell=args.draws, bootstrap_draws=B, margin_skill_points=MARGIN)
    Z.write_csv(OUT / "draws.csv", rows)
    Z.write_csv(OUT / "regret.csv", summary)
    (OUT / "hypotheses.json").write_text(json.dumps(dec, indent=2, default=float))
    make_figure(summary)
    L = ["# Experiment 59 summary (regret in percentage points of skill = 100 x regret, lower is better; A minus B positive = proper-score selection (B) better, negative = correlation selection (A) better)", ""]
    for name in Z.DATASETS:
        L += [f"## {Z.LABELS[name]} ({Z.PRIMARY[name]})", "", "n used | mean regret A | mean regret B | mean regret C | A minus B [95% interval] | share picking the oracle judge A / B / C | sensitivity: A minus C [95% interval]", "---|---|---|---|---|---|---"]
        for n in GRID:
            g = {r["path"][0]: r for r in summary if r["dataset"] == name and r["n_grid"] == n}
            L.append(f"{g['A']['n_used']} | {100*g['A']['mean_regret']:.2f} | {100*g['B']['mean_regret']:.2f} | {100*g['C']['mean_regret']:.2f} | "
                     f"{100*g['A']['mean_A_minus_B']:+.2f} [{100*g['A']['A_minus_B_lo']:+.2f}, {100*g['A']['A_minus_B_hi']:+.2f}] | "
                     f"{g['A']['share_picks_oracle_judge']:.2f} / {g['B']['share_picks_oracle_judge']:.2f} / {g['C']['share_picks_oracle_judge']:.2f} | "
                     f"{100*g['A']['sens_mean_A_minus_C']:+.2f} [{100*g['A']['sens_A_minus_C_lo']:+.2f}, {100*g['A']['sens_A_minus_C_hi']:+.2f}]")
        L.append("")
    L += ["## Decisions", f"- H1: {dec['H1']['decision']} (sets where B has lower regret at n = 25: {dec['H1']['sets_B_lower']}; with the interval excluding 0: {dec['H1']['sets_with_interval_excluding_zero']}).",
          f"- H2: {dec['H2']['decision']} (|A minus B| at n = 200: " + ", ".join(f"{k} {100*v:.2f}" for k, v in dec['H2']['abs_difference'].items()) + " percentage points of skill; margin 0.50).",
          f"- H3: smallest n at which |A minus B| < 0.005 in every set: {dec['H3']['value']}."]
    (OUT / "paths_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
