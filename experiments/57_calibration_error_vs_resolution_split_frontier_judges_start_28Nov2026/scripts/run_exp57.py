"""Experiment 57 analysis (protocol: ../PROTOCOL.md, frozen in commit ae58f5c before this script was run).

Run:  python experiments/57_*/scripts/run_exp57.py          (CPU, about one minute, no model calls)
Writes ../results/{cells.csv, correlations.csv, hypotheses.json, split_summary.md, split_figure.png, split_figure_values.csv}.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frozen as Z  # noqa: E402  (also puts src/judge_scores on sys.path)
from judge_scores import boot_mean, boot_weights_grouped, calibration_split, make_group_folds, pct_ci  # noqa: E402

B = 4000
OUT = HERE.parent / "results"


# ------------------------------------------------------------------ statistics shared with the tests
def tau_b_rows(X, Y):
    """Kendall's tau-b between the rows of X [R, J] and Y [R, J] (judges are the J objects ranked); nan when a row is all ties."""
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    num = np.zeros(len(X))
    sx = np.zeros(len(X))
    sy = np.zeros(len(X))
    for a, b in itertools.combinations(range(X.shape[1]), 2):
        dx = np.sign(X[:, a] - X[:, b])
        dy = np.sign(Y[:, a] - Y[:, b])
        num += dx * dy
        sx += dx * dx
        sy += dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / np.sqrt(sx * sy)


def weighted_pearson(X, y, W):
    """Pearson correlation of every column of X [N, J] with y [N] under item multiplicities W [R, N] -> [R, J]."""
    sw = W.sum(1)[:, None]
    mx = (W @ X) / sw
    my = (W @ y)[:, None] / sw
    cxy = (W @ (X * y[:, None])) / sw - mx * my
    vx = (W @ (X * X)) / sw - mx ** 2
    vy = (W @ (y * y))[:, None] / sw - my ** 2
    return cxy / np.sqrt(vx * vy)


def spearman_draws(X, y, W):
    """Spearman correlations [R, J] on the resamples defined by the integer multiplicities W [R, N]."""
    out = np.zeros((len(W), X.shape[1]))
    for r in range(len(W)):
        idx = np.repeat(np.arange(X.shape[0]), W[r].astype(int))
        ry = rankdata(y[idx])
        ry = ry - ry.mean()
        for j in range(X.shape[1]):
            rx = rankdata(X[idx, j])
            rx = rx - rx.mean()
            out[r, j] = (rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry))
    return out


def holm_adjust(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def forecast_mean(P, K):
    return P @ np.arange(K)


# ------------------------------------------------------------------ main analysis
def analyse():
    D, sha = Z.load()
    cells, corr_rows = [], []
    draws = {}  # (set, judge) -> dict of primary-rule draws used by the hypotheses
    per_set = {}
    for name in Z.DATASETS:
        d = D[name]
        si, K, counts, P = d["index"], d["K"], d["counts"], d["P"]
        folds = make_group_folds(d["groups"], 5, seed=Z.seed(57, 0, si))
        W = boot_weights_grouped(d["groups"], B, seed=Z.seed(57, 1, si))
        human = (counts * np.arange(K)).sum(1) / counts.sum(1)
        Xm = np.column_stack([forecast_mean(P[:, j, :], K) for j in range(len(Z.JUDGES))])
        pear = weighted_pearson(Xm, human, W)
        pear0 = weighted_pearson(Xm, human, np.ones((1, len(human))))[0]
        spr = spearman_draws(Xm, human, W)
        spr0 = np.array([spearmanr(Xm[:, j], human).statistic for j in range(Xm.shape[1])])
        per_set[name] = dict(pear=pear, pear0=pear0, spr=spr, spr0=spr0, skill_raw={}, skill_cal={}, cal_share={}, res_share={})
        for j, judge in enumerate(Z.JUDGES):
            lo, hi = pct_ci(pear[:, j])
            slo, shi = pct_ci(spr[:, j])
            corr_rows.append(dict(dataset=name, judge=judge, n_items=len(human), pearson=pear0[j], pearson_lo=lo, pearson_hi=hi,
                                  spearman=spr0[j], spearman_lo=slo, spearman_hi=shi))
            for rule in Z.RULES_FOR[name]:
                s = calibration_split(P[:, j, :], counts, rule, folds)
                unc_b = boot_mean(s["items_base"], W)
                raw_b = boot_mean(s["items_raw"], W)
                cal_b = boot_mean(s["items_cal"], W)
                stat = {"skill_raw": 1 - raw_b / unc_b, "skill_cal": 1 - cal_b / unc_b,
                        "calibration_share": (raw_b - cal_b) / unc_b, "resolution_share": (unc_b - cal_b) / unc_b}
                row = dict(dataset=name, judge=judge, rule=rule, primary=rule == Z.PRIMARY[name], K=K, n_items=len(human),
                           uncertainty_risk=s["unc"], raw_risk=s["risk_raw"], calibrated_risk=s["risk_cal"],
                           calibration_error=s["calibration_error"], resolution=s["resolution"],
                           identity_residual=s["identity_residual"])
                for key in ("skill_raw", "skill_cal", "calibration_share", "resolution_share"):
                    lo, hi = pct_ci(stat[key])
                    row[key], row[key + "_lo"], row[key + "_hi"] = s[key], lo, hi
                cells.append(row)
                if rule == Z.PRIMARY[name]:
                    draws[(name, judge)] = stat
                    per_set[name]["skill_raw"][judge] = (s["skill_raw"], stat["skill_raw"])
                    per_set[name]["skill_cal"][judge] = (s["skill_cal"], stat["skill_cal"])
                    per_set[name]["cal_share"][judge] = s["calibration_share"]
                    per_set[name]["res_share"][judge] = s["resolution_share"]
    return D, sha, cells, corr_rows, per_set, draws


def decide(per_set):
    out = {"H1": {"sets": {}}, "H2": {"sets": {}}, "H3": {}}
    n_h1 = n_h2 = 0
    for name in Z.DATASETS:
        ps = per_set[name]
        raw0 = np.array([ps["skill_raw"][j][0] for j in Z.JUDGES])
        cal0 = np.array([ps["skill_cal"][j][0] for j in Z.JUDGES])
        rawB = np.column_stack([ps["skill_raw"][j][1] for j in Z.JUDGES])
        calB = np.column_stack([ps["skill_cal"][j][1] for j in Z.JUDGES])
        rec = {}
        for tag, pt, dr in (("pearson", ps["pear0"], ps["pear"]), ("spearman", ps["spr0"], ps["spr"])):
            t_res = float(tau_b_rows(pt[None], cal0[None])[0])
            t_raw = float(tau_b_rows(pt[None], raw0[None])[0])
            d_res = tau_b_rows(dr, calB)
            d_raw = tau_b_rows(dr, rawB)
            diff = d_res - d_raw
            lo, hi = pct_ci(diff)
            rec[tag] = dict(tau_resolution=t_res, tau_raw=t_raw, difference=t_res - t_raw, difference_ci=[lo, hi],
                            n_draws_defined=int(np.isfinite(diff).sum()))
        rec["h1_set_supports"] = bool(rec["pearson"]["tau_resolution"] > rec["pearson"]["tau_raw"])
        n_h1 += rec["h1_set_supports"]
        out["H1"]["sets"][name] = rec
        sd_cal = float(np.std([ps["cal_share"][j] for j in Z.JUDGES], ddof=1))
        sd_res = float(np.std([ps["res_share"][j] for j in Z.JUDGES], ddof=1))
        out["H2"]["sets"][name] = dict(sd_calibration_share=sd_cal, sd_resolution_share=sd_res, h2_set_supports=bool(sd_cal > sd_res))
        n_h2 += sd_cal > sd_res
    out["H1"].update(rule="supported if tau_resolution > tau_raw (Pearson, point estimates) in >= 4 of 5 sets",
                     n_sets_supporting=int(n_h1), decision="supported" if n_h1 >= 4 else "falsified")
    out["H2"].update(rule="supported if the between-judge SD (ddof=1) of the calibration share exceeds that of the resolution share in >= 4 of 5 sets",
                     n_sets_supporting=int(n_h2), decision="supported" if n_h2 >= 4 else "falsified")
    ps = per_set["dices350"]
    neg = [j for j in Z.JUDGES if ps["skill_raw"][j][0] < 0]
    if not neg:
        out["H3"] = dict(decision="not applicable: no judge has negative raw skill on DICES-350", judges=[])
    else:
        p = np.array([(1 + np.sum(ps["skill_cal"][j][1] <= 0)) / (B + 1) for j in neg])
        padj = holm_adjust(p)
        rows = [dict(judge=j, raw_skill=ps["skill_raw"][j][0], calibrated_skill=ps["skill_cal"][j][0], p_one_sided=float(pi),
                     p_holm=float(pa), calibrated_skill_above_zero=bool(ps["skill_cal"][j][0] > 0 and pa < 0.05))
                for j, pi, pa in zip(neg, p, padj)]
        ok = all(r["calibrated_skill_above_zero"] for r in rows)
        out["H3"] = dict(rule="supported if every judge with raw skill < 0 on DICES-350 has calibrated skill > 0 with Holm-adjusted one-sided p < 0.05",
                         judges=rows, decision="supported" if ok else "falsified")
    return out


# ------------------------------------------------------------------ outputs
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def make_figure(per_set, corr_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"gpt-6-astra": "#0072B2", "claude-opus-5-5": "#D55E00", "kimi-k3": "#009E73", "glm-5.3": "#CC79A7", "deepseek-v4.1-flash": "#E69F00"}
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.2))
    vals = []
    for c, name in enumerate(Z.DATASETS):
        ps = per_set[name]
        ax, bx = axes[0, c], axes[1, c]
        for i, j in enumerate(Z.JUDGES):
            r0, c0 = ps["skill_raw"][j][0], ps["skill_cal"][j][0]
            ax.plot([r0, c0], [i, i], color=colors[j], lw=2, zorder=2)
            ax.scatter([r0], [i], s=70, facecolor="white", edgecolor=colors[j], lw=2, zorder=3)
            ax.scatter([c0], [i], s=70, color=colors[j], zorder=3)
            bx.scatter([ps["pear0"][list(Z.JUDGES).index(j)]], [i], s=70, color=colors[j], zorder=3)
            vals.append(dict(dataset=name, judge=j, raw_skill=r0, calibrated_skill=c0, pearson=ps["pear0"][list(Z.JUDGES).index(j)]))
        ax.axvline(0, color="0.4", lw=1, ls="--")
        for a in (ax, bx):
            a.set_yticks(range(len(Z.JUDGES)))
            a.set_yticklabels([Z.SHORT[j] for j in Z.JUDGES] if c == 0 else [""] * 5)
            a.invert_yaxis()
            a.grid(True)
            a.set_axisbelow(True)
        ax.set_title(f"{Z.LABELS[name]} (n={len(ps['pear0']) and D_N[name]}; {Z.PRIMARY[name].upper()})", fontsize=10)
        ax.set_xlabel("Skill vs human base rate, cross-fitted held-out\n(higher is better; 0 = no better than base rate)", fontsize=8)
        bx.set_xlabel("Pearson r with human item mean\n(higher is better; all items)", fontsize=8)
    from matplotlib.lines import Line2D
    axes[0, 0].legend(handles=[Line2D([], [], marker="o", mfc="white", mec="k", ls="", label="raw (as issued)"),
                               Line2D([], [], marker="o", color="k", ls="", label="after per-judge calibration")], fontsize=8, loc="lower left")
    fig.suptitle("Experiment 57: segment length = calibration error (in skill points); filled dot = resolution; bottom row = Pearson r "
                 "(frozen frontier judges, matched rule, 5-fold cross-fit)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "split_figure.png", dpi=130)
    plt.close(fig)
    write_csv(OUT / "split_figure_values.csv", vals)


D_N = {}


def main():
    OUT.mkdir(exist_ok=True)
    D, sha, cells, corr_rows, per_set, _ = analyse()
    for n in Z.DATASETS:
        D_N[n] = len(D[n]["counts"])
    dec = decide(per_set)
    write_csv(OUT / "cells.csv", cells)
    write_csv(OUT / "correlations.csv", corr_rows)
    dec["bundle_sha256"] = sha
    dec["draws"] = B
    dec["max_abs_identity_residual"] = max(abs(r["identity_residual"]) for r in cells)
    (OUT / "hypotheses.json").write_text(json.dumps(dec, indent=2, default=float))
    make_figure(per_set, corr_rows)
    lines = ["# Experiment 57 summary (primary rule per set; skill = 1 - risk/base-rate risk, higher is better)", "",
             f"Bundle sha256 {sha[:16]}...; B = {B} group-clustered draws; maximum |identity residual| = {dec['max_abs_identity_residual']:.2e}.", ""]
    for name in Z.DATASETS:
        lines += [f"## {Z.LABELS[name]} ({Z.PRIMARY[name]}, n = {D_N[name]})", "",
                  "judge | raw skill | calibrated skill (resolution) | calibration error (skill pts) | Pearson r",
                  "---|---|---|---|---"]
        for j in Z.JUDGES:
            ps = per_set[name]
            lines.append(f"{Z.SHORT[j]} | {ps['skill_raw'][j][0]:+.3f} | {ps['skill_cal'][j][0]:+.3f} | {ps['cal_share'][j]:+.3f} | {ps['pear0'][list(Z.JUDGES).index(j)]:+.3f}")
        h1 = dec["H1"]["sets"][name]
        lines += ["", f"Kendall tau-b of the judge ranking by Pearson r against resolution: {h1['pearson']['tau_resolution']:+.2f}; against raw skill: "
                  f"{h1['pearson']['tau_raw']:+.2f}; difference interval [{h1['pearson']['difference_ci'][0]:+.2f}, {h1['pearson']['difference_ci'][1]:+.2f}]. "
                  f"Spearman version: {h1['spearman']['tau_resolution']:+.2f} against {h1['spearman']['tau_raw']:+.2f}.", ""]
    lines += ["## Decisions", f"- H1: {dec['H1']['decision']} ({dec['H1']['n_sets_supporting']} of 5 sets).",
              f"- H2: {dec['H2']['decision']} ({dec['H2']['n_sets_supporting']} of 5 sets).", f"- H3: {dec['H3']['decision']}."]
    (OUT / "split_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
