"""Experiment 58 analysis (protocol: ../PROTOCOL.md, frozen in commit ae58f5c before this script was run).

Run:  python experiments/58_*/scripts/run_exp58.py          (CPU, seconds, no model calls)
Writes ../results/{human_rates.csv, rates.csv, flips.csv, hypotheses.json, decisions_summary.md, rates_figure.png, rates_figure_values.csv}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next((HERE.parents[1]).glob("57_*/scripts"))))
import frozen as Z  # noqa: E402
from judge_scores import boot_mean, boot_weights_grouped, crossfit_base_rate, crossfit_calibrate, make_group_folds, pct_ci  # noqa: E402

B = 4000
OUT = HERE.parent / "results"
# events fixed in the protocol before any computation (label codes: NLI 0 contradiction, 1 neutral, 2 entailment; alphaNLI 0, 1; DICES 0 No, 1 Unsure, 2 Yes)
EVENTS = {
    "chaosnli_snli": {"primary": ([2], "entailment")},
    "chaosnli_mnli": {"primary": ([2], "entailment")},
    "chaosnli_alphanli": {"primary": ([0], "label 0")},
    "dices350": {"primary": ([2], "Yes (unsafe)"), "secondary": ([1, 2], "Unsure or Yes")},
    "dices990": {"primary": ([2], "Yes (unsafe)"), "secondary": ([1, 2], "Unsure or Yes")},
}
ESTIMATORS = ("raw_soft", "hard_label", "calibrated_soft")


def hard_label_share(q, labels):
    """Per-item share of the argmax set that lies in the event; ties in the argmax share the item equally."""
    top = q >= q.max(1, keepdims=True) - 1e-12
    return top[:, labels].sum(1) / top.sum(1)


def gate_values(human_pct):
    c = 5.0 * np.round(human_pct / 5.0)
    return [c - 10, c - 5, c, c + 5, c + 10]


def analyse():
    D, sha = Z.load()
    human_rows, rate_rows, flip_rows, posthoc_rows = [], [], [], []
    for name in Z.DATASETS:
        d = D[name]
        si, K, counts, P = d["index"], d["K"], d["counts"], d["P"]
        folds = make_group_folds(d["groups"], 5, seed=Z.seed(58, 0, si))
        W = boot_weights_grouped(d["groups"], B, seed=Z.seed(58, 1, si))
        h = counts / counts.sum(1, keepdims=True)
        cal = {j: crossfit_calibrate(P[:, jj, :], counts, folds) for jj, j in enumerate(Z.JUDGES)}
        for ev, (labels, ev_name) in EVENTS[name].items():
            hum_item = h[:, labels].sum(1)
            hum0 = 100 * hum_item.mean()
            humB = 100 * boot_mean(hum_item, W)
            lo, hi = pct_ci(humB)
            human_rows.append(dict(dataset=name, event=ev, event_labels=ev_name, n_items=len(h), human_rate_pct=hum0, lo=lo, hi=hi))
            base_item = crossfit_base_rate(counts, folds)[:, labels].sum(1)  # POST HOC comparator, not pre-registered: no judge at all
            b0, bB = 100 * base_item.mean(), 100 * boot_mean(base_item, W)
            elo, ehi = pct_ci(bB - humB)
            posthoc_rows.append(dict(dataset=name, event=ev, human_rate_pct=hum0, cross_fitted_base_rate_pct=b0, error_pp=b0 - hum0, error_lo=elo, error_hi=ehi))
            for jj, j in enumerate(Z.JUDGES):
                q = P[:, jj, :]
                per_item = {"raw_soft": q[:, labels].sum(1), "hard_label": hard_label_share(q, labels),
                            "calibrated_soft": cal[j][:, labels].sum(1)}
                for est in ESTIMATORS:
                    r0 = 100 * per_item[est].mean()
                    rB = 100 * boot_mean(per_item[est], W)
                    err0, errB = r0 - hum0, rB - humB
                    rlo, rhi = pct_ci(rB)
                    elo, ehi = pct_ci(errB)
                    alo, ahi = pct_ci(np.abs(errB))
                    rate_rows.append(dict(dataset=name, event=ev, judge=j, estimator=est, human_rate_pct=hum0, rate_pct=r0, rate_lo=rlo, rate_hi=rhi,
                                          error_pp=err0, error_lo=elo, error_hi=ehi, abs_error_pp=abs(err0), abs_error_lo=alo, abs_error_hi=ahi))
                    for tau in gate_values(hum0):
                        est_pass, hum_pass = r0 <= tau, hum0 <= tau
                        diffB = (rB <= tau) != (humB <= tau)
                        flip_rows.append(dict(dataset=name, event=ev, judge=j, estimator=est, tau_pct=tau, estimator_passes=bool(est_pass),
                                              humans_pass=bool(hum_pass), flip=bool(est_pass != hum_pass), false_pass=bool(est_pass and not hum_pass),
                                              false_fail=bool(hum_pass and not est_pass), share_of_draws_flipped=float(diffB.mean()),
                                              significant_flip=bool(diffB.mean() >= 0.975)))
    return D, sha, human_rows, rate_rows, flip_rows, posthoc_rows


def decide(rate_rows, flip_rows):
    prim = [r for r in rate_rows if r["event"] == "primary"]
    by = {(r["dataset"], r["judge"], r["estimator"]): r for r in prim}
    cells = sorted({(r["dataset"], r["judge"]) for r in prim})
    h1 = sum(by[(d, j, "raw_soft")]["abs_error_lo"] > 5.0 for d, j in cells)
    h2 = sum(by[(d, j, "calibrated_soft")]["abs_error_pp"] < by[(d, j, "raw_soft")]["abs_error_pp"] for d, j in cells)
    h3 = sum(by[(d, j, "hard_label")]["abs_error_pp"] > by[(d, j, "calibrated_soft")]["abs_error_pp"] for d, j in cells)
    flips = {}
    for ev in ("primary", "secondary"):
        for est in ESTIMATORS:
            rows = [f for f in flip_rows if f["event"] == ev and f["estimator"] == est]
            if rows:
                flips[f"{ev}:{est}"] = dict(gates=len(rows), flips=sum(f["flip"] for f in rows), significant_flips=sum(f["significant_flip"] for f in rows),
                                            false_passes=sum(f["false_pass"] for f in rows), false_fails=sum(f["false_fail"] for f in rows))
    dec = {"cells": len(cells),
           "H1": dict(rule="raw soft rate: 95% interval of |error| entirely above 5 points in >= 10 of 25 primary cells", n=int(h1), decision="supported" if h1 >= 10 else "falsified"),
           "H2": dict(rule="calibrated soft |error| < raw soft |error| in >= 18 of 25 primary cells (point estimates)", n=int(h2), decision="supported" if h2 >= 18 else "falsified"),
           "H3": dict(rule="hard-label |error| > calibrated soft |error| in >= 18 of 25 primary cells (point estimates)", n=int(h3), decision="supported" if h3 >= 18 else "falsified"),
           "gate_flips": flips}
    # named cells: largest raw error, and the cell most unfavorable to calibration
    raw = [by[(d, j, "raw_soft")] for d, j in cells]
    worst = max(raw, key=lambda r: r["abs_error_pp"])
    delta = {(d, j): by[(d, j, "calibrated_soft")]["abs_error_pp"] - by[(d, j, "raw_soft")]["abs_error_pp"] for d, j in cells}
    k = max(delta, key=delta.get)
    dec["largest_raw_error"] = dict(dataset=worst["dataset"], judge=worst["judge"], error_pp=float(worst["error_pp"]),
                                    ci=[float(worst["error_lo"]), float(worst["error_hi"])])
    dec["calibration_helped_least"] = dict(dataset=k[0], judge=k[1], change_in_abs_error_pp=float(delta[k]),
                                           raw_abs_error=float(by[(k[0], k[1], "raw_soft")]["abs_error_pp"]),
                                           calibrated_abs_error=float(by[(k[0], k[1], "calibrated_soft")]["abs_error_pp"]),
                                           cells_where_calibration_increased_abs_error=int(sum(v > 0 for v in delta.values())))
    return dec


def make_figure(human_rows, rate_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.6), sharey=False)
    style = {"raw_soft": dict(marker="o", mfc="white", mec="#0072B2", color="#0072B2", label="raw soft (mean forecast mass)"),
             "hard_label": dict(marker="^", mfc="#D55E00", mec="#D55E00", color="#D55E00", label="hard label (share of verdicts)"),
             "calibrated_soft": dict(marker="o", mfc="#009E73", mec="#009E73", color="#009E73", label="calibrated soft (cross-fitted)")}
    vals = []
    for ax, name in zip(axes, Z.DATASETS):
        hr = next(r for r in human_rows if r["dataset"] == name and r["event"] == "primary")
        ax.axhspan(hr["lo"], hr["hi"], color="0.85", zorder=0)
        ax.axhline(hr["human_rate_pct"], color="0.2", lw=1.5, ls="--", zorder=1)
        for e, est in enumerate(ESTIMATORS):
            xs = np.arange(len(Z.JUDGES)) + (e - 1) * 0.22
            rows = [next(r for r in rate_rows if r["dataset"] == name and r["event"] == "primary" and r["judge"] == j and r["estimator"] == est) for j in Z.JUDGES]
            y = np.array([r["rate_pct"] for r in rows])
            lo = y - np.array([r["rate_lo"] for r in rows])
            hi = np.array([r["rate_hi"] for r in rows]) - y
            ax.errorbar(xs, y, yerr=[lo, hi], fmt=style[est]["marker"], mfc=style[est]["mfc"], mec=style[est]["mec"], color=style[est]["color"], ms=7, lw=1.2, capsize=2,
                        label=style[est]["label"] if name == Z.DATASETS[0] else None, zorder=3)
            for j, r in zip(Z.JUDGES, rows):
                vals.append(dict(dataset=name, judge=j, estimator=est, rate_pct=r["rate_pct"], rate_lo=r["rate_lo"], rate_hi=r["rate_hi"], human_rate_pct=hr["human_rate_pct"]))
        ax.set_xticks(range(len(Z.JUDGES)))
        ax.set_xticklabels([Z.SHORT[j] for j in Z.JUDGES], rotation=30, fontsize=8)
        ax.set_title(f"{Z.LABELS[name]}: event = {EVENTS[name]['primary'][1]}", fontsize=9)
        ax.set_ylabel("Event rate, % of ratings or outputs\n(dashed = human rate, band = its 95% interval)", fontsize=8)
        ax.set_xlabel("Judge (all items; calibration cross-fitted)", fontsize=8)
        ax.grid(True)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Experiment 58: event rate implied by each judge against the human rate (95% group-clustered bootstrap intervals; the estimator closest to the dashed line is best)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "rates_figure.png", dpi=120)
    plt.close(fig)
    Z.write_csv(OUT / "rates_figure_values.csv", vals)


def main():
    OUT.mkdir(exist_ok=True)
    D, sha, human_rows, rate_rows, flip_rows, posthoc_rows = analyse()
    dec = decide(rate_rows, flip_rows)
    dec.update(bundle_sha256=sha, draws=B)
    Z.write_csv(OUT / "human_rates.csv", human_rows)
    Z.write_csv(OUT / "rates.csv", rate_rows)
    Z.write_csv(OUT / "flips.csv", flip_rows)
    Z.write_csv(OUT / "posthoc_base_rate_rates.csv", posthoc_rows)
    (OUT / "hypotheses.json").write_text(json.dumps(dec, indent=2, default=float))
    make_figure(human_rows, rate_rows)
    L = ["# Experiment 58 summary (rates in percent; error = estimator minus human rate, in percentage points)", "",
         f"Bundle sha256 {sha[:16]}...; B = {B} group-clustered draws.", ""]
    for name in Z.DATASETS:
        hr = next(r for r in human_rows if r["dataset"] == name and r["event"] == "primary")
        L += [f"## {Z.LABELS[name]}: event = {EVENTS[name]['primary'][1]}; human rate {hr['human_rate_pct']:.1f}% (95% interval {hr['lo']:.1f} to {hr['hi']:.1f})", "",
              "judge | raw soft error | hard-label error | calibrated soft error", "---|---|---|---"]
        for j in Z.JUDGES:
            g = {e: next(r for r in rate_rows if r["dataset"] == name and r["event"] == "primary" and r["judge"] == j and r["estimator"] == e) for e in ESTIMATORS}
            L.append(f"{Z.SHORT[j]} | " + " | ".join(f"{g[e]['error_pp']:+.1f} [{g[e]['error_lo']:+.1f}, {g[e]['error_hi']:+.1f}]" for e in ESTIMATORS))
        L.append("")
    L += ["## Decisions", f"- H1: {dec['H1']['decision']} ({dec['H1']['n']} of 25 cells with the raw |error| interval above 5 points).",
          f"- H2: {dec['H2']['decision']} ({dec['H2']['n']} of 25 cells where calibration reduced |error|).",
          f"- H3: {dec['H3']['decision']} ({dec['H3']['n']} of 25 cells where the hard-label |error| exceeded the calibrated one).", "",
          "## Gate flips (gates = 5 thresholds per cell; flip = estimator and humans on opposite sides of 'rate <= tau'; significant = in >= 97.5% of draws)"]
    for k, v in dec["gate_flips"].items():
        L.append(f"- {k}: {v['flips']} of {v['gates']} gates flip ({v['significant_flips']} significant; {v['false_passes']} false passes, {v['false_fails']} false fails)")
    w, u = dec["largest_raw_error"], dec["calibration_helped_least"]
    L += ["", f"Largest raw soft error: {Z.LABELS[w['dataset']]}, {Z.SHORT[w['judge']]}: {w['error_pp']:+.1f} points (95% interval {w['ci'][0]:+.1f} to {w['ci'][1]:+.1f}).",
          f"Cell where calibration helped least (the one most unfavorable to it): {Z.LABELS[u['dataset']]}, {Z.SHORT[u['judge']]}: |error| {u['raw_abs_error']:.2f} to {u['calibrated_abs_error']:.2f} points; "
          f"cells where calibration made |error| larger: {u['cells_where_calibration_increased_abs_error']} of 25.", "",
          "Post hoc (not pre-registered): the cross-fitted human base rate, which uses no judge at all, has an error of "
          f"{min(r['error_pp'] for r in posthoc_rows if r['event'] == 'primary'):+.2f} to {max(r['error_pp'] for r in posthoc_rows if r['event'] == 'primary'):+.2f} points on the primary events, "
          "so a level near the human rate is not evidence that a judge discriminates (see results/posthoc_base_rate_rates.csv)."]
    (OUT / "decisions_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
