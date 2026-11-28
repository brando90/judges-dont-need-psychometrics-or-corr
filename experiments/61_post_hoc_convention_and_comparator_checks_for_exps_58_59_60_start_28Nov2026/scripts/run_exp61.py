"""Experiment 61 analysis: POST HOC convention and comparator checks for Experiments 58, 59 and 60 (protocol: ../PROTOCOL.md, committed in 28f6148 before this script existed).

Run:  OMP_NUM_THREADS=1 python3 -I -B experiments/61_*/scripts/run_exp61.py [--workers 2]     (CPU, about two minutes, no model calls)
Writes ../results/{exp60_conventions.csv, exp58_estimand_gap.csv, exp58_hard_vs_majority.csv, exp59_random_baseline.csv, exp59_draw_risks.csv, summary.md}.

Integrity assertions (the script stops with an AssertionError if one fails):
  * C1 reproduces the skill and 95% interval of Experiment 60's frozen collapsed forecast (rounded mean) on every set;
  * C3 reproduces the picks and regrets of paths A, B and C of EVERY draw in Experiment 59's committed draws.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPS = HERE.parents[1]
for pat in ("57_*/scripts", "58_*/scripts", "59_*/scripts", "60_*/scripts"):
    sys.path.insert(0, str(next(EXPS.glob(pat))))
import frozen as Z  # noqa: E402
import run_exp58 as E58  # noqa: E402
import run_exp59 as E59  # noqa: E402
import run_exp60 as E60  # noqa: E402
from judge_scores import (apply_calibrator, base_rate_forecast, boot_mean, boot_weights, boot_weights_grouped, fit_calibrator, floor_probs, item_scores,  # noqa: E402
                          pct_ci, risk)

B = 4000
OUT = HERE.parent / "results"
CONVENTIONS = ("rounded_mean", "modal_label", "median_label", "mean_preserving_two_point")
ORDERED_SETS = ("chaosnli_alphanli", "dices350", "dices990")  # the sets of Experiment 60
_DATA = None


def data():
    global _DATA
    if _DATA is None:
        _DATA = Z.load()[0]
    return _DATA


# ------------------------------------------------------------------------------------------------ C1
def collapsed_forecasts(pA, mu, K):
    """The four ways of turning half-A rating laws pA [N, K] with means mu [N] into forecasts (all floored at 1e-6)."""
    n = pA.shape[0]
    rows = np.arange(n)
    out = {"rounded_mean": E60.collapsed_forecast(mu, K)}  # Experiment 60's frozen convention, imported unchanged
    out["modal_label"] = floor_probs(np.eye(K)[pA.argmax(1)])  # ties go to the lowest label
    out["median_label"] = floor_probs(np.eye(K)[(np.cumsum(pA, 1) >= 0.5 - 1e-12).argmax(1)])  # lowest label with cumulative probability >= 1/2
    lo = np.minimum(np.floor(mu).astype(int), K - 1)
    t = mu - lo
    two = np.zeros((n, K))
    two[rows, lo] = 1.0 - t
    two[rows, np.minimum(lo + 1, K - 1)] += t
    out["mean_preserving_two_point"] = floor_probs(two)
    return out


def conventions_for_set(name):
    d = data()[name]
    si, K, N, rule = d["index"], d["K"], len(d["counts"]), Z.PRIMARY[name]
    W = boot_weights(N, B, seed=Z.seed(60, 99, si))  # Experiment 60's own bootstrap weights, so the frozen row reproduces exactly
    keys = ("truthful", "base") + CONVENTIONS
    V = {k: [] for k in keys}
    mid_round, mid_mode = [], []
    for s in range(E60.SPLITS):
        sp = E60.split_once(d, np.random.default_rng(Z.seed(60, si, s)), keep_obs=False)
        st = E60.stats_from_split(sp, K)
        cB = sp["cB"]
        V["truthful"].append(item_scores(floor_probs(st["pA"]), cB, rule))
        V["base"].append(item_scores(np.tile(base_rate_forecast(sp["cA"]), (N, 1)), cB, rule))
        cf = collapsed_forecasts(st["pA"], st["mu"], K)
        for k in CONVENTIONS:
            V[k].append(item_scores(cf[k], cB, rule))
        if K == 3:
            mid_round.append(float((cf["rounded_mean"][:, 1] > 0.5).mean()))
            mid_mode.append(float((cf["modal_label"][:, 1] > 0.5).mean()))
    M = {k: np.array(v).T for k, v in V.items()}  # [N, S]

    def risk_draws(X):
        return ((W @ X) / W.sum(1, keepdims=True)).mean(1)

    r0 = {k: float(M[k].mean()) for k in keys}
    rb = {k: risk_draws(M[k]) for k in keys}
    sk0 = {k: 1.0 - r0[k] / r0["base"] for k in keys}
    skb = {k: 1.0 - rb[k] / rb["base"] for k in keys}
    rows = []
    for k in keys:
        lo, hi = pct_ci(skb[k])
        row = dict(dataset=name, rule=rule, n_items=N, n_splits=E60.SPLITS, bootstrap_draws=B, convention=k, risk=r0[k], skill=sk0[k], skill_lo=lo, skill_hi=hi,
                   truthful_minus_this_skill=None, diff_lo=None, diff_hi=None, share_items_middle_label=None)  # one fixed column set for the CSV
        if k in CONVENTIONS:
            dlo, dhi = pct_ci(skb["truthful"] - skb[k])
            row.update(truthful_minus_this_skill=sk0["truthful"] - sk0[k], diff_lo=dlo, diff_hi=dhi)
            if K == 3 and k == "rounded_mean":
                row.update(share_items_middle_label=float(np.mean(mid_round)))
            if K == 3 and k == "modal_label":
                row.update(share_items_middle_label=float(np.mean(mid_mode)))
        rows.append(row)
    return rows


def check_against_exp60(rows):
    ref = {r["dataset"]: r for r in csv.DictReader(open(E60.OUT / "sampled_vs_mean.csv"))}
    for name in ORDERED_SETS:
        mine = next(r for r in rows if r["dataset"] == name and r["convention"] == "rounded_mean")
        theirs = ref[name]
        for a, b, tag in ((mine["skill"], theirs["skill_collapsed"], "skill"), (mine["skill_lo"], theirs["skill_collapsed_lo"], "lo"), (mine["skill_hi"], theirs["skill_collapsed_hi"], "hi")):
            assert abs(a - float(b)) < 1e-8, f"C1 does not reproduce Experiment 60 ({name}, {tag}): {a} vs {b}"
        t = next(r for r in rows if r["dataset"] == name and r["convention"] == "truthful")
        assert abs(t["skill"] - float(theirs["skill_truthful"])) < 1e-8, f"C1 truthful skill does not reproduce Experiment 60 ({name})"
    k2 = [r for r in rows if r["dataset"] == "chaosnli_alphanli"]
    tw = next(r for r in k2 if r["convention"] == "truthful")["skill"]
    tp = next(r for r in k2 if r["convention"] == "mean_preserving_two_point")["skill"]
    assert abs(tw - tp) < 1e-12, "with two labels the mean-preserving two-point forecast must equal the truthful forecast"


# ------------------------------------------------------------------------------------------------ C2
def estimand_rows():
    gap_rows, judge_rows = [], []
    for si, name in enumerate(Z.DATASETS):
        d = data()[name]
        counts, P, groups = d["counts"], d["P"], d["groups"]
        h = counts / counts.sum(1, keepdims=True)
        W = boot_weights_grouped(groups, B, seed=Z.seed(61, 1, si))
        top = h >= h.max(1, keepdims=True) - 1e-12
        tie_share = float((top.sum(1) > 1).mean())
        for ev, (labels, ev_name) in E58.EVENTS[name].items():
            soft_item = h[:, labels].sum(1)
            maj_item = E58.hard_label_share(h, labels)
            soft0, maj0 = 100 * soft_item.mean(), 100 * maj_item.mean()
            soft_b, maj_b = 100 * boot_mean(soft_item, W), 100 * boot_mean(maj_item, W)
            glo, ghi = pct_ci(maj_b - soft_b)
            gap_rows.append(dict(dataset=name, event=ev, event_labels=ev_name, n_items=len(h), human_rater_share_rate_pct=soft0, human_majority_rate_pct=maj0,
                                 perfect_judge_hard_label_error_pp=maj0 - soft0, gap_lo=glo, gap_hi=ghi, share_items_tied_top_label=tie_share))
            for jj, j in enumerate(Z.JUDGES):
                hard_item = E58.hard_label_share(P[:, jj, :], labels)
                r0, rb = 100 * hard_item.mean(), 100 * boot_mean(hard_item, W)
                slo, shi = pct_ci(rb - soft_b)
                mlo, mhi = pct_ci(rb - maj_b)
                judge_rows.append(dict(dataset=name, event=ev, judge=j, hard_label_rate_pct=r0, human_rater_share_rate_pct=soft0, human_majority_rate_pct=maj0,
                                       error_vs_rater_share_pp=r0 - soft0, err_share_lo=slo, err_share_hi=shi,
                                       error_vs_majority_pp=r0 - maj0, err_majority_lo=mlo, err_majority_hi=mhi))
    return gap_rows, judge_rows


# ------------------------------------------------------------------------------------------------ C3
def random_pick_regret(cal_t, base_t):
    """Regret of choosing one of the judges uniformly at random (no randomness: the mean over the judges of their regret): mean of (calibrated test risk
    minus the best judge's calibrated test risk) divided by the base-rate test risk, in units of skill."""
    cal_t = np.asarray(cal_t, dtype=float)
    return float(np.mean((cal_t - cal_t.min()) / base_t))


def draw_with_risks(name, ni, draw):
    """Copy of run_exp59.one_draw (same RNG consumption, same operations) that also returns the five judges' calibrated test risks and the random-pick regret."""
    d = E59._data()[name]
    si, K, counts, P, groups = d["index"], d["K"], d["counts"], d["P"], d["groups"]
    rule = Z.PRIMARY[name]
    rng = np.random.default_rng(Z.seed(59, si, ni, draw))
    ug = sorted(set(groups))
    perm = rng.permutation(len(ug))
    pool = {ug[j] for j in perm[: len(ug) // 2]}
    in_pool = np.array([g in pool for g in groups])
    idxP, idxT = np.where(in_pool)[0], np.where(~in_pool)[0]
    n = min(E59.GRID[ni], len(idxP))
    L = rng.choice(idxP, size=n, replace=False)
    y = np.array([rng.choice(Z.valid_ratings(d, i)) for i in L])
    onehot = np.eye(K)[y]
    folds = rng.permutation(n) % 3
    labels = np.arange(K)
    nj = len(Z.JUDGES)
    corr, raw_l, cv, cal_t = np.zeros(nj), np.zeros(nj), np.zeros(nj), np.zeros(nj)
    for j in range(nj):
        qL = P[L, j, :]
        corr[j] = E59.pearson_or_neg_inf(qL @ labels, y.astype(float))
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
    row = dict(dataset=name, n_grid=E59.GRID[ni], n_used=int(n), draw=draw, n_pool=int(len(idxP)), n_test=int(len(idxT)), pick_A=pick["A"], pick_B=pick["B"], pick_C=pick["C"],
               pick_oracle=pick["O"], regret_A=reg["A"], regret_B=reg["B"], regret_C=reg["C"],
               regret_random=random_pick_regret(cal_t, base_t), base_test_risk=float(base_t))
    for j, jn in enumerate(Z.JUDGES):
        row[f"cal_test_risk_{Z.SHORT[jn]}"] = float(cal_t[j])
    return row


def _task(args):
    return draw_with_risks(*args)


def check_against_exp59(rows):
    ref = list(csv.DictReader(open(E59.OUT / "draws.csv")))
    assert len(ref) == len(rows), f"C3: {len(rows)} recomputed draws against {len(ref)} committed"
    for mine, theirs in zip(rows, ref):
        assert mine["dataset"] == theirs["dataset"] and mine["n_grid"] == int(theirs["n_grid"]) and mine["draw"] == int(theirs["draw"]), "C3: draw order differs"
        for k in ("n_used", "n_pool", "n_test", "pick_A", "pick_B", "pick_C", "pick_oracle"):
            assert mine[k] == int(theirs[k]), f"C3 does not reproduce Experiment 59 draws.csv: {k} in {theirs['dataset']} n={theirs['n_grid']} draw {theirs['draw']}"
        for k in ("regret_A", "regret_B", "regret_C"):
            assert np.isclose(mine[k], float(theirs[k]), rtol=1e-7, atol=1e-12), f"C3 does not reproduce Experiment 59 draws.csv: {k} in {theirs['dataset']} n={theirs['n_grid']} draw {theirs['draw']}"


def summarize_c3(rows):
    out = []
    for si, name in enumerate(Z.DATASETS):
        for ni, n in enumerate(E59.GRID):
            r = sorted([x for x in rows if x["dataset"] == name and x["n_grid"] == n], key=lambda x: x["draw"])
            W = boot_weights(len(r), B, seed=Z.seed(61, 2, si, ni))
            v = {k: np.array([x[f"regret_{k}"] for x in r]) for k in ("A", "B", "C", "random")}
            row = dict(dataset=name, n_grid=n, n_used=r[0]["n_used"], draws=len(r), **{f"mean_regret_{k}": float(x.mean()) for k, x in v.items()})
            for tag in "ABC":
                diff = v[tag] - v["random"]
                lo, hi = pct_ci(boot_mean(diff, W))
                row[f"{tag}_minus_random"] = float(diff.mean())
                row[f"{tag}_minus_random_lo"] = lo
                row[f"{tag}_minus_random_hi"] = hi
            out.append(row)
    return out


# ------------------------------------------------------------------------------------------------ report
def pp(x):
    return f"{100 * x:+.2f}"


def build_summary(c1, gap_rows, judge_rows, c3):
    L = ["# Experiment 61 summary (POST HOC, descriptive; protocol `PROTOCOL.md`)", ""]
    L += ["## C1: skill of the truthful and of four collapsed forecasts (Experiment 60's splits; skill = 1 - risk / base-rate risk on half B; higher is better; 0 is the base rate)", ""]
    for name in ORDERED_SETS:
        rows = [r for r in c1 if r["dataset"] == name]
        L += [f"### {Z.LABELS[name]} (rule {Z.PRIMARY[name]}, {rows[0]['n_items']} items, {rows[0]['n_splits']} splits, {rows[0]['bootstrap_draws']} item-clustered draws)", "",
              "forecast | skill [95% interval] | truthful minus this forecast, same score, same half B [95% interval] | share of items whose forecast label is the middle label", "---|---|---|---"]
        for r in rows:
            diff = f"{r['truthful_minus_this_skill']:+.3f} [{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}]" if r["diff_lo"] is not None else ""
            mid = f"{100 * r['share_items_middle_label']:.0f}%" if r["share_items_middle_label"] is not None else ""
            L.append(f"{r['convention']} | {r['skill']:+.3f} [{r['skill_lo']:+.3f}, {r['skill_hi']:+.3f}] | {diff} | {mid}")
        L.append("")
    L += ["## C2: hard-label error of a perfect judge (human rater-share rate versus human majority rate, percentage points; majority minus rater-share)", "",
          "set | event | rater-share rate | majority rate | perfect-judge hard-label error [95% interval] | items with a tied top label", "---|---|---|---|---|---"]
    for r in gap_rows:
        L.append(f"{Z.LABELS[r['dataset']]} | {r['event']} ({r['event_labels']}) | {r['human_rater_share_rate_pct']:.1f}% | {r['human_majority_rate_pct']:.1f}% | "
                 f"{r['perfect_judge_hard_label_error_pp']:+.1f} [{r['gap_lo']:+.1f}, {r['gap_hi']:+.1f}] | {100 * r['share_items_tied_top_label']:.1f}%")
    L.append("")
    prim = [r for r in judge_rows if r["event"] == "primary"]
    closer = sum(abs(r["error_vs_majority_pp"]) < abs(r["error_vs_rater_share_pp"]) for r in prim)
    L += [f"Hard-label judges, primary events, {len(prim)} (set, judge) cells: error against the rater-share rate ranges {min(r['error_vs_rater_share_pp'] for r in prim):+.1f} to {max(r['error_vs_rater_share_pp'] for r in prim):+.1f} points; "
          f"against the human majority rate {min(r['error_vs_majority_pp'] for r in prim):+.1f} to {max(r['error_vs_majority_pp'] for r in prim):+.1f} points; "
          f"the error against the majority rate is smaller in absolute value than against the rater-share rate in {closer} of {len(prim)} cells.", ""]
    for name in Z.DATASETS:
        cells = [r for r in prim if r["dataset"] == name]
        L.append(f"- {Z.LABELS[name]}: error against rater-share {min(r['error_vs_rater_share_pp'] for r in cells):+.1f} to {max(r['error_vs_rater_share_pp'] for r in cells):+.1f}; "
                 f"against majority {min(r['error_vs_majority_pp'] for r in cells):+.1f} to {max(r['error_vs_majority_pp'] for r in cells):+.1f}.")
    L += ["", "## C3: Experiment 59 selection paths against a uniformly random pick (regret in percentage points of skill; lower is better; path minus random: positive means the path is worse than picking at random)", "",
          "set | n | regret A (Pearson) | regret B (3-fold CV calibrated proper risk) | regret C (raw proper risk) | random pick | A minus random [95%] | B minus random [95%] | C minus random [95%]", "---|---|---|---|---|---|---|---|---"]
    for r in c3:
        cells = " | ".join(f"{pp(r[f'{t}_minus_random'])} [{pp(r[f'{t}_minus_random_lo'])}, {pp(r[f'{t}_minus_random_hi'])}]" for t in "ABC")
        L.append(f"{Z.LABELS[r['dataset']]} | {r['n_used']} | {100 * r['mean_regret_A']:.2f} | {100 * r['mean_regret_B']:.2f} | {100 * r['mean_regret_C']:.2f} | {100 * r['mean_regret_random']:.2f} | {cells}")
    L.append("")
    for t, nm in (("A", "A (Pearson correlation)"), ("B", "B (three-fold cross-validated recalibrated proper risk)"), ("C", "C (raw proper risk)")):
        worse = [r for r in c3 if r[f"{t}_minus_random"] > 0]
        worse_sig = [r for r in worse if r[f"{t}_minus_random_lo"] > 0]
        better_sig = [r for r in c3 if r[f"{t}_minus_random_hi"] < 0]
        L.append(f"- Path {nm}: mean regret above the random pick's in {len(worse)} of {len(c3)} cells ({len(worse_sig)} with the whole interval above 0); whole interval below 0 (better than random) in {len(better_sig)}.")
    L += ["", "Integrity: C1 reproduces Experiment 60's rounded-mean and truthful skills and intervals; C3 reproduces the picks and regrets of paths A, B and C of all "
          f"{4 * len(Z.DATASETS) * E59.R_DRAWS} draws in Experiment 59's draws.csv (both asserted by this script)."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    data()
    c1 = [r for name in ORDERED_SETS for r in conventions_for_set(name)]
    check_against_exp60(c1)
    gap_rows, judge_rows = estimand_rows()
    tasks = [(name, ni, r) for name in Z.DATASETS for ni in range(len(E59.GRID)) for r in range(E59.R_DRAWS)]
    with Pool(args.workers) as pool:
        rows = pool.map(_task, tasks, chunksize=20)
    check_against_exp59(rows)
    c3 = summarize_c3(rows)
    Z.write_csv(OUT / "exp60_conventions.csv", c1)
    Z.write_csv(OUT / "exp58_estimand_gap.csv", gap_rows)
    Z.write_csv(OUT / "exp58_hard_vs_majority.csv", judge_rows)
    Z.write_csv(OUT / "exp59_random_baseline.csv", c3)
    Z.write_csv(OUT / "exp59_draw_risks.csv", rows)
    text = build_summary(c1, gap_rows, judge_rows, c3)
    (OUT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
