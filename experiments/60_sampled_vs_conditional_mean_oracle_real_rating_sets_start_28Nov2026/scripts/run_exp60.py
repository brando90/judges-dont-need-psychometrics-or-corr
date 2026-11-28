"""Experiment 60 analysis: ORACLE CHECK of the sampled-score theorem on real rating distributions (protocol: ../PROTOCOL.md, frozen in commit ae58f5c before this script was run).

No language model is involved. Run:  python experiments/60_*/scripts/run_exp60.py   (CPU, about a minute)
Writes ../results/{sampled_vs_mean.csv, hypotheses.json, summary.md, figure1_sampled_score.png/.pdf, figure1_values.csv}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next((HERE.parents[1]).glob("57_*/scripts"))))
import frozen as Z  # noqa: E402
from judge_scores import base_rate_forecast, boot_weights, floor_probs, item_scores, pct_ci  # noqa: E402

B = 4000
SPLITS = 200
MC_SPLITS = 20
SETS = ["chaosnli_alphanli", "dices350", "dices990"]  # SNLI and MNLI-m are excluded: unordered labels, Pearson with an arbitrary coding is not meaningful
OUT = HERE.parent / "results"


# ------------------------------------------------------------------ closed forms from sufficient statistics
def corr_stats(W, nB, S1, S2, mu, m2):
    """Correlations of the conditional-mean judge (outputs mu_A(i)) and of the sampled judge (outputs S ~ p_A(i), independent of Y given the item)
    with the half-B ratings, from sufficient statistics per item, under item multiplicities W [R, N]. Returns (r_mean, r_sampled), each [R].
      nB item's number of half-B ratings; S1, S2 sum and sum of squares of those ratings; mu, m2 first and second moment of the half-A distribution."""
    n = W @ nB
    ey, ey2 = (W @ S1) / n, (W @ S2) / n
    et, et2 = (W @ (nB * mu)) / n, (W @ (nB * mu * mu)) / n
    ety, es2 = (W @ (mu * S1)) / n, (W @ (nB * m2)) / n
    cov, vy = ety - et * ey, ey2 - ey ** 2
    return cov / np.sqrt((et2 - et ** 2) * vy), cov / np.sqrt((es2 - et ** 2) * vy)


def theory(W, n, m, q, nA):
    """Finite-rater-noise-corrected rho and the mean-judge prediction. n ratings per item, m their mean, q the mean of their squares, nA the half-A size."""
    s2 = (q - m * m) * n / (n - 1.0)  # unbiased within-item variance
    sw = W.sum(1)
    mb = (W @ m) / sw
    var_means = (W @ (m * m)) / sw - mb ** 2
    var_mu = var_means - (W @ (s2 / n)) / sw  # unbiased variance of the true conditional means
    ey, ey2 = (W @ (n * m)) / (W @ n), (W @ (n * q)) / (W @ n)
    vy = ey2 - ey ** 2
    rho = var_mu / vy
    r_mean_pred = var_mu / np.sqrt((var_mu + (W @ (s2 / nA)) / sw) * vy)
    return rho, r_mean_pred


def split_once(d, rng, keep_obs=False):
    """Random split of every item's ratings into half A (floor(n/2), builds the judges) and half B (the rest, held out); no rating is in both."""
    N, K = len(d["counts"]), d["K"]
    cA, cB = np.zeros((N, K)), np.zeros((N, K))
    obs_item, obs_y = [], []
    for i in range(N):
        r = Z.valid_ratings(d, i)
        r = r[rng.permutation(len(r))]
        na = len(r) // 2
        cA[i] = np.bincount(r[:na], minlength=K)
        cB[i] = np.bincount(r[na:], minlength=K)
        if keep_obs:
            obs_item.append(np.full(len(r) - na, i))
            obs_y.append(r[na:])
    out = dict(cA=cA, cB=cB)
    if keep_obs:
        out.update(obs_item=np.concatenate(obs_item), obs_y=np.concatenate(obs_y))
    return out


def stats_from_split(sp, K):
    lab = np.arange(K, dtype=float)
    cA, cB = sp["cA"], sp["cB"]
    pA = cA / cA.sum(1, keepdims=True)
    return dict(nB=cB.sum(1), S1=cB @ lab, S2=cB @ lab ** 2, mu=pA @ lab, m2=pA @ lab ** 2, pA=pA)


def collapsed_forecast(mu, K):
    nearest = np.clip(np.floor(mu + 0.5), 0, K - 1).astype(int)  # nearest grid label, halves round up
    return floor_probs(np.eye(K)[nearest])


def analyse_set(name):
    D, _ = Z.load()
    d = D[name]
    si, K, N = d["index"], d["K"], len(d["counts"])
    rule = Z.PRIMARY[name]
    W1 = np.ones((1, N))
    W = boot_weights(N, B, seed=Z.seed(60, 99, si))  # item-clustered bootstrap
    cols = {k: [] for k in ("nB", "S1", "S2", "mu", "m2")}
    V = {k: [] for k in ("truthful", "collapsed", "base")}
    mc_diffs = []
    for s in range(SPLITS):
        rng = np.random.default_rng(Z.seed(60, si, s))
        sp = split_once(d, rng, keep_obs=s < MC_SPLITS)
        st = stats_from_split(sp, K)
        for k in cols:
            cols[k].append(st[k])
        cB = sp["cB"]
        V["truthful"].append(item_scores(floor_probs(st["pA"]), cB, rule))
        V["collapsed"].append(item_scores(collapsed_forecast(st["mu"], K), cB, rule))
        V["base"].append(item_scores(np.tile(base_rate_forecast(sp["cA"]), (N, 1)), cB, rule))
        if s < MC_SPLITS:  # Monte Carlo check of the closed form: one sampled score per observation (seeded)
            mrng = np.random.default_rng(Z.seed(60, 77, si, s))
            cum = np.cumsum(st["pA"], 1)[sp["obs_item"]]
            sampled = np.minimum((mrng.random(len(sp["obs_y"]))[:, None] > cum).sum(1), K - 1)
            r_mc = np.corrcoef(sampled, sp["obs_y"])[0, 1]
            _, r_cf = corr_stats(W1, st["nB"], st["S1"], st["S2"], st["mu"], st["m2"])
            mc_diffs.append(float(r_mc - r_cf[0]))
    A = {k: np.array(v) for k, v in cols.items()}  # [S, N]
    # correlations: point estimates and bootstrap draws (statistic averaged over the splits inside every draw)
    rm0 = np.mean([corr_stats(W1, A["nB"][s], A["S1"][s], A["S2"][s], A["mu"][s], A["m2"][s])[0][0] for s in range(SPLITS)])
    rs0 = np.mean([corr_stats(W1, A["nB"][s], A["S1"][s], A["S2"][s], A["mu"][s], A["m2"][s])[1][0] for s in range(SPLITS)])
    rm_b, rs_b = np.zeros(B), np.zeros(B)
    for s in range(SPLITS):
        a, b = corr_stats(W, A["nB"][s], A["S1"][s], A["S2"][s], A["mu"][s], A["m2"][s])
        rm_b += a / SPLITS
        rs_b += b / SPLITS
    # theory from all ratings of each item
    n_all = d["counts"].sum(1)
    lab = np.arange(K, dtype=float)
    m_all = (d["counts"] @ lab) / n_all
    q_all = (d["counts"] @ lab ** 2) / n_all
    nA_all = np.floor(n_all / 2)
    rho0, rpred0 = theory(W1, n_all, m_all, q_all, nA_all)
    rho_b, rpred_b = theory(W, n_all, m_all, q_all, nA_all)
    # proper score side
    Vm = {k: np.array(v).T for k, v in V.items()}  # [N, S]
    def risk_draws(M):
        return ((W @ M) / W.sum(1, keepdims=True)).mean(1)
    rT0, rC0, rB0 = (Vm[k].mean() for k in ("truthful", "collapsed", "base"))
    rT_b, rC_b, rB_b = (risk_draws(Vm[k]) for k in ("truthful", "collapsed", "base"))
    sk = lambda r, b: 1.0 - r / b  # noqa: E731
    row = dict(dataset=name, rule=rule, n_items=N, n_splits=SPLITS, bootstrap_draws=B,
               rho_hat=float(rho0[0]), rho_hat_lo=pct_ci(rho_b)[0], rho_hat_hi=pct_ci(rho_b)[1], sqrt_rho_hat=float(np.sqrt(rho0[0])),
               r_mean_pred_finite_half=float(rpred0[0]),
               r_sampled=float(rs0), r_sampled_lo=pct_ci(rs_b)[0], r_sampled_hi=pct_ci(rs_b)[1],
               r_mean_judge=float(rm0), r_mean_lo=pct_ci(rm_b)[0], r_mean_hi=pct_ci(rm_b)[1],
               r_mean_minus_r_sampled=float(rm0 - rs0), diff_lo=pct_ci(rm_b - rs_b)[0], diff_hi=pct_ci(rm_b - rs_b)[1],
               sampled_minus_rho_hat=float(rs0 - rho0[0]), mean_minus_prediction=float(rm0 - rpred0[0]),
               mc_max_abs_diff=float(np.max(np.abs(mc_diffs))), mc_mean_diff=float(np.mean(mc_diffs)),
               risk_truthful=float(rT0), risk_collapsed=float(rC0), risk_base=float(rB0),
               skill_truthful=float(sk(rT0, rB0)), skill_truthful_lo=pct_ci(sk(rT_b, rB_b))[0], skill_truthful_hi=pct_ci(sk(rT_b, rB_b))[1],
               skill_collapsed=float(sk(rC0, rB0)), skill_collapsed_lo=pct_ci(sk(rC_b, rB_b))[0], skill_collapsed_hi=pct_ci(sk(rC_b, rB_b))[1],
               risk_collapsed_minus_truthful=float(rC0 - rT0), risk_diff_lo=pct_ci(rC_b - rT_b)[0], risk_diff_hi=pct_ci(rC_b - rT_b)[1])
    return row


def decide(rows):
    h1 = all(r["diff_lo"] > 0 for r in rows)
    ok = [(abs(r["sampled_minus_rho_hat"]) <= 0.02 and abs(r["mean_minus_prediction"]) <= 0.02) for r in rows]
    h3 = all(r["risk_diff_lo"] > 0 for r in rows)
    return {"H1": dict(rule="r(mean judge) minus r(sampled judge) positive with the 95% interval excluding 0 on every one of the 3 sets", decision="supported" if h1 else "falsified",
                       per_set={r["dataset"]: [r["diff_lo"], r["r_mean_minus_r_sampled"], r["diff_hi"]] for r in rows}),
            "H2": dict(rule="|r(sampled) - rho_hat| <= 0.02 and |r(mean judge) - finite-half prediction| <= 0.02 on at least 2 of 3 sets", n_sets=int(sum(ok)),
                       per_set={r["dataset"]: [r["sampled_minus_rho_hat"], r["mean_minus_prediction"]] for r in rows}, decision="supported" if sum(ok) >= 2 else "falsified"),
            "H3": dict(rule="matched-rule risk of the collapsed forecast minus that of the truthful forecast is positive with the 95% interval excluding 0 on every set", decision="supported" if h3 else "falsified",
                       per_set={r["dataset"]: [r["risk_diff_lo"], r["risk_collapsed_minus_truthful"], r["risk_diff_hi"]] for r in rows})}


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.3, 5.4))
    blue, orange = "#0072B2", "#D55E00"
    rho = np.linspace(0, 1, 200)
    ax.plot(rho, rho, color=blue, lw=1.2, ls="--")
    ax.plot(rho, np.sqrt(rho), color=orange, lw=1.2)
    mk = {"chaosnli_alphanli": "o", "dices350": "s", "dices990": "^"}
    nm = {"chaosnli_alphanli": "alphaNLI", "dices350": "DICES-350", "dices990": "DICES-990"}
    for r in rows:
        x = r["rho_hat"]
        xerr = [[x - r["rho_hat_lo"]], [r["rho_hat_hi"] - x]]
        for key, lo, hi, col, fill in (("r_sampled", "r_sampled_lo", "r_sampled_hi", blue, "white"), ("r_mean_judge", "r_mean_lo", "r_mean_hi", orange, orange)):
            ax.errorbar([x], [r[key]], xerr=xerr, yerr=[[r[key] - r[lo]], [r[hi] - r[key]]], fmt=mk[r["dataset"]], mfc=fill, mec=col, color=col, ms=5, lw=0.8, capsize=1.5, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Between-item share of rating variance $\rho$")
    ax.set_ylabel("Pearson r with held-out ratings (half B)")
    ax.set_title("(a) Same rating law, two readouts", fontsize=7.5)
    ax.grid(True)
    ax.text(0.03, 0.97, "collapsed judge (filled)\noutputs the mean:\n" + r"$r=\sqrt{\rho}$", color=orange, ha="left", va="top", fontsize=7, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
    ax.text(0.97, 0.47, "truthful judge (open)\nscore sampled from p:\n" + r"$r=\rho$", color=blue, ha="right", va="top", fontsize=7, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
    ax.legend(handles=[Line2D([], [], color="0.2", marker=mk[k], ls="", mfc="0.2", ms=4.5, label=nm[k]) for k in mk], loc="lower right", frameon=False, handletextpad=0.2, labelspacing=0.25)
    xs = np.arange(len(rows))
    for off, key, lab, col in ((-0.2, "truthful", "truthful forecast", blue), (0.2, "collapsed", "collapsed forecast (point mass)", orange)):
        y = np.array([r[f"skill_{key}"] for r in rows])
        lo = y - np.array([r[f"skill_{key}_lo"] for r in rows])
        hi = np.array([r[f"skill_{key}_hi"] for r in rows]) - y
        bx.bar(xs + off, y, width=0.38, color=col, label=lab, zorder=3)
        bx.errorbar(xs + off, y, yerr=[lo, hi], fmt="none", ecolor="k", lw=0.8, capsize=1.5, zorder=4)
    bx.axhline(0, color="0.3", lw=0.8)
    bx.set_ylim(-0.75, 1.15)
    bx.set_xticks(xs)
    bx.set_xticklabels([nm[r["dataset"]] for r in rows])
    bx.set_ylabel("Skill on held-out half B\n(higher is better; 0 = base rate)")
    bx.set_xlabel("Rating set (matched proper score: Brier for alphaNLI,\nnormalized ranked probability score for DICES)")
    bx.set_title("(b) The collapsed forecast scores worse", fontsize=7.5)
    bx.grid(True, axis="both")
    bx.set_axisbelow(True)
    bx.legend(loc="upper right", fontsize=6.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / "figure1_sampled_score.png", dpi=300)
    fig.savefig(OUT / "figure1_sampled_score.pdf")
    plt.close(fig)
    Z.write_csv(OUT / "figure1_values.csv", [dict(dataset=r["dataset"], rho_hat=r["rho_hat"], r_sampled=r["r_sampled"], r_mean_judge=r["r_mean_judge"],
                                                  skill_truthful=r["skill_truthful"], skill_collapsed=r["skill_collapsed"]) for r in rows])


def main():
    OUT.mkdir(exist_ok=True)
    rows = [analyse_set(n) for n in SETS]
    dec = decide(rows)
    dec.update(splits=SPLITS, bootstrap_draws=B, mc_splits=MC_SPLITS)
    Z.write_csv(OUT / "sampled_vs_mean.csv", rows)
    (OUT / "hypotheses.json").write_text(json.dumps(dec, indent=2, default=float))
    make_figure(rows)
    L = ["# Experiment 60 summary (ORACLE CHECK on empirical human rating laws; no language model involved)", "",
         "set | rho_hat [95%] | r(sampled) [95%] | r(mean judge) [95%] | sqrt(rho_hat) | finite-half prediction | difference r(mean) - r(sampled) [95%]", "---|---|---|---|---|---|---"]
    for r in rows:
        L.append(f"{r['dataset']} | {r['rho_hat']:.3f} [{r['rho_hat_lo']:.3f}, {r['rho_hat_hi']:.3f}] | {r['r_sampled']:.3f} [{r['r_sampled_lo']:.3f}, {r['r_sampled_hi']:.3f}] | "
                 f"{r['r_mean_judge']:.3f} [{r['r_mean_lo']:.3f}, {r['r_mean_hi']:.3f}] | {r['sqrt_rho_hat']:.3f} | {r['r_mean_pred_finite_half']:.3f} | "
                 f"{r['r_mean_minus_r_sampled']:+.3f} [{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}]")
    L += ["", "set | matched rule | skill truthful [95%] | skill collapsed [95%] | risk collapsed minus risk truthful [95%] (positive = truthful better) | Monte Carlo max |difference| from closed form", "---|---|---|---|---|---"]
    for r in rows:
        L.append(f"{r['dataset']} | {r['rule']} | {r['skill_truthful']:+.3f} [{r['skill_truthful_lo']:+.3f}, {r['skill_truthful_hi']:+.3f}] | {r['skill_collapsed']:+.3f} [{r['skill_collapsed_lo']:+.3f}, {r['skill_collapsed_hi']:+.3f}] | "
                 f"{r['risk_collapsed_minus_truthful']:+.4f} [{r['risk_diff_lo']:+.4f}, {r['risk_diff_hi']:+.4f}] | {r['mc_max_abs_diff']:.4f}")
    L += ["", "## Decisions", f"- H1: {dec['H1']['decision']}", f"- H2: {dec['H2']['decision']} ({dec['H2']['n_sets']} of 3 sets within 0.02)", f"- H3: {dec['H3']['decision']}"]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
