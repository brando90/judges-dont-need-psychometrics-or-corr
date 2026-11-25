"""Scaling-law fits for CE-fit vs Pearson-fit judges (11-25-2026), from the saved results of Exp 26, 28 and 29. No model calls.
Usage: python3 fit_scaling_laws.py [REPO_ROOT]   (needs numpy, scipy-free; matplotlib for figures)
Model: error(N) = floor + a * (N / N_min)^(-b), floor >= 0, a >= 0, 0.02 <= b <= 3, N = human ratings used to fit.
Fit: weighted least squares (weights 1/SE^2), exact over a grid of 150 values of b (floor and a solved in closed form).
Intervals: parametric bootstrap (2000 draws, each point ~ Normal(value, SE^2), points independent), percentile 95%.
SE per point: Exp 26 from the saved 95% interval, Exp 28 from the saved paired test-item bootstrap SE, Exp 29 draw SD / sqrt(10 draws)."""
import csv, glob, sys, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1]
EXP = ROOT / "experiments"
E26, E28, E29 = [Path(glob.glob(str(EXP / f"{n}_*"))[0]) for n in (26, 28, 29)]
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
RES = HERE / "results"; RES.mkdir(exist_ok=True)
rd = lambda p: list(csv.DictReader(open(p)))
BOOT, NDRAW29 = 2000, 10
BGRID = np.exp(np.linspace(np.log(0.02), np.log(3.0), 150))
RNG = np.random.default_rng(20261125)
MET = ("rps", "ce", "tvd")
MNAME = {"rps": "RPS", "ce": "CE (nats)", "tvd": "TVD"}
COL = {"ce": "#0072B2", "pearson": "#D55E00", "uniform": "#7F7F7F", "raw": "#000000"}
SETS = {"chaosnli_snli": "ChaosNLI SNLI", "chaosnli_mnli": "ChaosNLI MNLI-m", "chaosnli_alphanli": "alphaNLI (2 classes)", "dices350": "DICES-350", "dices990": "DICES-990"}
JUDGES = {"gpt-6-astra": "GPT-6 Astra", "claude-opus-5-5": "Claude Opus 5.5", "kimi-k3": "Kimi K3", "glm-5.3": "GLM 5.3", "deepseek-v4.1-flash": "DeepSeek V4.1 Flash"}
PANELS29 = {("chaos3_qwen", "1"): "ChaosNLI, Qwen3-4B, 1 rating per item", ("chaos3_qwen", "5"): "ChaosNLI, Qwen3-4B, 5 ratings per item",
            ("alpha2_qwen", "1"): "alphaNLI, Qwen3-4B", ("dices3_qwen", "1"): "DICES, Qwen3-4B", ("hs2_qwen", "1"): "HelpSteer2, Qwen3-4B",
            ("chaos3_llama", "1"): "ChaosNLI, Llama-3.1-8B", ("dices3_llama", "1"): "DICES, Llama-3.1-8B"}
H28 = "Exp 28 harness: judges GPT-6 Astra, Claude Opus 5.5, Kimi K3, GLM 5.3, DeepSeek V4.1 Flash via the Stanford AI API Gateway; fit = 27-transform head chosen by CE or by Pearson"
H26 = "Exp 26 harness: the same five judges via the Stanford AI API Gateway chat completions; CE-fit and Pearson-fit heads, mean of 10 folds, 2 ratings per answer"
H29 = "Exp 29 harness: LoRA fine-tuning of Qwen3-4B-Instruct-2507 or Llama-3.1-8B with a CE or a Pearson objective (raw, uncalibrated), mean of 10 nested draws"


def fit_batch(N, Y, se):
    """Y [S,K] -> floor [S], a [S], b [S] (nan where the fitted a is 0), fitted [S,K]. Weighted LS, exact over BGRID."""
    K = len(N); w = 1.0 / np.maximum(se, 1e-9) ** 2
    best = None
    for b in BGRID:
        x = (N / N[0]) ** (-b)
        S11, S12, S22 = w.sum(), (w * x).sum(), (w * x * x).sum()
        t1, t2 = Y @ w, Y @ (w * x)
        det = S11 * S22 - S12 ** 2
        f = (S22 * t1 - S12 * t2) / det; a = (S11 * t2 - S12 * t1) / det
        m = a < 0; a = np.where(m, 0.0, a); f = np.where(m, t1 / S11, f)
        m = f < 0; f = np.where(m, 0.0, f); a = np.where(m & (a > 0), t2 / S22, a)
        a = np.maximum(a, 0.0)
        fit = f[:, None] + a[:, None] * x[None, :]
        rss = (((Y - fit) / np.maximum(se, 1e-9)) ** 2).sum(1)
        if best is None: best = [rss, f, a, np.full(len(Y), b), fit]
        else:
            u = rss < best[0] - 1e-12
            best[0] = np.where(u, rss, best[0]); best[1] = np.where(u, f, best[1]); best[2] = np.where(u, a, best[2])
            best[3] = np.where(u, b, best[3]); best[4] = np.where(u[:, None], fit, best[4])
    b = np.where(best[2] > 1e-12, best[3], np.nan)
    return best[1], best[2], b, best[4]


def curve_fit(N, y, se):
    """Fit one curve and bootstrap it. Returns a dict of fit results and flags."""
    N = np.asarray(N, float); y = np.asarray(y, float); se = np.asarray(se, float); K = len(N)
    out = dict(K=K, N=N.tolist(), y=y.tolist(), se=se.tolist())
    rng = max(y[0] - y[-1], 0.0); first_last_se = np.hypot(se[0], se[-1])
    steps = y[1:] - y[:-1]; thr = 1.96 * np.hypot(se[1:], se[:-1])
    out["nonmono_pts"] = int((steps > 0).sum()); out["nonmono_sig"] = int((steps > thr).sum())
    out["decline_z"] = float((y[0] - y[-1]) / first_last_se) if first_last_se > 0 else float("nan")
    if K < 4:
        out.update(fit=False, verdict="not fit (3 points)"); return out
    f, a, b, fit = fit_batch(N, y[None, :], se)
    Yb = y[None, :] + se[None, :] * RNG.standard_normal((BOOT, K))
    fb, ab, bb, fitb = fit_batch(N, Yb, se)
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(v).any() else [float("nan")] * 2
    resid = y - fit[0]
    dof = K - 3
    out.update(fit=True, floor=float(f[0]), a=float(a[0]), b=float(b[0]) if np.isfinite(b[0]) else float("nan"),
               floor_ci=q(fb), b_ci=q(bb), a_ci=q(ab), b_nan_frac=float(np.isnan(bb).mean()),
               b_at_upper=float(np.nanmean(bb >= 2.9)) if np.isfinite(bb).any() else float("nan"),
               rel_rmse=float(np.sqrt(np.mean(resid ** 2)) / rng) if rng > 0 else float("nan"),
               chi2_dof=float((((resid) / np.maximum(se, 1e-9)) ** 2).sum() / dof) if dof > 0 else float("nan"),
               fitted=fit[0].tolist(), floor_boot=fb, a_boot=ab, b_boot=bb, fit_boot=fitb, N0=float(N[0]))
    width = out["floor_ci"][1] - out["floor_ci"][0]
    out["floor_identified"] = bool(width <= rng) if rng > 0 else False
    if out["nonmono_sig"] > 0: v = "non-monotone"
    elif out["decline_z"] < 2: v = "flat (no decline beyond noise)"
    elif out["rel_rmse"] > 0.10: v = "poor fit"
    elif not out["floor_identified"]: v = "fits, floor not identified"
    else: v = "fits"
    if np.isfinite(out["b"]) and out["b"] <= 0.2 and (v.startswith("fits") or v.startswith("poor")): v += "; b <= 0.2, floor is an extrapolation"
    if K == 4: v += " (4 points, 1 dof: indicative)"
    out["verdict"] = v
    return out


# ----------------------------- load curves -----------------------------
def load28():
    s = rd(E28 / "results/scores_absolute.csv"); con = rd(E28 / "results/contrasts.csv"); cor = rd(E28 / "results/correlations.csv")
    raw = {(r["dataset"], r["judge"]): r for r in rd(E28 / "results/raw_judge_scores.csv")}
    val = {(r["dataset"], r["judge"], r["d"], int(r["B"]), r["method"]): r for r in s}
    sec = {(r["dataset"], r["judge"], r["d"], int(r["B"]), r["method"]): r for r in cor}
    cse = {(r["dataset"], r["judge"], r["d"], int(r["B"]), r["metric"], r["comparator"]): float(r["bootstrap_se"]) for r in con if r["bootstrap_se"] != ""}
    cells = {}
    for ds in SETS:
        for j in JUDGES:
            for d in ("1", "5"):
                Bs = sorted({int(r["B"]) for r in s if r["dataset"] == ds and r["judge"] == j and r["d"] == d})
                for m in MET:
                    se_c = np.array([cse[(ds, j, d, B, m, "uniform")] for B in Bs])        # uniform has no fit noise: its contrast SE is the CE-fit SE
                    se_d = np.array([cse[(ds, j, d, B, m, "pearson")] for B in Bs])
                    cells[(ds, j, d, m, "ce")] = dict(N=Bs, y=[float(val[(ds, j, d, B, "ce")][m]) for B in Bs], se=se_c)
                    cells[(ds, j, d, m, "pearson")] = dict(N=Bs, y=[float(val[(ds, j, d, B, "pearson")][m]) for B in Bs], se=np.hypot(se_c, se_d))  # approximation, see results.md
                    cells[(ds, j, d, m, "uniform")] = float(val[(ds, j, d, Bs[0], "uniform")][m])
                    cells[(ds, j, d, m, "raw")] = float(raw[(ds, j)][m])
                for meth in ("ce", "pearson"):
                    for k, nm in (("pearson_expected_vs_human_mean", "r"), ("spearman_expected_vs_human_mean", "rho")):
                        cells[(ds, j, d, nm, meth)] = float(sec[(ds, j, d, max(Bs), meth)][k])
    return cells


def load26():
    s = rd(E26 / "results/new5/scores.csv"); g = defaultdict(dict)
    for r in s: g[(r["judge"], r["ratings_per_answer"], r["metric"], r["method"])][int(r["training_ratings"])] = r
    cells = {}
    for j in JUDGES:
        for rpa in ("1", "2"):
            for m in MET:
                for meth, key in (("ce", "ce_fit"), ("pearson", "pearson_fit")):
                    d = g[(j, rpa, m, key)]; Ns = sorted(d)
                    cells[(j, rpa, m, meth)] = dict(N=Ns, y=[float(d[n]["mean"]) for n in Ns], se=np.array([(float(d[n]["upper95"]) - float(d[n]["lower95"])) / 3.92 for n in Ns]))
                cells[(j, rpa, m, "uniform")] = float(np.mean([float(r["mean"]) for r in g[(j, rpa, m, "uniform")].values()]))
                cells[(j, rpa, m, "raw")] = float(np.mean([float(r["mean"]) for r in g[(j, rpa, m, "identity")].values()]))
    return cells


def load29():
    s = rd(E29 / "results/scores.csv"); by = {(r["study"], r["depth"], int(r["b_total"]), r["method"]): r for r in s}
    cells = {}
    for (st, dp) in PANELS29:
        Bs = sorted({int(r["b_total"]) for r in s if r["study"] == st and r["depth"] == dp and r["method"] == "ce_raw"})
        for m in MET:
            for meth, key in (("ce", "ce_raw"), ("pearson", "pearson_raw")):
                cells[(st, dp, m, meth)] = dict(N=Bs, y=[float(by[(st, dp, B, key)][m]) for B in Bs],
                                                se=np.array([float(by[(st, dp, B, key)][m + "_draw_sd"]) / np.sqrt(NDRAW29) for B in Bs]))
            cells[(st, dp, m, "uniform")] = float(by[(st, dp, Bs[0], "uniform")][m])
            cells[(st, dp, m, "raw")] = float(by[(st, dp, Bs[0], "base_raw")][m])
            cells[(st, dp, m, "hist")] = float(by[(st, dp, Bs[-1], "all_dev_histogram")][m])
        for meth, key in (("ce", "ce_raw"), ("pearson", "pearson_raw")):
            for k, nm in (("pearson", "r"), ("spearman", "rho")):
                cells[(st, dp, nm, meth)] = float(by[(st, dp, Bs[-1], key)][k])
    return cells


def avg_curves(cs):
    """Average a list of cell curves (same N): mean of y, mean of SEs (the conservative, fully correlated case)."""
    return dict(N=cs[0]["N"], y=np.mean([c["y"] for c in cs], 0), se=np.mean([c["se"] for c in cs], 0))


# ----------------------------- plots -----------------------------
def plot_one(key, title, caption_h, metric, curves, refs, fname):
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    for meth in ("pearson", "ce"):
        c = curves[meth]; N = np.array(c["N"]); y = np.array(c["y"]); se = np.array(c["se"])
        ax.errorbar(N, y, yerr=1.96 * se, color=COL[meth], marker="o", ms=4, lw=0, elinewidth=1, capsize=2, zorder=3)
        if c.get("fit"):
            xs = np.exp(np.linspace(np.log(N[0]), np.log(N[-1] * 4), 80))
            xin = xs <= N[-1]
            # band and law on a dense grid: refit curve from each bootstrap (floor, a, b)
            fb, ab, bb = c["floor_boot"], c["a_boot"], c["b_boot"]
            bbf = np.where(np.isfinite(bb), bb, 1.0)       # a = 0 there, so b is moot
            curve = fb[:, None] + ab[:, None] * (xs[None, :] / N[0]) ** (-bbf[:, None])
            lo, hi = np.percentile(curve, [2.5, 97.5], 0)
            ax.fill_between(xs[xin], lo[xin], hi[xin], color=COL[meth], alpha=0.12, lw=0)
            law = c["floor"] + c["a"] * (xs / N[0]) ** (-(c["b"] if np.isfinite(c["b"]) else 1.0))
            nm = "CE-fit" if meth == "ce" else "Pearson-fit"
            lab = f"{nm}: {c['floor']:.3f} + a N^-{c['b']:.2f}" if np.isfinite(c["b"]) else f"{nm}: flat at {c['floor']:.3f}"
            ax.plot(xs[xin], law[xin], color=COL[meth], lw=2.2, label=lab)
            ax.plot(xs[~xin | (xs >= N[-1])], law[~xin | (xs >= N[-1])], color=COL[meth], lw=1.2, ls=":")
            ax.axhline(c["floor"], color=COL[meth], lw=0.8, ls=(0, (1, 3)), alpha=0.8)
        else:
            ax.plot(N, y, color=COL[meth], lw=1.6, label=("CE-fit" if meth == "ce" else "Pearson-fit") + " (not fit)")
    ax.axhline(refs["uniform"], color=COL["uniform"], lw=1.6, label=f"uniform {refs['uniform']:.3f}")
    ax.axhline(refs["raw"], color=COL["raw"], lw=1.2, ls="--", label=f"{refs.get('raw_name', 'raw judge')} {refs['raw']:.3f}")
    allv = np.concatenate([np.array(curves[m_]["y"]) + sg * 1.96 * np.array(curves[m_]["se"]) for m_ in ("ce", "pearson") for sg in (-1, 1)])
    span = allv.max() - allv.min(); top = max(allv.max(), refs["uniform"], refs["raw"]) + 0.05 * span
    ax.set_ylim(allv.min() - 0.5 * span, top)      # floors near 0 would squash the data: clip below
    ax.set_xscale("log", base=2); ax.grid(True, alpha=0.35)
    ax.set_xlabel("human ratings used to fit (N)", fontsize=10); ax.set_ylabel(MNAME[metric] + " (held-out, lower is better)", fontsize=10)
    ax.set_title(title + " | " + MNAME[metric], fontsize=11); ax.tick_params(labelsize=9)
    ax.legend(fontsize=7.5, loc="best", framealpha=0.9)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=130); plt.close(fig)


def main():
    c28, c26, c29 = load28(), load26(), load29()
    rows = []          # one row per fitted curve
    pair = []          # one row per panel x metric: CE vs Pearson floor comparison
    plotted = []

    def one(exp, panel_id, panel_name, harness, metric, curves, refs, fname_stem, plot=True, level="panel"):
        fits = {}
        for meth in ("ce", "pearson"):
            c = curves[meth]; fits[meth] = curve_fit(c["N"], c["y"], c["se"])
        for meth in ("ce", "pearson"):
            r = fits[meth]
            c_ = curves[meth]
            rows.append(dict(exp=exp, level=level, panel_id=panel_id, panel=panel_name, metric=metric, objective="CE" if meth == "ce" else "Pearson", K=r["K"],
                             N_min=r["N"][0], N_max=r["N"][-1], first=r["y"][0], last=r["y"][-1],
                             floor=r.get("floor"), floor_lo=(r.get("floor_ci") or [None, None])[0], floor_hi=(r.get("floor_ci") or [None, None])[1],
                             b=r.get("b"), b_lo=(r.get("b_ci") or [None, None])[0], b_hi=(r.get("b_ci") or [None, None])[1], a_at_Nmin=r.get("a"),
                             boot_frac_no_decay=r.get("b_nan_frac"), boot_frac_b_at_upper=r.get("b_at_upper"),
                             rel_rmse=r.get("rel_rmse"), chi2_per_dof=r.get("chi2_dof"), nonmono_points=r["nonmono_pts"], nonmono_significant=r["nonmono_sig"],
                             decline_z=r["decline_z"], floor_identified=r.get("floor_identified"), verdict=r["verdict"],
                             uniform=refs["uniform"], raw=refs["raw"], hist_ref=refs.get("hist")))
        fc, fp = fits["ce"], fits["pearson"]
        if fc.get("fit") and fp.get("fit"):
            d = fp["floor_boot"] - fc["floor_boot"]
            gb = fp["y"][-1] + fp["se"][-1] * RNG.standard_normal(BOOT) - fc["floor_boot"]
            pair.append(dict(exp=exp, level=level, panel_id=panel_id, panel=panel_name, metric=metric, floor_ce=fc["floor"], floor_pearson=fp["floor"],
                             gap=fp["y"][-1] - fc["floor"], gap_lo=float(np.percentile(gb, 2.5)), gap_hi=float(np.percentile(gb, 97.5)), pearson_last=fp["y"][-1],
                             diff=fp["floor"] - fc["floor"], diff_lo=float(np.percentile(d, 2.5)), diff_hi=float(np.percentile(d, 97.5)),
                             pearson_floor_above_ce_floor=bool(np.percentile(d, 2.5) > 0),
                             ce_floor_over_uniform=fc["floor"] / refs["uniform"], ce_floor_minus_raw=fc["floor"] - refs["raw"],
                             ce_floor_below_raw=bool(fc["floor_ci"][1] < refs["raw"]), pearson_floor_below_raw=bool(fp["floor_ci"][1] < refs["raw"])))
        if plot and (fc.get("fit") or fp.get("fit") or True):
            plot_one(panel_id, panel_name, harness, metric, fits, refs, fname_stem + ".png")
            plotted.append(fname_stem + ".png")
        return fits

    # Exp 28: per set (mean over five judges) and per judge-by-set cell; d=1 and d=5
    for d in ("1", "5"):
        for ds, nm in SETS.items():
            for m in MET:
                curves = {meth: avg_curves([c28[(ds, j, d, m, meth)] for j in JUDGES]) for meth in ("ce", "pearson")}
                refs = dict(uniform=float(np.mean([c28[(ds, j, d, m, "uniform")] for j in JUDGES])), raw=float(np.mean([c28[(ds, j, d, m, "raw")] for j in JUDGES])))
                one("28", f"28_{ds}_d{d}", f"{nm} ({'1' if d == '1' else '5'} rating{'s' if d == '5' else ''} per item), mean of 5 judges", H28, m, curves, refs, f"fit_exp28_{m}_{ds}_d{d}", level="set")
            for j, jn in JUDGES.items():
                for m in MET:
                    curves = {meth: c28[(ds, j, d, m, meth)] for meth in ("ce", "pearson")}
                    refs = dict(uniform=c28[(ds, j, d, m, "uniform")], raw=c28[(ds, j, d, m, "raw")])
                    one("28", f"28_{ds}_{j}_d{d}", f"{nm}, {jn}, d={d}", H28, m, curves, refs, "", plot=False, level="cell")
    # Exp 26 per judge, 2 and 1 ratings per answer
    for rpa in ("2", "1"):
        for j, jn in JUDGES.items():
            for m in MET:
                curves = {meth: c26[(j, rpa, m, meth)] for meth in ("ce", "pearson")}
                refs = dict(uniform=c26[(j, rpa, m, "uniform")], raw=c26[(j, rpa, m, "raw")])
                one("26", f"26_{j}_rpa{rpa}", f"Exp 26 {jn}, {rpa} per answer", H26, m, curves, refs, f"fit_exp26_{m}_{j}_rpa{rpa}", plot=(rpa == "2"), level="judge")
    # Exp 29 per panel
    for (st, dp), nm in PANELS29.items():
        for m in MET:
            curves = {meth: c29[(st, dp, m, meth)] for meth in ("ce", "pearson")}
            refs = dict(uniform=c29[(st, dp, m, "uniform")], raw=c29[(st, dp, m, "raw")], hist=c29[(st, dp, m, "hist")], raw_name="base model, no tuning")
            one("29", f"29_{st}_d{dp}", f"Exp 29 {nm}", H29, m, curves, refs, f"fit_exp29_{m}_{st}_d{dp}", level="panel")

    def clean(v):
        return "" if v is None else (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v)
    for name, data in (("fits", rows), ("floor_comparisons", pair)):
        with open(RES / f"{name}_11-25-2026.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader()
            for r in data: w.writerow({k: clean(v) for k, v in r.items()})
    sec = []
    for (ds, j, d, nm, meth), v in [(k, v) for k, v in c28.items() if k[3] in ("r", "rho")]: sec.append(("28", f"{ds}|{j}|d{d}", nm, meth, v))
    json.dump(dict(secondary_corr_28=[list(x) for x in sec],
                   secondary_corr_26=None,
                   secondary_corr_29=[[f"{st}|d{dp}", nm, meth, c29[(st, dp, nm, meth)]] for (st, dp) in PANELS29 for nm in ("r", "rho") for meth in ("ce", "pearson")],
                   plots=plotted), open(RES / "secondary_and_plots_11-25-2026.json", "w"), indent=1)
    print(len(rows), "curves fitted;", len(pair), "pairs;", len(plotted), "plots")


if __name__ == "__main__":
    main()
