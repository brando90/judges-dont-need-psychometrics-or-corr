#!/usr/bin/env python3
"""Matched-calibration control for Experiment 36's trained heads (commit fcd8db9), and the correlation the Pearson-trained head was trained on.

Claude, 11-26-2026. It uses Experiment 36's own loaders, split, features, head fitting and per-item loss code, and adds two things the experiment does not contain:

(A) Matched calibration. Experiment 36's headline compares the held-out proper scores of two heads whose raw outputs were trained by different
    objectives. The Pearson objective is blind to the scale and location of the scalar score, so the Pearson-trained head's raw output
    distribution is not calibrated by anything in its training. Here every head's scalar mean score (the mean of its output distribution)
    is passed through the SAME calibration map before scoring: a multinomial logit on [standardized mean score, 1] with a tiny ridge on the
    slope (1e-4), fitted by cross-entropy on the VALIDATION ratings (Experiment 36 never reads its validation split), then scored on the
    held-out TEST items. The map has two parameters per class, and it uses only the validation items, never the test items.
    The fair contrast is "calibrated Pearson-trained head minus calibrated CE-trained head". The raw contrast is Experiment 36's headline.
    Differences are paired (same test items), item-clustered percentile bootstrap, 2,000 draws, seed 1, 95% level. Lower normalized RPS is better;
    a positive "Pearson minus CE" difference means the CE-trained head is better.

(B) The correlation the Pearson head was trained on. Experiment 36's Pearson objective divides by Var(X) = E[Var(X|prompt)] + Var(mean), the variance of a
    SAMPLED judge, but its held-out Pearson column scores only the head's MEAN. This script reports both on the test items for both heads, so that
    "the Pearson-trained head does not win its own objective" can be judged against the objective that was actually trained.

Sanity checks built in: my per-item normalized RPS and mean-score Pearson for the RAW heads must reproduce the numbers in the committed results.json
(tolerance 5e-4; the Pearson head's numbers can drift by that much between library versions, see reproduce_matrices.py).

Usage:
    python3 -I matched_calibration_check.py EXPERIMENT_36_FOLDER CACHE_DIR SISTER_REPO_ROOT [name ...]
Default datasets: summeval_expert chaosnli_snli_mnli dices990 popquorn_offensiveness popquorn_politeness veribench_fsc_native_main.
Exit status is nonzero when a sanity check fails.
"""
import json
import os
import sys

sys.dont_write_bytecode = True
e36 = os.path.abspath(sys.argv[1])
os.environ["SANMI36_DATA"] = os.path.abspath(sys.argv[2])
os.environ["TTJ_ROOT"] = os.path.abspath(sys.argv[3])
names = sys.argv[4:] or ["summeval_expert", "chaosnli_snli_mnli", "dices990", "popquorn_offensiveness", "popquorn_politeness", "veribench_fsc_native_main"]
sys.path.insert(0, e36)
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import data_io as d  # noqa: E402
import fit_eval as fe  # noqa: E402
import run_all as ra  # noqa: E402
from metrics import softmax, spearman_rho  # noqa: E402

BUILD = {
    "summeval_expert": lambda: d.load_summeval("expert"),
    "chaosnli_snli_mnli": lambda: d.load_chaos_3way(),
    "dices990": lambda: d._load_dices("990"),
    "popquorn_offensiveness": lambda: d.load_popquorn("offensiveness"),
    "popquorn_politeness": lambda: d.load_popquorn("politeness"),
    "veribench_fsc_native_main": lambda: d.load_veribench_fsc_native_main(),
}
committed = json.load(open(os.path.join(e36, "results.json")))["matrices"]
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (("  " + detail) if detail else ""))
    if not ok:
        fails += 1


def fit_logit_map(z, targets, ridge=1e-4):
    """Multinomial logit on [z, 1], cross-entropy against the validation items' empirical rating histograms."""
    x = np.column_stack([z, np.ones_like(z)])
    n, k = targets.shape

    def f(w):
        wm = w.reshape(2, k)
        p = softmax(x @ wm)
        ll = -(targets * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1).mean()
        g = x.T @ (p - targets) / n
        g[0] += ridge * wm[0]
        return float(ll + 0.5 * ridge * (wm[0] ** 2).sum()), g.ravel()

    res = minimize(f, np.zeros(2 * k), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x.reshape(2, k)


def histograms(ratings, idx, k):
    return np.stack([np.bincount(np.asarray(ratings[i], int), minlength=k) / len(ratings[i]) for i in idx])


def calibrate(mu_val, mu_test, ratings, val_idx, k):
    m, s = float(mu_val.mean()), float(mu_val.std())
    s = s if s > 1e-12 else 1.0
    w = fit_logit_map((mu_val - m) / s, histograms(ratings, val_idx, k))
    x = np.column_stack([(mu_test - m) / s, np.ones_like(mu_test)])
    return softmax(x @ w)


def sampled_and_mean_pearson(prob, mu, ratings, idx, values):
    """Pooled over ratings: correlation of a SAMPLED judge output with a human rating, and of the head's MEAN with a human rating."""
    kk = np.array([len(ratings[i]) for i in idx], float)
    yv = [values[np.asarray(ratings[i], int)] for i in idx]
    ybar = np.array([v.mean() for v in yv])
    y2 = np.array([(v ** 2).mean() for v in yv])
    w = kk / kk.sum()
    mm, my = float((w * mu).sum()), float((w * ybar).sum())
    cov = float((w * (mu - mm) * (ybar - my)).sum())
    var_y = float((w * y2).sum() - my ** 2)
    var_mean = float((w * (mu - mm) ** 2).sum())
    var_cond = float((w * (prob @ values ** 2 - mu ** 2)).sum())
    return cov / np.sqrt((var_cond + var_mean) * var_y), cov / np.sqrt(var_mean * var_y)


def paired(a, b, boot=2000, seed=1):
    n = len(a)
    idx = np.random.default_rng(seed).integers(0, n, size=(boot, n))
    dd = (a[idx] - b[idx]).mean(axis=1)
    return float((a - b).mean()), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))


for name in names:
    data = BUILD[name]()
    values = np.asarray(data.class_values, float)
    k = len(values)
    prep = ra.prepare(data)
    tr, va, te = prep["train"], prep["val"], prep["test"]
    fitted = {}
    for obj in ("ce", "pearson"):
        pack, _ = ra.fit_and_eval(data, prep, obj, None, seed=0, bootstrap=20)
        pv, mv = fe.predict_distributions(pack, prep["x_val"], obj)
        pt, mt = fe.predict_distributions(pack, prep["x_test"], obj)
        fitted[obj] = (pt, mt, mv)
    print(f"\n=== {name}: {len(tr)} train / {len(va)} validation / {len(te)} test items; {k} classes ===")
    per = {}
    for obj in ("ce", "pearson"):
        pt, mt, mv = fitted[obj]
        raw = fe._item_records(pt, mt, data.ratings, te, values)
        qc = calibrate(mv, mt, data.ratings, va, k)
        cal = fe._item_records(qc, qc @ values, data.ratings, te, values)
        per[obj] = {"raw": raw["rps"] / (k - 1), "cal": cal["rps"] / (k - 1), "raw_ce": raw["ce"], "cal_ce": cal["ce"]}
        samp, meanp = sampled_and_mean_pearson(pt, mt, data.ratings, te, values)
        per[obj]["samp"], per[obj]["meanp"] = samp, meanp
        c = committed[name]["rows"][obj]["point"]
        check(f"{name} {obj}: my normalized RPS and mean-score Pearson of the raw head reproduce results.json",
              abs(raw["rps"].mean() / (k - 1) - c["rps_norm"]) < 5e-4 and abs(meanp - c["pearson_r"]) < 5e-4,
              f"nRPS {raw['rps'].mean() / (k - 1):.4f} vs {c['rps_norm']:.4f}; r {meanp:.4f} vs {c['pearson_r']:.4f}")
    mg = committed[name]["marginal"]["point"]
    print(f"FACT  reference forecasts on the same test items (committed): train-marginal constant nRPS {mg['rps_norm']:.4f}, CE {mg['ce']:.4f}; flat forecast nRPS {committed[name]['uniform']['point']['rps_norm']:.4f}")
    print("FACT  held-out test normalized RPS (lower is better) and CE:")
    for obj, label in (("ce", "CE-trained head"), ("pearson", "Pearson-trained head")):
        p = per[obj]
        print(f"        {label:22s} raw nRPS {p['raw'].mean():.4f}, raw CE {p['raw_ce'].mean():.4f} | after the validation-fitted calibration map: nRPS {p['cal'].mean():.4f}, CE {p['cal_ce'].mean():.4f}")
    for lab, a, b in (("RAW (Experiment 36's headline): Pearson-trained head minus CE-trained head", per["pearson"]["raw"], per["ce"]["raw"]),
                      ("MATCHED CALIBRATION: calibrated Pearson-trained head minus calibrated CE-trained head", per["pearson"]["cal"], per["ce"]["cal"]),
                      ("calibrated Pearson-trained head minus RAW CE-trained head (the comparison the Opus reviewer ran)", per["pearson"]["cal"], per["ce"]["raw"])):
        m, lo, hi = paired(a, b)
        print(f"RESULT  nRPS, {lab}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  ({'interval excludes 0' if lo > 0 or hi < 0 else 'interval includes 0'}; positive = CE-trained head better)")
    m, lo, hi = paired(per["pearson"]["cal_ce"], per["ce"]["cal_ce"])
    print(f"RESULT  CE, MATCHED CALIBRATION: calibrated Pearson-trained head minus calibrated CE-trained head: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] nats")
    s_ce, s_pe = per["ce"], per["pearson"]
    print(f"FACT  correlation on the test items, SAMPLED-judge version (the one the Pearson head was trained on) / MEAN-score version (the one Experiment 36 reports): "
          f"CE-trained head {s_ce['samp']:.4f} / {s_ce['meanp']:.4f}; Pearson-trained head {s_pe['samp']:.4f} / {s_pe['meanp']:.4f}")
    rho = spearman_rho(fitted["ce"][1], fitted["pearson"][1])
    print(f"FACT  Spearman correlation between the two heads' scalar mean scores on the test items: {rho:.3f}")
print("\n" + ("ALL SANITY CHECKS PASS" if fails == 0 else f"{fails} SANITY CHECK(S) FAILED"))
sys.exit(1 if fails else 0)
