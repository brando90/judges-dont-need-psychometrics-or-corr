"""Probe: why does L-BFGS-B stop early, and is the CE head shrunk by L2=0.01? Read-only use of a COPY of the repo code."""
import sys, json
from pathlib import Path
COPY = Path("/tmp/opus_review_exp40/copy/experiments/40_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/scripts")
sys.path.insert(0, str(COPY))
import numpy as np
from scipy.optimize import minimize
from e40_common import d, fe, calibrate, three_level
from metrics import softmax

def view(name):
    if name == "pol_K2":
        b = d.load_popquorn("politeness"); return d.binarize(b, 4, "x"), 2
    if name == "pol_K5":
        return d.load_popquorn("politeness"), 5
    if name == "off_K5":
        return d.load_popquorn("offensiveness"), 5
    if name == "dices_K3":
        return d._load_dices("990"), 3
    if name == "chaos_K3":
        return d.load_chaos_3way(), 3
    if name == "se_exp_K5":
        return d.load_summeval("expert"), 5
    raise ValueError(name)

def nrps(prob, mu, data, idx, k, values):
    rec = fe._item_records(prob, mu, data.ratings, idx, values)
    return float(rec["rps"].mean() / (k - 1))

def fit(x, targets, l2, maxiter, gtol=None, objective="ce", ybar=None, values=None):
    n, dd = x.shape; k = targets.shape[1]
    def loss(w):
        W = w.reshape(dd, k); prob = softmax(x @ W)
        pen = 0.5 * l2 * float(np.sum(W[:-1] ** 2))
        if objective == "ce":
            return float(-(targets * np.log(np.clip(prob, 1e-12, 1))).sum(1).mean() + pen)
    opts = {"maxiter": maxiter, "ftol": 1e-9}
    if gtol is not None: opts["gtol"] = gtol
    r = minimize(loss, np.zeros(dd * k), method="L-BFGS-B", options=opts)
    return r

out = {}
for name in sys.argv[1:]:
    data, k = view(name)
    values = np.asarray(data.class_values, float)
    train, val, test = d.split_by_group(data.groups, seed=0)
    xtr, (xva, xte), rank = fe.design_matrices(data.texts, train, [val, test])
    feat_std = xtr[:, :-1].std(0)
    pack = fe.fit_objective(xtr, data.ratings, train, data.class_values, "ce", None, 0)
    # reproduce the optimizer message with the exact same call
    targets, ybar, kept = fe._soft_targets(data.ratings, train, k, None, np.random.default_rng(0))
    r = fit(xtr, targets, fe.L2, 35)
    W = r.x.reshape(xtr.shape[1], k)
    pt = softmax(xte @ W); mt = pt @ values
    pv = softmax(xva @ W); mv = pv @ values
    base = nrps(pt, mt, data, test, k, values)
    cal = calibrate(mv, mt, data.ratings, val, k)
    base_cal = nrps(cal, cal @ values, data, test, k, values)
    cprob, cmu = fe.constant_marginal(data.ratings, train, k, values)
    const = nrps(np.repeat(cprob[None], len(test), 0), np.full(len(test), cmu), data, test, k, values)
    res = {"feature_std_min_med_max": [float(feat_std.min()), float(np.median(feat_std)), float(feat_std.max())],
           "repo_pack_nit": pack["nit"], "repro_nit": int(r.nit), "message": str(r.message), "success": bool(r.success),
           "fun": float(r.fun), "max_abs_grad_fd": float(np.max(np.abs(r.jac))) if r.jac is not None else None,
           "test_nrps_constant": const, "test_nrps_ce_head_L2_0.01": base, "test_nrps_ce_head_L2_0.01_calibrated": base_cal,
           "max_abs_weight": float(np.max(np.abs(W[:-1])))}
    for l2 in (1e-3, 1e-4, 1e-6):
        r2 = fit(xtr, targets, l2, 2000, gtol=1e-10)
        W2 = r2.x.reshape(xtr.shape[1], k); pt2 = softmax(xte @ W2); mt2 = pt2 @ values
        res[f"test_nrps_ce_head_L2_{l2}"] = nrps(pt2, mt2, data, test, k, values)
        res[f"nit_L2_{l2}"] = int(r2.nit); res[f"msg_L2_{l2}"] = str(r2.message)
    # same L2 0.01, but tight tolerance and many iterations: is the repo head at its optimum?
    r3 = fit(xtr, targets, fe.L2, 2000, gtol=1e-12)
    W3 = r3.x.reshape(xtr.shape[1], k); pt3 = softmax(xte @ W3)
    res["L2_0.01_tight: fun"] = float(r3.fun); res["L2_0.01_tight: nit"] = int(r3.nit)
    res["L2_0.01_tight: test_nrps"] = nrps(pt3, pt3 @ values, data, test, k, values)
    out[name] = res
    print(name, json.dumps(res, indent=1), flush=True)
