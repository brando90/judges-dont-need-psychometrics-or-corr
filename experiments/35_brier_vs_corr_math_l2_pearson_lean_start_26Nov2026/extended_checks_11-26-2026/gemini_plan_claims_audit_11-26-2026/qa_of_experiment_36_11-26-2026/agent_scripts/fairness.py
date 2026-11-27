"""QA diagnostic for experiment 36: is the Pearson-trained head a fair competitor?

Same data, same splits, same 24-component hashed features as exp36 (imports its code).
Adds analytic gradients, a longer optimizer budget, several starts, a small
validation-selected L2 grid, and a second Pearson objective (correlation of the
expected score, as in experiments 24/25/29). Reports test metrics with
item-clustered bootstrap intervals, including paired differences.

Usage: python3 -I fairness.py <dataset_name> <out_json>
"""
import json
import os
import sys
import time

os.environ["SANMI36_DATA"] = "/tmp/qa55/data"
sys.path.insert(0, "/tmp/qa55/code/exp36")
import numpy as np
from scipy.optimize import minimize

import fit_eval
from data_io import SUMMEVAL_BINARY_CUT, _load_dices, binarize, load_chaos_3way, load_summeval
from run_all import prepare

L2_GRID = (0.0, 1e-4, 1e-3, 1e-2)
MAXITER = 3000
N_RANDOM = 2
BOOT = 1000


def load(name):
    if name.startswith("summeval_expert"):
        d = load_summeval("expert")
        return binarize(d, SUMMEVAL_BINARY_CUT, name) if name.endswith("binary") else d
    if name == "chaosnli_snli_mnli":
        return load_chaos_3way()
    if name.startswith("dices990"):
        d = _load_dices("990")
        return binarize(d, 2, name) if name.endswith("binary_yes") else d
    raise ValueError(name)


def targets_of(ratings, idx, k):
    t = np.stack([np.bincount(np.asarray(ratings[i]), minlength=k) / len(ratings[i]) for i in idx])
    return t


def make_obj(x, T, values, kind, l2):
    n, d = x.shape
    K = T.shape[1]
    v = np.asarray(values, float)
    ybar = T @ v
    yc = ybar - ybar.mean()
    var_y = float(np.mean(T @ (v ** 2) - ybar ** 2) + np.var(ybar))  # total rating variance, as exp36

    def f(wflat):
        W = wflat.reshape(d, K)
        P = fit_eval.softmax(x @ W)
        pen = 0.5 * l2 * float(np.sum(W[:-1] ** 2))
        gW_pen = np.zeros_like(W)
        gW_pen[:-1] = l2 * W[:-1]
        if kind == "ce":
            Pc = np.clip(P, 1e-300, 1)
            val = float(-(T * np.log(Pc)).sum(1).mean())
            dZ = (P - T) / n
        else:
            mu = P @ v
            mbar = mu.mean()
            muc = mu - mbar
            cov = float(np.mean(muc * yc))
            if kind == "psamp":  # exp36 surrogate: correlation of a SAMPLED rating
                var_x = float(np.mean(P @ (v ** 2)) - mbar ** 2)
                dvx = (np.outer(np.ones(n), v ** 2) - 2 * mbar * np.outer(np.ones(n), v)) / n
            elif kind == "pmean":  # correlation of the EXPECTED score (exp24/25/29 definition)
                var_x = float(np.mean(muc ** 2))
                dvx = 2 * np.outer(muc, v) / n
            else:
                raise ValueError(kind)
            if var_x <= 1e-15:
                return pen, gW_pen.ravel()
            S = np.sqrt(var_x * var_y)
            val = -cov / S
            dcov = np.outer(yc, v) / n
            G = -dcov / S + (cov / (2 * S * var_x)) * dvx
            dZ = P * (G - np.sum(P * G, axis=1, keepdims=True))
        gW = x.T @ dZ + gW_pen
        return val + pen, gW.ravel()

    return f


def check_grad(f, w):
    val, g = f(w)
    rng = np.random.default_rng(0)
    errs = []
    for j in rng.choice(w.size, size=8, replace=False):
        e = np.zeros_like(w)
        e[j] = 1e-6
        num = (f(w + e)[0] - f(w - e)[0]) / 2e-6
        errs.append(abs(num - g[j]) / max(1e-8, abs(num) + abs(g[j])))
    return max(errs)


def fit(x, T, values, kind, l2, w0):
    f = make_obj(x, T, values, kind, l2)
    r = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": MAXITER, "ftol": 1e-13, "gtol": 1e-10})
    return r


def pooled_r(mu, ratings, idx, values):
    n = sy = sy2 = ss = ss2 = ssy = 0.0
    for row, i in enumerate(idx):
        ys = values[np.asarray(ratings[i])]
        k = ys.size
        s = mu[row]
        n += k; sy += ys.sum(); sy2 += ys @ ys; ss += s * k; ss2 += s * s * k; ssy += s * ys.sum()
    my, ms = sy / n, ss / n
    vy, vs = sy2 / n - my * my, ss2 / n - ms * ms
    if vs <= 1e-15:
        return float("nan")
    return float((ssy / n - my * ms) / np.sqrt(vy * vs))


def item_losses(P, ratings, idx, values):
    K = P.shape[1]
    ce, rpsn, stats = [], [], []
    v = np.asarray(values, float)
    for row, i in enumerate(idx):
        y = np.asarray(ratings[i])
        q = P[row]
        cdf = np.cumsum(q)[:-1]
        c = np.mean([-np.log(max(q[t], 1e-300)) for t in y])
        r = np.mean([np.sum((cdf - (np.arange(K - 1) >= t)) ** 2) for t in y]) / (K - 1)
        ys = v[y]
        s = float(q @ v)
        k = ys.size
        ce.append(c); rpsn.append(r)
        stats.append((k, ys.sum(), ys @ ys, s * k, s * s * k, s * ys.sum()))
    return np.array(ce), np.array(rpsn), np.array(stats)


def r_from_stats(st):
    n, sy, sy2, ss, ss2, ssy = st.sum(0)
    my, ms = sy / n, ss / n
    vy, vs = sy2 / n - my * my, ss2 / n - ms * ms
    if vs <= 1e-15:
        return float("nan")
    return float((ssy / n - my * ms) / np.sqrt(vy * vs))


def paired(la, lb, sa, sb, seed=1):
    """Item-clustered percentile 95% CI of (b minus a) for mean loss, and (r_a minus r_b)."""
    rng = np.random.default_rng(seed)
    n = la.size
    d_loss, d_r = [], []
    for _ in range(BOOT):
        idx = rng.integers(0, n, n)
        d_loss.append(lb[idx].mean() - la[idx].mean())
        d_r.append(r_from_stats(sa[idx]) - r_from_stats(sb[idx]))
    q = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    return {"loss_b_minus_a": float(lb.mean() - la.mean()), "ci_loss": q(d_loss),
            "r_a_minus_r_b": float(r_from_stats(sa) - r_from_stats(sb)), "ci_r": q(d_r)}


def main():
    name, out = sys.argv[1], sys.argv[2]
    t0 = time.time()
    data = load(name)
    prep = prepare(data)
    v = data.class_values
    K = data.n_classes
    xtr, xva, xte = prep["x_train"], prep["x_val"], prep["x_test"]
    Ttr = targets_of(data.ratings, prep["train"], K)
    d = xtr.shape[1]
    res = {"dataset": name, "n_train": int(len(prep["train"])), "n_val": int(len(prep["val"])),
           "n_test": int(len(prep["test"])), "runs": []}

    # gradient check at a random point
    w_rand = np.random.default_rng(5).normal(0, 0.3, d * K)
    res["grad_check_max_rel_err"] = {k: check_grad(make_obj(xtr, Ttr, v, k, 1e-3), w_rand) for k in ("ce", "psamp", "pmean")}

    # exp36 default heads (numeric gradient, zero start, 35 iterations, L2=1e-2)
    defaults = {}
    for obj in ("ce", "pearson"):
        pack = fit_eval.fit_objective(xtr, data.ratings, prep["train"], v, obj, None, 0)
        defaults[obj] = pack["weight"]

    def evaluate(W, tag):
        Pva = fit_eval.softmax(xva @ W)
        Pte = fit_eval.softmax(xte @ W)
        Ptr = fit_eval.softmax(xtr @ W)
        ce_va, rp_va, st_va = item_losses(Pva, data.ratings, prep["val"], v)
        ce_te, rp_te, st_te = item_losses(Pte, data.ratings, prep["test"], v)
        mu_tr = Ptr @ v
        fs = make_obj(xtr, Ttr, v, "psamp", 0.0)(W.ravel())[0]
        fm = make_obj(xtr, Ttr, v, "pmean", 0.0)(W.ravel())[0]
        return {"tag": tag,
                "train_surrogate_psamp_corr": -fs, "train_expected_score_corr": -fm,
                "val_ce": float(ce_va.mean()), "val_rps_norm": float(rp_va.mean()), "val_r": r_from_stats(st_va),
                "test_ce": float(ce_te.mean()), "test_rps_norm": float(rp_te.mean()), "test_r": r_from_stats(st_te),
                "test_pred_entropy": float(np.mean(-np.sum(np.clip(Pte, 1e-300, 1) * np.log(np.clip(Pte, 1e-300, 1)), 1))),
                "_ce_te": ce_te, "_rp_te": rp_te, "_st_te": st_te}

    E = {}
    E["exp36_ce"] = evaluate(defaults["ce"], "exp36 default CE head (L2=1e-2, 35 it, zero start)")
    E["exp36_pearson"] = evaluate(defaults["pearson"], "exp36 default Pearson head (psamp, L2=1e-2, 35 it, zero start)")

    # longer budget, several starts, L2 grid; CE solution used as one start for the Pearson objectives
    best = {}
    for kind in ("ce", "psamp", "pmean"):
        for l2 in L2_GRID:
            starts = [("zero", np.zeros(d * K))]
            if kind != "ce":
                ce_w = best.get(("ce", l2), {}).get("w")
                if ce_w is not None:
                    starts.append(("ce_solution", ce_w.copy()))
            rng = np.random.default_rng(int(1e6 * l2) + 17)
            for s in range(N_RANDOM):
                starts.append((f"rand{s}", rng.normal(0, 0.5, d * K)))
            cand = []
            for sname, w0 in starts:
                r = fit(xtr, Ttr, v, kind, l2, w0)
                cand.append((float(r.fun), sname, r))
                res["runs"].append({"kind": kind, "l2": l2, "start": sname, "fun": float(r.fun), "nit": int(r.nit),
                                    "msg": str(r.message)})
            cand.sort(key=lambda c: c[0])
            fun, sname, r = cand[0]
            best[(kind, l2)] = {"w": r.x, "start": sname, "fun": fun}
    # validation selection over the L2 grid
    sel = {}
    for kind in ("ce", "psamp", "pmean"):
        evs = {l2: evaluate(best[(kind, l2)]["w"].reshape(d, K), f"{kind} l2={l2} start={best[(kind, l2)]['start']}") for l2 in L2_GRID}
        for l2, ev in evs.items():
            E[f"{kind}_l2={l2}"] = ev
        if kind == "ce":
            crit = {"own": lambda e: e["val_ce"], "rps": lambda e: e["val_rps_norm"]}
        else:
            crit = {"own": lambda e: -e["val_r"] if e["val_r"] == e["val_r"] else 1e9, "rps": lambda e: e["val_rps_norm"]}
        for cname, c in crit.items():
            l2_star = min(L2_GRID, key=lambda l2: c(evs[l2]))
            sel[f"{kind}_sel_{cname}"] = l2_star
            E[f"{kind}_sel_{cname}"] = dict(evs[l2_star], tag=f"{kind} selected by val {cname}: l2={l2_star}")
    res["selected_l2"] = sel

    # paired comparisons against the CE head selected on validation RPS (and the exp36 CE head)
    comps = {}
    ref_pairs = [("exp36_ce", "exp36_pearson"), ("ce_sel_rps", "psamp_sel_own"), ("ce_sel_rps", "pmean_sel_own"),
                 ("ce_sel_rps", "psamp_sel_rps"), ("ce_sel_rps", "pmean_sel_rps"), ("exp36_ce", "pmean_sel_own")]
    for a, b in ref_pairs:
        ea, eb = E[a], E[b]
        comps[f"{b} vs {a}"] = {"rps_norm": paired(ea["_rp_te"], eb["_rp_te"], ea["_st_te"], eb["_st_te"]),
                                "ce": paired(ea["_ce_te"], eb["_ce_te"], ea["_st_te"], eb["_st_te"])}
    res["paired"] = comps
    res["evals"] = {k: {kk: vv for kk, vv in e.items() if not kk.startswith("_")} for k, e in E.items()}
    res["seconds"] = time.time() - t0
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("dataset", "grad_check_max_rel_err", "selected_l2", "seconds")}, default=float))


if __name__ == "__main__":
    main()
