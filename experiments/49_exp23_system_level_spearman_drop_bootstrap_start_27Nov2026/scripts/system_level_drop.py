#!/usr/bin/env python3
"""Experiment 49: is Experiment 23's SummEval system-level Spearman drop (raw Pearson fit 0.877 vs calibrated 0.835) significant?

Re-runs Experiment 23's analyze.py logic (native mode, depth 1, budgets up to 768, 64 draws), importing its modules unchanged,
keeps every draw's test forecasts for the Pearson-selected head ("pearson") and the four-bin distribution calibration
("calibrated_pearson"), then resamples test documents and systems.  Also adds a continuous calibrator (Experiment 41's
multinomial logit on the standardized Pearson-selected mean score, fitted on the same training votes) as a diagnostic.
Usage: python system_level_drop.py INPUTS/summeval_consistency.npz OUT_DIR
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, kendalltau

HERE = Path(__file__).resolve().parent
E23 = HERE.parents[1] / "23_public_ordinal_judge_ratings_and_label_cost_start_21Nov2026"
E41S = HERE.parents[1] / "41_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026" / "scripts"
sys.path.insert(0, str(E23))
from candidates import R, best, bin_distribution, bin_indices, build, loss_by_grade, pearson_for_pair, pearson_from_totals, per_item  # noqa
from design import draw, load  # noqa
from diagnostics import diagnostics, system_scores, tie_levels  # noqa
from scipy.optimize import minimize

REPEATS, BUDGET, DEPTH, BOOT, SEED = 64, 768, 1, 2000, 20261127
BUDGETS = (6, 12, 24, 48, 96, 192, 384, 768)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True); e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)


def fit_logit_map(z, targets, ridge=1e-4):  # identical to 41_*/scripts/e40_common.fit_logit_map
    x = np.column_stack([z, np.ones_like(z)]); n, k = targets.shape
    def f(w):
        wm = w.reshape(2, k); p = softmax(x @ wm)
        ll = -(targets * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1).mean()
        g = x.T @ (p - targets) / n; g[0] += ridge * wm[0]
        return float(ll + 0.5 * ridge * (wm[0] ** 2).sum()), g.ravel()
    return minimize(f, np.zeros(2 * k), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x.reshape(2, k)


def main(path, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    data = load(path)
    q, names = build(data, "native")
    means = q @ R
    qlog, qrps = loss_by_grade(q[data.train])
    local = np.full(len(q), -1, int); local[data.train] = np.arange(len(data.train))
    test = data.test
    counts = data.counts[test]; h = counts / counts.sum(1, keepdims=True); y = h @ R
    system = data.system[test]; docs = data.group[test]
    X = {"pearson": [], "calibrated_pearson": [], "continuous_calibrated_pearson": []}
    sel = []
    PS = {"pearson": [], "calibrated_pearson": [], "continuous_calibrated_pearson": []}  # test (normalized RPS, CE) per draw
    for rep in range(REPEATS):
        order, picked = draw(data, rep, DEPTH)
        train_rows = local[order]
        sums = np.zeros((5, len(names))); sum_y = sq_y = 0.0; prev = 0
        for b in BUDGETS:
            n = b // DEPTH; ix = order[prev:n]; loc = train_rows[prev:n]; normalized = picked[prev:n] / DEPTH
            yy = normalized @ R; m = means[ix]
            for k in range(5):
                nz = np.where(normalized[:, k] > 0)[0]
                if len(nz):
                    w = normalized[nz, k]
                    sums[0] += np.einsum("i,ih->h", w, qlog[loc[nz], k]); sums[1] += np.einsum("i,ih->h", w, qrps[loc[nz], k])
            sums[2] += m.sum(0); sums[3] += np.einsum("ih,ih->h", m, m); sums[4] += np.einsum("i,ih->h", yy, m)
            sum_y += yy.sum(); sq_y += yy @ yy; prev = n
        n = BUDGET // DEPTH
        pearson, undefined = pearson_from_totals(*sums[2:], sum_y, sq_y, n)
        hist_count = picked[:n].sum(0); prior = (hist_count + .5) / (BUDGET + 2.5)
        bins = bin_indices(means[data.train, pearson], means[:, pearson])
        posterior = bin_distribution(bins, order[:n], picked[:n], prior)
        X["pearson"].append(q[test, pearson] @ R)
        X["calibrated_pearson"].append(posterior[bins[test]] @ R)
        # continuous diagnostic calibrator on the same training votes
        mtr = means[order[:n], pearson]; mu, sd = mtr.mean(), mtr.std() or 1.0
        W = fit_logit_map((mtr - mu) / sd, picked[:n].astype(float))
        X["continuous_calibrated_pearson"].append(softmax(np.column_stack([(means[test, pearson] - mu) / sd, np.ones(len(test))]) @ W) @ R)
        sel.append(names[pearson])
        for key, dist in (("pearson", q[test, pearson]), ("calibrated_pearson", posterior[bins[test]]),
                          ("continuous_calibrated_pearson", softmax(np.column_stack([(means[test, pearson] - mu) / sd, np.ones(len(test))]) @ W))):
            sc = per_item(np.clip(dist, 1e-12, 1), h)
            PS[key].append([float(sc[:, 0].mean()), float(sc[:, 1].mean())])
    X = {k: np.array(v) for k, v in X.items()}  # (draws, test items) expected grades
    sysnames, sidx = np.unique(system, return_inverse=True)
    docnames, didx = np.unique(docs, return_inverse=True)
    S, D = len(sysnames), len(docnames)
    # cell means: (draw, system, doc) and human (system, doc)
    def cells(v):
        c = np.zeros(v.shape[:-1] + (S, D)); cnt = np.zeros((S, D))
        np.add.at(cnt, (sidx, didx), 1)
        for t in range(v.shape[-1]):
            pass
        flat = sidx * D + didx
        out_ = np.zeros(v.shape[:-1] + (S * D,))
        for j in range(S * D):
            out_[..., j] = v[..., flat == j].mean(axis=-1)
        return out_.reshape(v.shape[:-1] + (S, D))
    Hc = cells(y[None, :])[0]
    C = {k: cells(v) for k, v in X.items()}

    def spearman_sys(pred_sys, hum_sys):  # same as diagnostics.system_scores: Spearman of tie-level ranks
        return np.corrcoef(rankdata(tie_levels(pred_sys)), rankdata(tie_levels(hum_sys)))[0, 1]

    def stat(doc_w, sys_list):
        hs = (Hc * doc_w).sum(1) / doc_w.sum()
        res = {}
        for k, c in C.items():
            ps = (c * doc_w).sum(-1) / doc_w.sum()  # (draw, S)
            res[k] = np.array([spearman_sys(ps[r][sys_list], hs[sys_list]) for r in range(REPEATS)])
        return res

    full = stat(np.ones(D), np.arange(S))
    # check against Experiment 23's own function
    chk = {k: float(np.mean([system_scores(X[k][r], y, system)["system_spearman"] for r in range(REPEATS)])) for k in X}
    rng = np.random.default_rng(SEED)
    boot = {"documents": [], "systems": [], "both": []}
    for unit in boot:
        for _ in range(BOOT):
            w = np.bincount(rng.integers(0, D, D), minlength=D).astype(float) if unit in ("documents", "both") else np.ones(D)
            sl = rng.integers(0, S, S) if unit in ("systems", "both") else np.arange(S)
            r = stat(w, sl)
            boot[unit].append([np.nanmean(r[k]) for k in ("pearson", "calibrated_pearson", "continuous_calibrated_pearson")])
        boot[unit] = np.array(boot[unit])
    jack = []
    for s in range(S):
        keep = np.array([i for i in range(S) if i != s]); r = stat(np.ones(D), keep)
        jack.append([r["pearson"].mean(), r["calibrated_pearson"].mean(), r["continuous_calibrated_pearson"].mean()])
    jack = np.array(jack)

    def summarize(a, b):  # a, b arrays of resampled means
        d = a - b; d = d[np.isfinite(d)]
        p = min(1.0, 2 * min((1 + (d <= 0).sum()) / (len(d) + 1), (1 + (d >= 0).sum()) / (len(d) + 1)))
        return {"lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "p_two_sided": p, "n": int(len(d))}

    res = {"reproduction": {"exp23_reported": {"pearson": 0.8765050449206083, "calibrated_pearson": 0.8348534698350497},
                            "this_run_exp23_function": chk,
                            "this_run_cell_means": {k: float(v.mean()) for k, v in full.items()}},
           "selected_heads": {s: sel.count(s) for s in set(sel)},
           "n_systems": S, "n_test_documents": D, "n_test_items": int(len(test)), "draws": REPEATS, "boot": BOOT}
    dd = full["pearson"] - full["calibrated_pearson"]; dc = full["pearson"] - full["continuous_calibrated_pearson"]
    res["delta_pearson_minus_calibrated"] = {"mean": float(dd.mean()), "draw_sd": float(dd.std(ddof=1)),
        "draw_p2_5": float(np.percentile(dd, 2.5)), "draw_p97_5": float(np.percentile(dd, 97.5)),
        "documents": summarize(boot["documents"][:, 0], boot["documents"][:, 1]),
        "systems": summarize(boot["systems"][:, 0], boot["systems"][:, 1]),
        "both": summarize(boot["both"][:, 0], boot["both"][:, 1]),
        "jackknife_systems": {"min": float((jack[:, 0] - jack[:, 1]).min()), "max": float((jack[:, 0] - jack[:, 1]).max()),
                              "se": float(np.sqrt((S - 1) / S * (((jack[:, 0] - jack[:, 1]) - (jack[:, 0] - jack[:, 1]).mean()) ** 2).sum()))}}
    res["delta_pearson_minus_continuous_calibrated"] = {"mean": float(dc.mean()),
        "documents": summarize(boot["documents"][:, 0], boot["documents"][:, 2]),
        "systems": summarize(boot["systems"][:, 0], boot["systems"][:, 2]), "both": summarize(boot["both"][:, 0], boot["both"][:, 2])}
    res["level_intervals"] = {k: {u: [float(np.nanpercentile(boot[u][:, j], 2.5)), float(np.nanpercentile(boot[u][:, j], 97.5))] for u in boot}
                              for j, k in enumerate(("pearson", "calibrated_pearson", "continuous_calibrated_pearson"))}
    res["level_sd_systems_bootstrap"] = {k: float(np.nanstd(boot["systems"][:, j])) for j, k in enumerate(("pearson", "calibrated_pearson", "continuous_calibrated_pearson"))}
    # diagnostics: distinct values, ties, pair accuracy, Kendall
    hs = Hc.mean(1)
    diag = {"human_system_means": dict(zip(sysnames.tolist(), hs.round(4).tolist())),
            "human_tied_system_pairs": int(sum(abs(hs[i] - hs[j]) < 1e-12 for i in range(S) for j in range(i + 1, S))),
            "human_min_gap_between_system_means": float(np.min(np.diff(np.sort(hs))))}
    for k, v in X.items():
        ps = C[k].mean(-1)
        diag[k] = {"distinct_item_expected_scores_median_over_draws": float(np.median([len(np.unique(v[r].round(10))) for r in range(REPEATS)])),
                   "tied_system_pairs_mean": float(np.mean([sum(abs(ps[r, i] - ps[r, j]) < 1e-12 for i in range(S) for j in range(i + 1, S)) for r in range(REPEATS)])),
                   "system_pair_accuracy_mean": float(np.mean([system_scores(v[r], y, system)["system_pair_accuracy"] for r in range(REPEATS)])),
                   "kendall_tau_b_mean": float(np.mean([kendalltau(ps[r], hs).statistic for r in range(REPEATS)])),
                   "item_level_pearson_mean": float(np.mean([np.corrcoef(v[r], y)[0, 1] for r in range(REPEATS)]))}
    res["diagnostics"] = diag
    res["test_proper_scores_mean_over_draws"] = {k: {"normalized_rps": float(np.mean([a for a, _ in v])), "ce_nats": float(np.mean([b for _, b in v]))} for k, v in PS.items()}
    (out / "system_level_drop.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out / "per_draw_system_spearman.npz", **{k: v for k, v in full.items()})
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
