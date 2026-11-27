#!/usr/bin/env python3
"""Study A: proper-score vs D-study allocation of n_items vs n_raters.

CPU only. Writes results/study_a_summary.json, study_a_replications.csv, study_a_meta.json.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
MASTER = 20261126
N_REPS = 150
N_PILOT_ITEMS = 80
N_PILOT_RATERS = 16
N_FOLDS = 5
N_INNER = 8
N_I_GRID = (16, 32, 64)
N_R_GRID = (2, 4, 8, 16)
TOL = 1e-12
CE_FLOOR = 1e-6
DATASET_ORDER = ["chaosnli_snli", "chaosnli_mnli", "chaosnli_alphanli", "dices350", "dices990"]
DATASET_K = {"chaosnli_snli": 0, "chaosnli_mnli": 1, "chaosnli_alphanli": 2, "dices350": 3, "dices990": 4}

COEFFICIENTS = sorted(
    itertools.product((0.5, 1.0, 2.0), (-1, 0, 1), (-1, 0, 1)),
    key=lambda v: (np.log2(v[0]) ** 2 + v[1] ** 2 + v[2] ** 2, v),
)
DESIGNS = [(ni, nr) for ni in N_I_GRID for nr in N_R_GRID]  # 12 designs
DESIGN_ID = {d: i for i, d in enumerate(DESIGNS)}
BUDGETS = sorted({ni * nr for ni, nr in DESIGNS})


def rng_ss(*keys):
    return np.random.default_rng(np.random.SeedSequence([int(x) for x in keys]))


def transforms(p):
    K = p.shape[-1]
    p = (1 - K * CE_FLOOR) * p + CE_FLOOR
    t, b, c = np.asarray(COEFFICIENTS).T
    u = (np.arange(K) - (K - 1) / 2) / ((K - 1) / 2)
    logits = np.log(p)[:, None, :] * t[None, :, None] + b[None, :, None] * u + c[None, :, None] * u ** 2
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def first_min(loss):
    return (loss <= loss.min(-1, keepdims=True) + TOL).argmax(-1)


def rps_tvd_ce(q, h):
    """q,h [..., K] -> rps, ce, tvd."""
    K = q.shape[-1]
    f = q.cumsum(-1)[..., :-1]
    g = h.cumsum(-1)[..., :-1]
    rps = (f * f + g * (1 - 2 * f)).mean(-1)
    # floor already in q from transforms; still guard zeros
    q_safe = np.clip(q, CE_FLOOR, 1.0)
    q_safe = q_safe / q_safe.sum(-1, keepdims=True)
    ce = -(h * np.log(q_safe)).sum(-1)
    tvd = np.abs(h - q).sum(-1) / 2
    return rps, ce, tvd


def ce_fit_and_score(Q, train_idx, train_counts, test_idx, test_h):
    """Q [N,27,K]; train_counts [n,K]; returns mean rps,ce,tvd over test items and the transform index."""
    logq = np.log(Q[train_idx])  # [n,27,K]
    ce_loss = -(train_counts[:, None, :] * logq).sum(-1).sum(0)  # [27]
    ix = int(first_min(ce_loss[None, :])[0])
    q = Q[test_idx, ix, :]
    rps, ce, tvd = rps_tvd_ce(q, test_h)
    return float(rps.mean()), float(ce.mean()), float(tvd.mean()), ix



ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)  # Amendment A1, before the 150-rep run


def ce_mix_fit_and_score(Q, train_idx, train_counts, test_idx, test_h):
    """Amendment A1: CE over 27 transforms times alpha in ALPHAS, q=(1-a)*q_tr + a*pi_hat."""
    K = train_counts.shape[-1]
    tot = train_counts.sum(0)
    pi = (tot + 0.5) / (tot.sum() + 0.5 * K)
    best = (np.inf, 0, 0.0)
    for a in ALPHAS:
        if a == 0.0:
            qm = Q[train_idx]
        else:
            qm = (1.0 - a) * Q[train_idx] + a * pi
            qm = np.clip(qm, CE_FLOOR, 1.0)
            qm = qm / qm.sum(-1, keepdims=True)
        loss = -(train_counts[:, None, :] * np.log(qm)).sum(-1).sum(0)
        ix = int(first_min(loss[None, :])[0])
        val = float(loss[ix])
        if val < best[0] - TOL or (abs(val - best[0]) <= TOL and (ix, a) < (best[1], best[2])):
            best = (val, ix, a)
    _, ix, a = best
    q = (1.0 - a) * Q[test_idx, ix, :] + a * pi
    q = np.clip(q, CE_FLOOR, 1.0)
    q = q / q.sum(-1, keepdims=True)
    rps, ce, tvd = rps_tvd_ce(q, test_h)
    return float(rps.mean()), float(ce.mean()), float(tvd.mean()), ix, float(a)


def sample_counts(ratings, n_avail, idx, n_r, rng, K):
    """Sample n_r ratings without replacement for each item in idx. Returns counts [n,K] and means / heldout means."""
    n = len(idx)
    counts = np.zeros((n, K), dtype=int)
    mean_s = np.zeros(n)
    mean_ho = np.zeros(n)
    ok = np.ones(n, dtype=bool)
    for t, i in enumerate(idx):
        avail = int(n_avail[i])
        if n_r > avail:
            ok[t] = False
            continue
        choose = rng.choice(avail, size=n_r, replace=False)
        ys = ratings[i, choose]
        counts[t] = np.bincount(ys, minlength=K)
        mean_s[t] = ys.mean()
        rest_idx = np.ones(avail, dtype=bool)
        rest_idx[choose] = False
        rest = ratings[i, :avail][rest_idx]
        mean_ho[t] = rest.mean() if rest.size else np.nan
    return counts, mean_s, mean_ho, ok


def anova_one_facet(X):
    """Balanced items x ratings ANOVA. X [n_i, n_r]. Returns v_u, v_e (clipped)."""
    X = np.asarray(X, float)
    n_i, n_r = X.shape
    g = X.mean()
    mu = X.mean(1)
    ma = X.mean(0)
    ss_u = n_r * ((mu - g) ** 2).sum()
    ss_e = ((X - mu[:, None] - ma[None, :] + g) ** 2).sum()
    df_u, df_e = n_i - 1, (n_i - 1) * (n_r - 1)
    ms_u = ss_u / df_u
    ms_e = ss_e / df_e
    v_u = (ms_u - ms_e) / n_r
    v_e = ms_e
    return {"v_u": max(0.0, float(v_u)), "v_e": max(1e-12, float(v_e)), "ms_u": float(ms_u), "ms_e": float(ms_e)}


def build_pilot_table(ratings, n_avail, items, n_r, rng):
    n = len(items)
    X = np.zeros((n, n_r), dtype=float)
    for t, i in enumerate(items):
        avail = int(n_avail[i])
        choose = rng.choice(avail, size=n_r, replace=False)
        X[t] = ratings[i, choose]
    return X


def tci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m = float(x.mean()) if x.size else float("nan")
    if x.size < 2:
        return {"mean": m, "ci95": [None, None], "se": None, "n": int(x.size)}
    se = float(x.std(ddof=1) / np.sqrt(x.size))
    t = float(stats.t.ppf(0.975, x.size - 1))
    return {"mean": m, "ci95": [m - t * se, m + t * se], "se": se, "n": int(x.size)}


def wilson(k, n):
    if n == 0:
        return {"proportion": None, "k": 0, "n": 0, "ci95": [None, None]}
    z, p = 1.959963984540054, k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"proportion": float(p), "k": int(k), "n": int(n), "ci95": [float(c - h), float(c + h)]}


def load_bundle():
    raw = np.load(ROOT / "data" / "processed" / "study_a_bundle.npz", allow_pickle=True)
    return raw["bundle"].item()


def designs_for_budget(B):
    return [d for d in DESIGNS if d[0] * d[1] == B]


def run_dataset(name, data, n_reps=N_REPS):
    k = DATASET_K[name]
    K = int(data["K"])
    ratings = data["ratings"]
    n_avail = data["n_avail"]
    counts_full = data["counts"]
    h_full = counts_full / counts_full.sum(1, keepdims=True)
    universe_mean = (ratings.clip(min=0) * (ratings >= 0)).sum(1) / n_avail  # item means; ratings uses -1 pad
    # fix universe mean properly
    universe_mean = np.array([ratings[i, : n_avail[i]].mean() for i in range(len(n_avail))])
    Q = transforms(data["p_mix"])
    test_idx = np.asarray(data["test_idx"], int)
    dev_idx = np.asarray(data["dev_idx"], int)
    test_h = h_full[test_idx]
    dev_universe = universe_mean[dev_idx].mean()

    # precompute list of designs and budget map
    rows = []
    t0 = time.time()
    for rep in range(n_reps):
        rng_items = rng_ss(MASTER, 2, k, rep)
        rng_rate = rng_ss(MASTER, 3, k, rep)
        pilot_items = rng_items.choice(dev_idx, size=N_PILOT_ITEMS, replace=False)
        # G-study table and components
        X = build_pilot_table(ratings, n_avail, pilot_items, N_PILOT_RATERS, rng_rate)
        vc = anova_one_facet(X)
        # 5-fold assignment of the 80 pilot items
        rng_fold = rng_ss(MASTER, 6, k, rep)
        fold = np.arange(N_PILOT_ITEMS) % N_FOLDS
        rng_fold.shuffle(fold)

        # proper-score validation RPS estimates
        val_rps = {d: [] for d in DESIGNS}
        val_rps_mix = {d: [] for d in DESIGNS}
        for fold_id in range(N_FOLDS):
            val_items = pilot_items[fold == fold_id]
            train_pool = pilot_items[fold != fold_id]  # 64 items
            val_h = h_full[val_items]
            for d_id, (n_i, n_r) in enumerate(DESIGNS):
                if n_i > len(train_pool):
                    continue
                for m in range(N_INNER):
                    rng = rng_ss(MASTER, 4, k, rep, fold_id, d_id, m)
                    tr = rng.choice(train_pool, size=n_i, replace=False)
                    cc, _, _, ok = sample_counts(ratings, n_avail, tr, n_r, rng, K)
                    if not ok.all():
                        continue
                    rps, _, _, _ = ce_fit_and_score(Q, tr, cc, val_items, val_h)
                    val_rps[(n_i, n_r)].append(rps)
                    rps_m, _, _, _, _ = ce_mix_fit_and_score(Q, tr, cc, val_items, val_h)
                    val_rps_mix[(n_i, n_r)].append(rps_m)

        mean_val = {d: (float(np.mean(v)) if v else np.inf) for d, v in val_rps.items()}
        mean_val_mix = {d: (float(np.mean(v)) if v else np.inf) for d, v in val_rps_mix.items()}

        # deploy every design once
        deploy = {}
        for d_id, (n_i, n_r) in enumerate(DESIGNS):
            rng = rng_ss(MASTER, 5, k, rep, d_id)
            tr = rng.choice(dev_idx, size=n_i, replace=False)
            cc, mean_s, mean_ho, ok = sample_counts(ratings, n_avail, tr, n_r, rng, K)
            assert ok.all()
            rps, ce, tvd, ix = ce_fit_and_score(Q, tr, cc, test_idx, test_h)
            rps_m, ce_m, tvd_m, ix_m, a_m = ce_mix_fit_and_score(Q, tr, cc, test_idx, test_h)
            item_mse = float(np.mean((mean_s - mean_ho) ** 2))
            grand_mse = float((mean_s.mean() - dev_universe) ** 2)
            deploy[(n_i, n_r)] = {
                "rps": rps, "ce": ce, "tvd": tvd, "transform": ix,
                "rps_mix": rps_m, "ce_mix": ce_m, "tvd_mix": tvd_m,
                "transform_mix": ix_m, "alpha_mix": a_m,
                "item_mse": item_mse, "grand_mse": grand_mse,
            }

        def pick_min(score_fn, designs, tie_items="more"):
            # score_fn(design) -> float, lower better
            best = None
            best_s = None
            for d in designs:
                s = score_fn(d)
                if best is None or s < best_s - TOL or (
                    abs(s - best_s) <= TOL and (
                        (tie_items == "more" and d[0] > best[0]) or
                        (tie_items == "more" and d[0] == best[0] and d[1] > best[1]) or
                        (tie_items == "more_raters" and d[1] > best[1])
                    )
                ):
                    best, best_s = d, s
            return best

        rec = {"dataset": name, "rep": rep, "v_u": vc["v_u"], "v_e": vc["v_e"]}
        for B in BUDGETS:
            cands = designs_for_budget(B)
            # U1: min v_e / n_r  -> max n_r; tie more items (should not happen)
            u1 = pick_min(lambda d: vc["v_e"] / d[1], cands, tie_items="more")
            # U2: min v_u/n_i + v_e/B -> max n_i if v_u>0
            u2 = pick_min(lambda d: vc["v_u"] / d[0] + vc["v_e"] / B, cands, tie_items="more")
            ps = pick_min(lambda d: mean_val[d], cands, tie_items="more")
            ps_mix = pick_min(lambda d: mean_val_mix[d], cands, tie_items="more")
            rps_oracle = pick_min(lambda d: deploy[d]["rps"], cands, tie_items="more")
            rps_mix_oracle = pick_min(lambda d: deploy[d]["rps_mix"], cands, tie_items="more")
            item_oracle = pick_min(lambda d: deploy[d]["item_mse"], cands, tie_items="more_raters")
            grand_oracle = pick_min(lambda d: deploy[d]["grand_mse"], cands, tie_items="more")
            rec[f"B{B}_u1"] = f"{u1[0]}x{u1[1]}"
            rec[f"B{B}_u2"] = f"{u2[0]}x{u2[1]}"
            rec[f"B{B}_ps"] = f"{ps[0]}x{ps[1]}"
            rec[f"B{B}_ps_mix"] = f"{ps_mix[0]}x{ps_mix[1]}"
            rec[f"B{B}_rps_oracle"] = f"{rps_oracle[0]}x{rps_oracle[1]}"
            rec[f"B{B}_rps_mix_oracle"] = f"{rps_mix_oracle[0]}x{rps_mix_oracle[1]}"
            rec[f"B{B}_item_oracle"] = f"{item_oracle[0]}x{item_oracle[1]}"
            rec[f"B{B}_grand_oracle"] = f"{grand_oracle[0]}x{grand_oracle[1]}"
            rec[f"B{B}_ps_val_rps"] = mean_val[ps]
            rec[f"B{B}_ps_mix_val_rps"] = mean_val_mix[ps_mix]
            for tag, d in (("u1", u1), ("u2", u2), ("ps", ps), ("ps_mix", ps_mix),
                           ("rps_oracle", rps_oracle), ("rps_mix_oracle", rps_mix_oracle),
                           ("item_oracle", item_oracle), ("grand_oracle", grand_oracle)):
                dep = deploy[d]
                rec[f"B{B}_{tag}_rps"] = dep["rps"]
                rec[f"B{B}_{tag}_ce"] = dep["ce"]
                rec[f"B{B}_{tag}_tvd"] = dep["tvd"]
                rec[f"B{B}_{tag}_rps_mix"] = dep["rps_mix"]
                rec[f"B{B}_{tag}_ce_mix"] = dep["ce_mix"]
                rec[f"B{B}_{tag}_tvd_mix"] = dep["tvd_mix"]
                rec[f"B{B}_{tag}_item_mse"] = dep["item_mse"]
                rec[f"B{B}_{tag}_grand_mse"] = dep["grand_mse"]
                rec[f"B{B}_{tag}_transform"] = dep["transform"]
                rec[f"B{B}_{tag}_alpha_mix"] = dep["alpha_mix"]
            # also store every candidate's deployment scores for this budget
            for d in cands:
                key = f"B{B}_cand_{d[0]}x{d[1]}"
                dep = deploy[d]
                rec[f"{key}_rps"] = dep["rps"]
                rec[f"{key}_ce"] = dep["ce"]
                rec[f"{key}_tvd"] = dep["tvd"]
                rec[f"{key}_rps_mix"] = dep["rps_mix"]
                rec[f"{key}_ce_mix"] = dep["ce_mix"]
                rec[f"{key}_tvd_mix"] = dep["tvd_mix"]
                rec[f"{key}_item_mse"] = dep["item_mse"]
                rec[f"{key}_grand_mse"] = dep["grand_mse"]
                rec[f"{key}_transform"] = dep["transform"]
        rows.append(rec)
        if (rep + 1) % 25 == 0 or rep == 0:
            print(f"  {name} rep {rep+1}/{n_reps}  elapsed {time.time()-t0:.1f}s", flush=True)
    return rows


def summarize(all_rows):
    import pandas as pd
    df = pd.DataFrame(all_rows)
    out = {"n_reps": N_REPS, "pilot": {"n_items": N_PILOT_ITEMS, "n_raters": N_PILOT_RATERS},
           "designs": [f"{a}x{b}" for a, b in DESIGNS], "budgets": BUDGETS, "datasets": {}}
    selectors = ["u1", "u2", "ps", "ps_mix", "rps_oracle", "rps_mix_oracle", "item_oracle", "grand_oracle"]
    for name in DATASET_ORDER:
        sub = df[df.dataset == name]
        ds = {"n": int(len(sub)), "components": {"v_u": tci(sub["v_u"]), "v_e": tci(sub["v_e"])}, "budgets": {}}
        for B in BUDGETS:
            cands = [f"{a}x{b}" for a, b in designs_for_budget(B)]
            cell = {"candidates": cands, "selection": {}, "metrics": {}, "paired": {}}
            for tag in selectors:
                col = f"B{B}_{tag}"
                counts = sub[col].value_counts().to_dict()
                mode = sub[col].mode().iloc[0] if len(sub) else None
                cell["selection"][tag] = {
                    "mode": mode,
                    "counts": {k: int(v) for k, v in counts.items()},
                    "agree_rps_oracle": wilson(int((sub[col] == sub[f"B{B}_rps_oracle"]).sum()), len(sub)),
                    "agree_item_oracle": wilson(int((sub[col] == sub[f"B{B}_item_oracle"]).sum()), len(sub)),
                    "agree_grand_oracle": wilson(int((sub[col] == sub[f"B{B}_grand_oracle"]).sum()), len(sub)),
                }
                for metric in ("rps", "ce", "tvd", "rps_mix", "ce_mix", "tvd_mix", "item_mse", "grand_mse"):
                    cell["metrics"].setdefault(metric, {})[tag] = tci(sub[f"B{B}_{tag}_{metric}"])
            # paired differences: u1 - ps, etc. positive means first is worse (for lower-better metrics)
            cell["primary_transform_tie_u1_u2"] = wilson(
                int((sub[f"B{B}_u1_transform"] == sub[f"B{B}_u2_transform"]).sum()), len(sub)
            )
            pairs = [("u1", "ps"), ("u2", "ps"), ("u1", "u2"),
                     ("u1", "ps_mix"), ("u2", "ps_mix"), ("ps", "ps_mix"),
                     ("u1", "rps_oracle"), ("ps", "rps_oracle"),
                     ("u1", "rps_mix_oracle"), ("ps_mix", "rps_mix_oracle"),
                     ("u1", "item_oracle"), ("u2", "grand_oracle")]
            for a, b in pairs:
                for metric in ("rps", "ce", "tvd", "rps_mix", "ce_mix", "tvd_mix", "item_mse", "grand_mse"):
                    diff = sub[f"B{B}_{a}_{metric}"].to_numpy() - sub[f"B{B}_{b}_{metric}"].to_numpy()
                    rec = tci(diff)
                    rec["ci_excludes_0"] = rec["ci95"][0] is not None and not (rec["ci95"][0] <= 0 <= rec["ci95"][1])
                    rec["first_beats_second"] = bool(rec["ci_excludes_0"] and rec["mean"] < 0)
                    rec["second_beats_first"] = bool(rec["ci_excludes_0"] and rec["mean"] > 0)
                    # p-value paired t
                    if len(diff) > 1 and np.isfinite(diff).all():
                        rec["paired_t_pvalue"] = float(stats.ttest_1samp(diff, 0.0).pvalue)
                    else:
                        rec["paired_t_pvalue"] = None
                    cell["paired"][f"{a}_minus_{b}_{metric}"] = rec
            # candidate-level mean scores
            cell["candidates_scores"] = {}
            for cand in cands:
                cell["candidates_scores"][cand] = {
                    m: tci(sub[f"B{B}_cand_{cand}_{m}"])
                    for m in ("rps", "ce", "tvd", "rps_mix", "ce_mix", "tvd_mix", "item_mse", "grand_mse")
                }
            ds["budgets"][str(B)] = cell
        out["datasets"][name] = ds
    return out, df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    t0 = time.time()
    rows = []
    for name in DATASET_ORDER:
        print(f"== {name} ==", flush=True)
        rows.extend(run_dataset(name, bundle[name], N_REPS))
    summary, df = summarize(rows)
    summary["runtime_seconds"] = round(time.time() - t0, 2)
    summary["software"] = {
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "n_reps": N_REPS,
        "n_inner": N_INNER,
        "n_folds": N_FOLDS,
        "master_seed": MASTER,
    }
    df.to_csv(OUT / "study_a_replications.csv", index=False)
    (OUT / "study_a_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    meta = {"n_rows": int(len(df)), "runtime_seconds": summary["runtime_seconds"]}
    (OUT / "study_a_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("done study A in", summary["runtime_seconds"], "s")


if __name__ == "__main__":
    main()
