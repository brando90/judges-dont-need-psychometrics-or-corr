"""Held-out CCA and proper-score audit on real multi-annotator ratings.

Reads SummEval, ChaosNLI, and DICES-990 through experiment 36's loaders.
Reads VeriBench functional semantic correctness from the experiment 09 CSV export
in trustworthy-theorem-judges when that export is on disk. Writes only aggregates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ["SANMI36_DATA"] = str(ROOT / "data_cache")
EXP36 = ROOT.parent / "36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026"
sys.path.insert(0, str(EXP36))
sys.path.insert(0, str(ROOT))

import data_io  # noqa: E402
from cca_tools import (  # noqa: E402
    TOL_CCA,
    brier_one,
    canonical_correlations,
    ce_one,
    correlations_close,
    fit_ce_softmax,
    item_cluster_bootstrap,
    pearson,
    percentile_interval,
    project_train_basis,
    rps_one,
    softmax,
    spearman,
    tvd_one,
)

SEED = 37
B = 1000
HASH_DIM = 64
N_COMP = 8
L2 = 1e-2
MAXITER = 200
FLOOR = 1e-6
TOKEN = re.compile(r"[a-z0-9]+")
TTJ_RATINGS = Path(
    "/tmp/ttj/experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026/ratings"
)


def hash_features(texts: list[str], dim: int = HASH_DIM) -> np.ndarray:
    x = np.zeros((len(texts), dim), dtype=np.float64)
    for i, text in enumerate(texts):
        toks = TOKEN.findall(str(text).lower())
        grams = toks + [a + "_" + b for a, b in zip(toks, toks[1:])] or ["empty"]
        for gram in grams:
            digest = hashlib.md5(gram.encode()).hexdigest()
            idx = int(digest[:8], 16) % dim
            sign = 1.0 if int(digest[8], 16) % 2 == 0 else -1.0
            x[i, idx] += sign
        norm = np.linalg.norm(x[i])
        if norm > 0:
            x[i] /= norm
    return x


def load_veribench():
    """Numeric FSC scores from the public experiment 09 CSV export. Names are not returned."""
    folder = Path(os.environ["VERIBENCH_FSC_DIR"]) if os.environ.get("VERIBENCH_FSC_DIR") else TTJ_RATINGS
    status = {
        "looked_at": str(folder),
        "r_v1_file_found": False,
        "note": (
            "Experiment 36 described CertJudge / VeriBench FSC R_v1 as about 343 items and about 5 experts. "
            "No file with that name is in this repository or in the trustworthy-theorem-judges tree "
            "fetched for this run. The scores below are the numeric final_score_0_5 cells in the "
            "experiment 09 CSV export, not a reconstructed 343-item table."
        ),
    }
    if not folder.is_dir():
        status["loaded"] = False
        status["error"] = f"directory missing: {folder}"
        return None, status
    chosen = {}
    n_numeric_rows = 0
    n_duplicate_dropped = 0
    rank = {"native_main": 3, "setN": 2, "warmup": 1}
    for path in sorted(folder.glob("*.csv")):
        packet = "other"
        if "native_main" in path.name:
            packet = "native_main"
        elif "setN" in path.name:
            packet = "setN"
        elif "warmup" in path.name:
            packet = "warmup"
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                raw = (row.get("final_score_0_5") or "").strip()
                if raw not in {"0", "1", "2", "3", "4", "5"}:
                    continue
                n_numeric_rows += 1
                item = (row.get("item_id") or "").strip()
                rater = (row.get("rater_id") or "").strip()
                if not item or not rater:
                    continue
                key = (rater, item)
                rec = {
                    "score": int(raw),
                    "task": (row.get("task_folder") or "").strip() or item,
                    "text": f"{(row.get('task_folder') or '').strip()} {(row.get('candidate_file') or '').strip()}".strip(),
                    "rank": rank.get(packet, 0),
                }
                if key in chosen:
                    n_duplicate_dropped += 1
                    if rec["rank"] > chosen[key]["rank"]:
                        chosen[key] = rec
                else:
                    chosen[key] = rec
    bucket = {}
    for (rater, item), rec in chosen.items():
        slot = bucket.setdefault(item, {"text": rec["text"], "group": rec["task"], "y": [], "rater": []})
        slot["y"].append(rec["score"])  # already the class index on 0..5
        slot["rater"].append(rater)
    data = data_io._pack(
        "veribench_fsc_exp09",
        "ordinal",
        [0, 1, 2, 3, 4, 5],
        bucket,
        "VeriBench FSC final_score_0_5 from trustworthy-theorem-judges experiment 09. "
        "Not the undescribed R_v1 343-item table.",
    )
    ks = np.array([r.size for r in data.ratings]) if data.ratings else np.array([])
    status.update({
        "loaded": True,
        "n_numeric_rows_before_dedup": n_numeric_rows,
        "n_duplicate_rater_item_dropped": n_duplicate_dropped,
        "n_items": data.n_items,
        "n_ratings": int(ks.sum()) if ks.size else 0,
        "ratings_per_item_min": int(ks.min()) if ks.size else 0,
        "ratings_per_item_median": float(np.median(ks)) if ks.size else 0,
        "ratings_per_item_max": int(ks.max()) if ks.size else 0,
        "n_raters": len({r for rs in data.rater_ids for r in rs}),
        "items_with_at_least_3": int(np.sum(ks >= 3)) if ks.size else 0,
        "items_with_at_least_4": int(np.sum(ks >= 4)) if ks.size else 0,
    })
    return data, status


def _as_float(v):
    if isinstance(v, str) or v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def fmt(x, lo=None, hi=None, digits=4) -> str:
    xv = _as_float(x)
    if xv is None:
        return "infinite" if x == "inf" or x == float("inf") else "undefined"
    lv, hv = _as_float(lo), _as_float(hi)
    if lv is None or hv is None:
        return f"{xv:.{digits}f}"
    return f"{xv:.{digits}f} [{lv:.{digits}f}, {hv:.{digits}f}]"


def boot_p(samples: np.ndarray) -> str:
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return "n/a"
    p = 2.0 * min(float(np.mean(s <= 0.0)), float(np.mean(s >= 0.0)))
    if p == 0.0:
        return f"< {1.0 / s.size:.4f}"
    return f"{p:.4f}"


def ci_of(samples: np.ndarray) -> dict:
    lo, hi, n = percentile_interval(samples)
    return {"lo": lo, "hi": hi, "n_finite": n, "p_two_sided_percentile": boot_p(samples)}


def pmf_counts(idx: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(np.asarray(idx, dtype=int), minlength=k).astype(np.float64)


def add_one(idx: np.ndarray, k: int) -> np.ndarray:
    c = pmf_counts(idx, k) + 1.0
    return c / c.sum()


def point_floor(index: int, k: int, floor: float = FLOOR) -> np.ndarray:
    q = np.full(k, floor, dtype=np.float64)
    q[index] = 1.0 - floor * (k - 1)
    return q


def item_losses(q: np.ndarray, ratings: np.ndarray) -> dict:
    ce = br = rp = 0.0
    for y in np.asarray(ratings, dtype=int):
        ce += ce_one(q, int(y))
        br += brier_one(q, int(y))
        rp += rps_one(q, int(y))
    n = float(len(ratings))
    p_hat = pmf_counts(ratings, q.shape[0])
    p_hat = p_hat / p_hat.sum()
    return {"ce": ce / n, "brier": br / n, "rps": rp / n, "sqrt_rps": float(np.sqrt(rp / n)), "tvd": tvd_one(q, p_hat)}


def mean_metric(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows]
    if any(not np.isfinite(v) for v in vals):
        return float("inf")
    return float(np.mean(vals))


def boot_mean(values: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    def stat(idx):
        take = values[idx]
        if np.any(~np.isfinite(take)):
            return float("inf")
        return float(np.mean(take))

    return item_cluster_bootstrap(stat, values.size, B, seed)


def pearson_from_sum_rows(rows: list[dict]) -> float:
    n = sum(r["n"] for r in rows)
    if n < 2:
        return float("nan")
    sx = sum(r["sx"] for r in rows)
    sy = sum(r["sy"] for r in rows)
    sx2 = sum(r["sx2"] for r in rows)
    sy2 = sum(r["sy2"] for r in rows)
    sxy = sum(r["sxy"] for r in rows)
    mx, my = sx / n, sy / n
    vx, vy = sx2 / n - mx * mx, sy2 / n - my * my
    cov = sxy / n - mx * my
    if vx <= 1e-15 or vy <= 1e-15:
        return float("nan")
    return float(cov / np.sqrt(vx * vy))


def pack_pairs(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return {"n": int(x.size), "sx": float(x.sum()), "sy": float(y.sum()),
            "sx2": float(np.dot(x, x)), "sy2": float(np.dot(y, y)), "sxy": float(np.dot(x, y))}


def moment_block(data, indices: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pairs_x, pairs_y = [], []
    sum_rows = []
    mean_x, mean_y = [], []
    for i in indices:
        y_idx = np.asarray(data.ratings[i], dtype=int)
        if y_idx.size < 2:
            continue
        order = rng.permutation(y_idx.size)
        a = float(data.class_values[y_idx[order[0]]])
        b = float(data.class_values[y_idx[order[1]]])
        pairs_x.append(a)
        pairs_y.append(b)
        sum_rows.append(pack_pairs(np.array([a]), np.array([b])))
        if y_idx.size >= 4:
            h = y_idx.size // 2
            left = data.class_values[y_idx[order[:h]]]
            right = data.class_values[y_idx[order[h:]]]
            mean_x.append(float(np.mean(left)))
            mean_y.append(float(np.mean(right)))
    x = np.asarray(pairs_x, dtype=float)
    y = np.asarray(pairs_y, dtype=float)
    out = {
        "n_items_paired": int(x.size),
        "pearson_one_pair": pearson(x, y),
        "spearman_one_pair": spearman(x, y),
        "n_items_half_means": int(len(mean_x)),
    }
    if x.size >= 5 and np.unique(x).size >= 2:
        def r2(yhat):
            tot = float(np.sum((y - y.mean()) ** 2))
            if tot <= 0:
                return float("nan")
            return float(1.0 - np.sum((y - yhat) ** 2) / tot)

        a1 = np.column_stack([np.ones(x.size), x])
        c1, *_ = np.linalg.lstsq(a1, y, rcond=None)
        a2 = np.column_stack([a1, x ** 2])
        c2, *_ = np.linalg.lstsq(a2, y, rcond=None)
        out["r2_linear"] = r2(a1 @ c1)
        out["r2_quadratic_gain"] = r2(a2 @ c2) - out["r2_linear"]
    else:
        out["r2_linear"] = float("nan")
        out["r2_quadratic_gain"] = float("nan")
    if mean_x:
        out["pearson_half_means"] = pearson(np.asarray(mean_x), np.asarray(mean_y))
    else:
        out["pearson_half_means"] = float("nan")

    def boot_pearson(idx):
        # idx indexes the paired items, which are exactly sum_rows in order
        return pearson_from_sum_rows([sum_rows[i] for i in idx])

    if sum_rows:
        draws = item_cluster_bootstrap(boot_pearson, len(sum_rows), B, seed)
        out["pearson_one_pair_ci"] = ci_of(draws)
        # spearman bootstrap rebuilds the pairs
        def boot_sp(idx):
            return spearman(x[idx], y[idx])
        out["spearman_one_pair_ci"] = ci_of(item_cluster_bootstrap(boot_sp, x.size, B, seed + 1))
    out["point"] = float(out["pearson_one_pair"]) if np.isfinite(out["pearson_one_pair"]) else None
    return out


def collapse_block(data, indices: np.ndarray, seed: int, mode: str) -> dict:
    rng = np.random.default_rng(seed)
    k = data.n_classes
    spread_sums, coll_sums = [], []
    brier_pmf, brier_pt, ce_pmf, ce_pt, rps_pmf, rps_pt = [], [], [], [], [], []
    # one-hot matrices built after the loop would be large; store per-item blocks for the top CCA on the full set
    oh_x, oh_y, pmf_x = [], [], []
    used = 0
    for i in indices:
        idx = np.asarray(data.ratings[i], dtype=int)
        if mode == "half" and idx.size < 4:
            continue
        if mode == "loro" and idx.size < 3:
            continue
        order = rng.permutation(idx.size)
        if mode == "half":
            h = idx.size // 2
            a, b = idx[order[:h]], idx[order[h:]]
        else:
            b, a = idx[order[:1]], idx[order[1:]]
        if a.size < 2 or b.size < 1:
            continue
        n = min(a.size, b.size) if mode == "half" else a.size
        if mode == "half":
            aa, bb = a[:n], b[:n]
        else:
            aa, bb = a, np.repeat(b, a.size)
        va, vb = data.class_values[aa], data.class_values[bb]
        spread_sums.append(pack_pairs(va, vb))
        mu = float(np.mean(data.class_values[a]))
        coll_sums.append(pack_pairs(np.full(va.shape, mu), vb))
        q_pmf = add_one(a, k)
        nearest = int(np.argmin(np.abs(data.class_values - mu)))
        q_pt = point_floor(nearest, k)
        loss_pmf = item_losses(q_pmf, b)
        loss_pt = item_losses(q_pt, b)
        brier_pmf.append(loss_pmf["brier"])
        brier_pt.append(loss_pt["brier"])
        ce_pmf.append(loss_pmf["ce"])
        ce_pt.append(loss_pt["ce"])
        rps_pmf.append(loss_pmf["rps"])
        rps_pt.append(loss_pt["rps"])
        # cap stored one-hots: one row per item (the PMF, and one sampled one-hot) for CCA
        e = np.zeros(k)
        e[int(aa[0])] = 1.0
        oh_x.append(e)
        e2 = np.zeros(k)
        e2[int(bb[0])] = 1.0
        oh_y.append(e2)
        pmf_x.append(q_pmf)
        used += 1
    if used < 5:
        return {"mode": mode, "n_items": used, "available": False}
    r_spread = pearson_from_sum_rows(spread_sums)
    r_coll = pearson_from_sum_rows(coll_sums)
    rho_spread = canonical_correlations(np.vstack(oh_x), np.vstack(oh_y))
    rho_pmf = canonical_correlations(np.vstack(pmf_x), np.vstack(oh_y))

    def boot_diff_corr(idx):
        return pearson_from_sum_rows([coll_sums[i] for i in idx]) - pearson_from_sum_rows([spread_sums[i] for i in idx])

    diff_draws = item_cluster_bootstrap(boot_diff_corr, used, B, seed)
    brier_diff = np.asarray(brier_pt) - np.asarray(brier_pmf)
    ce_diff = np.asarray(ce_pt) - np.asarray(ce_pmf)
    rps_diff = np.asarray(rps_pt) - np.asarray(rps_pmf)
    return {
        "mode": mode,
        "available": True,
        "n_items": used,
        "pearson_spread": r_spread,
        "pearson_collapsed_mean": r_coll,
        "pearson_collapsed_minus_spread": r_coll - r_spread,
        "pearson_diff_ci": ci_of(diff_draws),
        "cca_top_onehot_sample": float(rho_spread[0]) if rho_spread.size else float("nan"),
        "cca_top_pmf_vs_onehot": float(rho_pmf[0]) if rho_pmf.size else float("nan"),
        "brier_add_one_pmf": float(np.mean(brier_pmf)),
        "brier_point_mass": float(np.mean(brier_pt)),
        "brier_point_minus_pmf": float(np.mean(brier_diff)),
        "brier_diff_ci": ci_of(boot_mean(brier_diff, seed + 3)),
        "ce_add_one_pmf": float(np.mean(ce_pmf)),
        "ce_point_mass": float(np.mean(ce_pt)),
        "ce_point_minus_pmf": float(np.mean(ce_diff)),
        "ce_diff_ci": ci_of(boot_mean(ce_diff, seed + 4)),
        "rps_add_one_pmf": float(np.mean(rps_pmf)),
        "rps_point_mass": float(np.mean(rps_pt)),
        "sqrt_rps_add_one_pmf": float(np.sqrt(np.mean(rps_pmf))),
        "sqrt_rps_point_mass": float(np.sqrt(np.mean(rps_pt))),
        "rps_point_minus_pmf": float(np.mean(rps_diff)),
        "rps_diff_ci": ci_of(boot_mean(rps_diff, seed + 5)),
    }


def fit_and_attack(data, train, test) -> dict:
    k = data.n_classes
    x_all = hash_features(data.texts)
    x_train, (x_test,), n_comp = project_train_basis(x_all[train], [x_all[test]], N_COMP)
    targets = np.stack([pmf_counts(data.ratings[i], k) for i in train])
    targets = targets / targets.sum(axis=1, keepdims=True)
    _prob, weight, success, nit, fun = fit_ce_softmax(x_train, targets, L2, MAXITER)
    q = softmax(x_test @ weight)
    q = np.clip(q, 1e-12, 1.0)
    q = q / q.sum(axis=1, keepdims=True)
    human = np.stack([pmf_counts(data.ratings[i], k) / len(data.ratings[i]) for i in test])
    losses = [item_losses(q[row], data.ratings[test[row]]) for row in range(test.size)]
    panel = {}
    for key in ("ce", "brier", "rps", "sqrt_rps", "tvd"):
        vals = np.array([row[key] for row in losses])
        # sqrt_rps above is within-item sqrt. The reported sqrt(RPS) is the sqrt of the unbiased RPS.
        if key == "sqrt_rps":
            continue
        draws = boot_mean(vals, SEED)
        panel[key] = {"estimate": float(np.mean(vals)), "ci": ci_of(draws)}
    rps_vals = np.array([row["rps"] for row in losses])
    sqrt_draws = np.sqrt(np.maximum(boot_mean(rps_vals, SEED), 0.0))
    panel["sqrt_rps"] = {"estimate": float(np.sqrt(np.mean(rps_vals))), "ci": ci_of(sqrt_draws)}
    panel["uniform_ce"] = float(np.log(k))
    expected = q @ data.class_values
    human_mean = np.array([float(np.mean(data.class_values[data.ratings[i]])) for i in test])
    panel["pearson_expected_vs_item_mean"] = pearson(expected, human_mean)
    panel["spearman_expected_vs_item_mean"] = spearman(expected, human_mean)

    def boot_pear(idx):
        return pearson(expected[idx], human_mean[idx])

    panel["pearson_ci"] = ci_of(item_cluster_bootstrap(boot_pear, test.size, B, SEED))
    rho = canonical_correlations(q, human)
    panel["cca_top"] = float(rho[0]) if rho.size else float("nan")
    panel["cca_all"] = rho.tolist()

    rng = np.random.default_rng(SEED)
    k_idx = np.arange(k)
    maps = {
        "reversal": q[:, ::-1],
        "roll_permutation": q[:, np.roll(k_idx, 1)],
        "sign_flip_raw": -q,
    }
    gauss = rng.normal(size=(k, k))
    orth, _ = np.linalg.qr(gauss)
    if np.linalg.det(orth) < 0:
        orth[:, 0] *= -1
    shift = rng.normal(size=k)
    maps["orthogonal_mix"] = q @ orth.T + shift
    attacks = {}
    for name, q2 in maps.items():
        rho2 = canonical_correlations(q2, human)
        # proper scores: reversal and roll stay on the simplex; the others generally do not
        on_simplex = name in {"reversal", "roll_permutation"}
        if on_simplex:
            loss2 = [item_losses(q2[row], data.ratings[test[row]]) for row in range(test.size)]
            packed = {}
            for key in ("ce", "brier", "rps"):
                vals = np.array([row[key] for row in loss2])
                packed[key] = {"estimate": float(np.mean(vals)), "ci": ci_of(boot_mean(vals, SEED))}
            rps_v = np.array([row["rps"] for row in loss2])
            packed["sqrt_rps"] = {
                "estimate": float(np.sqrt(np.mean(rps_v))),
                "ci": ci_of(np.sqrt(np.maximum(boot_mean(rps_v, SEED), 0.0))),
            }
        else:
            algebraic = []
            n_nonpos = 0
            n_ratings = 0
            for row, item in enumerate(test):
                qq = q2[row]
                ratings = data.ratings[item]
                br = rp = 0.0
                for y in ratings:
                    n_ratings += 1
                    if qq[int(y)] <= 0.0:
                        n_nonpos += 1
                    br += brier_one(qq, int(y))
                    rp += rps_one(qq, int(y))
                algebraic.append((br / len(ratings), rp / len(ratings)))
            br_v = np.array([a[0] for a in algebraic])
            rp_v = np.array([a[1] for a in algebraic])
            packed = {
                "ce": {"estimate": float("inf"), "ci": {"lo": float("inf"), "hi": float("inf"), "n_finite": 0, "p_two_sided_percentile": "n/a"}},
                "fraction_ratings_with_nonpositive_mass": n_nonpos / max(n_ratings, 1),
                "brier_algebraic_raw_vector": {"estimate": float(np.mean(br_v)), "ci": ci_of(boot_mean(br_v, SEED))},
                "rps_algebraic_raw_vector": {"estimate": float(np.mean(rp_v)), "ci": ci_of(boot_mean(rp_v, SEED))},
                "sqrt_rps_algebraic_raw_vector": {"estimate": float(np.sqrt(np.mean(rp_v)))},
            }
        n = max(rho.size, rho2.size)
        aa, bb = np.zeros(n), np.zeros(n)
        if rho.size:
            aa[: rho.size] = rho
        if rho2.size:
            bb[: rho2.size] = rho2
        attacks[name] = {
            "max_abs_cca_difference": float(np.max(np.abs(aa - bb))) if n else float("nan"),
            "cca_identical_at_1e-8": bool(correlations_close(rho, rho2, TOL_CCA)),
            "cca_top": float(rho2[0]) if rho2.size else float("nan"),
            "scores": packed,
        }
    return {
        "n_comp": n_comp,
        "lbfgs_success": success,
        "lbfgs_iterations": nit,
        "train_objective": fun,
        "n_train": int(train.size),
        "n_test": int(test.size),
        "panel": panel,
        "attacks": attacks,
        "q": q,
        "human": human,
        "test_index": test,
    }


def multi_aspect(pool: str) -> dict:
    path = data_io.CACHE / "model_annotations.aligned.jsonl"
    data_io._fetch(data_io.SUMMEVAL_URL, path, data_io.SUMMEVAL_SHA)
    key = "expert_annotations" if pool == "expert" else "turker_annotations"
    rows = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            grades = row[key]
            mat = np.array([[int(g[a]) for a in data_io.ASPECTS] for g in grades], dtype=int)
            rows.append({
                "group": row["filepath"],
                "text": row["decoded"],
                "grades": mat,  # raters by aspects, values 1..5
            })
    groups = [r["group"] for r in rows]
    train, _val, test = data_io.split_by_group(groups, seed=SEED)
    texts = [r["text"] for r in rows]
    x_all = hash_features(texts)
    x_train, (x_test,), n_comp = project_train_basis(x_all[train], [x_all[test]], N_COMP)
    pred = []
    human_mean = []
    human_pmf = []
    q_concat = []
    successes = []
    for aspect in range(4):
        targets = []
        for i in train:
            idx = rows[i]["grades"][:, aspect] - 1
            targets.append(pmf_counts(idx, 5) / len(idx))
        targets = np.stack(targets)
        _p, weight, success, nit, _fun = fit_ce_softmax(x_train, targets, L2, MAXITER)
        successes.append({"aspect": data_io.ASPECTS[aspect], "success": success, "iterations": nit})
        q = softmax(x_test @ weight)
        q = np.clip(q, 1e-12, 1.0)
        q = q / q.sum(axis=1, keepdims=True)
        q_concat.append(q)
        values = np.arange(1, 6, dtype=float)
        pred.append(q @ values)
        means = []
        pmfs = []
        for i in test:
            grades = rows[i]["grades"][:, aspect] - 1
            means.append(float(np.mean(grades + 1)))
            pmfs.append(pmf_counts(grades, 5) / len(grades))
        human_mean.append(np.asarray(means))
        human_pmf.append(np.stack(pmfs))
    pred = np.column_stack(pred)
    human4 = np.column_stack(human_mean)
    q_all = np.concatenate(q_concat, axis=1)
    p_all = np.concatenate(human_pmf, axis=1)
    rho4 = canonical_correlations(pred, human4)
    rho20 = canonical_correlations(q_all, p_all)
    rng = np.random.default_rng(SEED)
    orth, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    if np.linalg.det(orth) < 0:
        orth[:, 0] *= -1
    mixed = pred @ orth.T + rng.normal(size=4)
    perm = pred[:, [1, 2, 3, 0]]
    flipped = -pred
    return {
        "pool": pool,
        "n_comp": n_comp,
        "n_train_summaries": int(train.size),
        "n_test_summaries": int(test.size),
        "fits": successes,
        "cca_top_4d_expected_grade": float(rho4[0]) if rho4.size else float("nan"),
        "cca_all_4d": rho4.tolist(),
        "cca_top_20d_pmf": float(rho20[0]) if rho20.size else float("nan"),
        "max_abs_cca_diff_aspect_roll": float(np.max(np.abs(
            _pad(rho4, canonical_correlations(perm, human4))
        ))),
        "max_abs_cca_diff_sign_flip": float(np.max(np.abs(
            _pad(rho4, canonical_correlations(flipped, human4))
        ))),
        "max_abs_cca_diff_orthogonal_mix": float(np.max(np.abs(
            _pad(rho4, canonical_correlations(mixed, human4))
        ))),
        "aspect_roll_identical": bool(correlations_close(rho4, canonical_correlations(perm, human4))),
        "sign_flip_identical": bool(correlations_close(rho4, canonical_correlations(flipped, human4))),
        "mix_identical": bool(correlations_close(rho4, canonical_correlations(mixed, human4))),
    }


def _pad(a, b):
    n = max(a.size, b.size, 1)
    aa, bb = np.zeros(n), np.zeros(n)
    aa[: a.size] = a
    bb[: b.size] = b
    return aa - bb


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items() if k not in {"q", "human", "test_index"}}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        if np.isfinite(obj):
            return float(obj)
        return "inf" if obj > 0 else "nan"
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def run() -> dict:
    (ROOT / "results.md").write_text(
        "**Status:** RUNNING\n\n**Last updated:** 2026-11-26\n\nLoading public rating files and the VeriBench FSC export.\n"
    )
    sets = [
        data_io.load_summeval("expert"),
        data_io.load_summeval("crowd"),
        data_io.load_chaos_3way(),
        data_io.load_chaos_alpha(),
        data_io._load_dices("990"),
    ]
    vb, vb_status = load_veribench()
    if vb is not None:
        sets.append(vb)
    report = {"veribench_fsc": vb_status, "datasets": {}}
    for data in sets:
        print(f"dataset {data.name} items={data.n_items}", flush=True)
        train, val, test = data_io.split_by_group(data.groups, seed=SEED)
        data_io.assert_no_prompt_leakage(data.groups, data.rater_ids, train, val, test)
        block = {
            "summary": data.summary(),
            "n_train_items": int(train.size),
            "n_val_items": int(val.size),
            "n_test_items": int(test.size),
            "moment_test": moment_block(data, test, SEED),
            "collapse_half_full": collapse_block(data, np.arange(data.n_items), SEED, "half"),
            "collapse_loro_full": collapse_block(data, np.arange(data.n_items), SEED, "loro"),
            "collapse_half_test": collapse_block(data, test, SEED + 11, "half"),
            "collapse_loro_test": collapse_block(data, test, SEED + 11, "loro"),
        }
        print(f"  fit {data.name}", flush=True)
        fitted = fit_and_attack(data, train, test)
        block["judge"] = clean(fitted)
        report["datasets"][data.name] = clean(block)
        print(f"  done {data.name} test CE {block['judge']['panel']['ce']['estimate']:.4f}", flush=True)
    print("multi-aspect CCA", flush=True)
    report["summeval_multiaspect"] = {
        "expert": multi_aspect("expert"),
        "crowd": multi_aspect("crowd"),
    }
    return clean(report)


def write_reports(report: dict) -> None:
    (ROOT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Canonical correlation versus proper scores", "",
             "Held-out test prompts unless a row says full collection. "
             "Percentile 95% intervals, item resample, B = 1000, seed 37. "
             "Lower is better for cross-entropy (CE), Brier, ranked probability score (RPS), and total variation distance (TVD).",
             ""]
    lines.append("| Dataset | Test items | CE | Brier | RPS | sqrt(RPS) | TVD plug-in | Pearson of E[grade] vs item mean |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, block in report["datasets"].items():
        p = block["judge"]["panel"]
        lines.append(
            f"| {name} | {block['n_test_items']} | {fmt(p['ce']['estimate'], p['ce']['ci']['lo'], p['ce']['ci']['hi'])} | "
            f"{fmt(p['brier']['estimate'], p['brier']['ci']['lo'], p['brier']['ci']['hi'])} | "
            f"{fmt(p['rps']['estimate'], p['rps']['ci']['lo'], p['rps']['ci']['hi'])} | "
            f"{fmt(p['sqrt_rps']['estimate'], p['sqrt_rps']['ci']['lo'], p['sqrt_rps']['ci']['hi'])} | "
            f"{fmt(p['tvd']['estimate'], p['tvd']['ci']['lo'], p['tvd']['ci']['hi'])} | "
            f"{fmt(p['pearson_expected_vs_item_mean'], p['pearson_ci']['lo'], p['pearson_ci']['hi'])} |"
        )
    (ROOT / "tables" / "heldout_proper_scores.md").write_text("\n".join(lines) + "\n")

    att = ["# Adversarial maps of the cross-entropy judge", "",
           "CCA is between the judge probability vector and the item's empirical human distribution, on held-out test items. "
           "A difference of 0 at 1e-8 means the canonical correlations matched to that tolerance. "
           "Reversal and the label roll stay on the simplex. Sign flip and the orthogonal mix do not, so their cross-entropy is infinite.",
           ""]
    att.append("| Dataset | Map | Max abs CCA difference | Identical at 1e-8 | CE | Brier | sqrt(RPS) |")
    att.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, block in report["datasets"].items():
        base = block["judge"]["panel"]
        att.append(
            f"| {name} | original | 0 | yes | {fmt(base['ce']['estimate'], base['ce']['ci']['lo'], base['ce']['ci']['hi'])} | "
            f"{fmt(base['brier']['estimate'], base['brier']['ci']['lo'], base['brier']['ci']['hi'])} | "
            f"{fmt(base['sqrt_rps']['estimate'], base['sqrt_rps']['ci']['lo'], base['sqrt_rps']['ci']['hi'])} |"
        )
        for map_name, attack in block["judge"]["attacks"].items():
            sc = attack["scores"]
            if "brier" in sc:
                ce_s = fmt(sc["ce"]["estimate"], sc["ce"]["ci"]["lo"], sc["ce"]["ci"]["hi"])
                br_s = fmt(sc["brier"]["estimate"], sc["brier"]["ci"]["lo"], sc["brier"]["ci"]["hi"])
                sr_s = fmt(sc["sqrt_rps"]["estimate"], sc["sqrt_rps"]["ci"]["lo"], sc["sqrt_rps"]["ci"]["hi"])
            else:
                ce_s = "infinite"
                br_s = fmt(sc["brier_algebraic_raw_vector"]["estimate"], sc["brier_algebraic_raw_vector"]["ci"]["lo"], sc["brier_algebraic_raw_vector"]["ci"]["hi"]) + " algebraic"
                sr_s = fmt(sc["sqrt_rps_algebraic_raw_vector"]["estimate"]) + " algebraic"
            att.append(
                f"| {name} | {map_name} | {attack['max_abs_cca_difference']:.3e} | {attack['cca_identical_at_1e-8']} | {ce_s} | {br_s} | {sr_s} |"
            )
    (ROOT / "tables" / "cca_invariance.md").write_text("\n".join(att) + "\n")

    col = ["# Within-prompt collapse", "",
           "Positive Pearson difference: collapsed conditional mean correlates more with the held-out ratings than the raw ratings do. "
           "Positive Brier, CE, or RPS difference: the floored point mass is worse than the add-one probability mass function. "
           "Half-split needs at least 4 ratings. Leave-one-rater-out needs at least 3.",
           ""]
    col.append("| Dataset | Design | Split | Items | Pearson collapsed minus spread | Brier point minus add-one | CE point minus add-one | RPS point minus add-one |")
    col.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, block in report["datasets"].items():
        for key, design, split in (
            ("collapse_half_test", "half-split", "held-out test"),
            ("collapse_loro_test", "leave-one-rater-out", "held-out test"),
            ("collapse_half_full", "half-split", "full collection"),
            ("collapse_loro_full", "leave-one-rater-out", "full collection"),
        ):
            c = block[key]
            if not c.get("available"):
                col.append(f"| {name} | {design} | {split} | {c.get('n_items', 0)} | unavailable | unavailable | unavailable | unavailable |")
                continue
            pc, bc, cc, rc = c["pearson_diff_ci"], c["brier_diff_ci"], c["ce_diff_ci"], c["rps_diff_ci"]
            col.append(
                f"| {name} | {design} | {split} | {c['n_items']} | "
                f"{fmt(c['pearson_collapsed_minus_spread'], pc['lo'], pc['hi'])} p-val={pc['p_two_sided_percentile']} | "
                f"{fmt(c['brier_point_minus_pmf'], bc['lo'], bc['hi'])} p-val={bc['p_two_sided_percentile']} | "
                f"{fmt(c['ce_point_minus_pmf'], cc['lo'], cc['hi'])} p-val={cc['p_two_sided_percentile']} | "
                f"{fmt(c['rps_point_minus_pmf'], rc['lo'], rc['hi'])} p-val={rc['p_two_sided_percentile']} |"
            )
    (ROOT / "tables" / "collapse.md").write_text("\n".join(col) + "\n")

    mom = ["# Held-out human-human association", "",
           "One seeded pair of ratings per test item. Pearson and Spearman use the numeric grade. "
           "Quadratic gain is the increase in in-sample R² from adding the square of the first rating, on the same test pairs. It is a description of the test pairs, not a trained model.",
           ""]
    mom.append("| Dataset | Test items with a pair | Pearson | Spearman | Linear R² | Quadratic R² gain | Pearson of half means |")
    mom.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, block in report["datasets"].items():
        m = block["moment_test"]
        pci, sci = m["pearson_one_pair_ci"], m["spearman_one_pair_ci"]
        mom.append(
            f"| {name} | {m['n_items_paired']} | {fmt(m['pearson_one_pair'], pci['lo'], pci['hi'])} | "
            f"{fmt(m['spearman_one_pair'], sci['lo'], sci['hi'])} | {fmt(m['r2_linear'])} | {fmt(m['r2_quadratic_gain'])} | "
            f"{fmt(m['pearson_half_means'])} (n={m['n_items_half_means']}) |"
        )
    (ROOT / "tables" / "moment_linear.md").write_text("\n".join(mom) + "\n")

    vb = report["veribench_fsc"]
    (ROOT / "tables" / "veribench_status.md").write_text(
        "\n".join([
            "# VeriBench FSC status",
            "",
            vb["note"],
            "",
            f"- Directory: `{vb['looked_at']}`",
            f"- Loaded: {vb.get('loaded')}",
            f"- R_v1 file found: {vb.get('r_v1_file_found')}",
            f"- Numeric rows before de-duplication: {vb.get('n_numeric_rows_before_dedup', 'n/a')}",
            f"- Duplicate rater-item rows dropped: {vb.get('n_duplicate_rater_item_dropped', 'n/a')}",
            f"- Items: {vb.get('n_items', 'n/a')}",
            f"- Ratings kept: {vb.get('n_ratings', 'n/a')}",
            f"- Ratings per item min / median / max: {vb.get('ratings_per_item_min', 'n/a')} / {vb.get('ratings_per_item_median', 'n/a')} / {vb.get('ratings_per_item_max', 'n/a')}",
            f"- Distinct rater_id values among kept scores: {vb.get('n_raters', 'n/a')}",
            f"- Items with at least 3 ratings: {vb.get('items_with_at_least_3', 'n/a')}",
            f"- Items with at least 4 ratings: {vb.get('items_with_at_least_4', 'n/a')}",
            "",
        ])
    )


if __name__ == "__main__":
    result = run()
    write_reports(result)
    print("wrote tables and results.json")
