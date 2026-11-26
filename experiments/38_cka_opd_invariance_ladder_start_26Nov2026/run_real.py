"""Held-out invariance ladder on real multi-annotator ratings.

The judge is a softmax linear head on a 256-dimensional signed token hash,
reduced to 24 train-only principal components plus an intercept. Weights
minimize mean cross-entropy against the train histogram, with L2 = 0.01 on
the non-intercept weights, from the zero start, L-BFGS, at most 40 iterations.
Cross-entropy in the logits is convex, so this is a convex program up to the
iteration cap. A temperature is then chosen on validation cross-entropy from
the grid (0.5, 0.75, 1, 1.25, 1.5, 2, 3). Test prompts are not used for either
choice.

Proper-score risk is the mean over test prompts of the mean loss of that
prompt's ratings. Intervals are item-clustered percentile 95% intervals,
B = 1000, seed 20261126.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from metrics import (  # noqa: E402
    cca_sq,
    ce_one,
    item_mean_losses,
    linear_cka,
    normalized_opd,
    pearson,
    percentile_interval,
    rbf_cka,
    softmax,
    unbiased_risk,
    median_gamma,
)

sys.path.insert(0, str(ROOT.parent / "36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026"))
from data_io import (  # noqa: E402
    ASPECTS,
    _chaos_items,
    _load_dices,
    load_summeval,
    split_by_group,
)

HASH_DIM = 256
N_COMP = 24
L2 = 1e-2
MAXITER = 40
SPLIT_SEED = 0
BOOT_SEED = 20261126
B = 1000
TEMPS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
TOKEN = re.compile(r"[a-z0-9]+")
VERIBENCH_COMMIT = "460466dd10b735c5ff25f9e9cb21f0441fcf960b"
VERIBENCH_ROOT = Path(
    "/tmp/ttj/experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026"
)


def hash_features(texts: list[str], dim: int = HASH_DIM) -> np.ndarray:
    x = np.zeros((len(texts), dim), dtype=np.float64)
    for i, text in enumerate(texts):
        toks = TOKEN.findall(text.lower())
        grams = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        if not grams:
            grams = ["empty"]
        for gram in grams:
            digest = hashlib.md5(gram.encode()).hexdigest()
            idx = int(digest[:8], 16) % dim
            sign = 1.0 if int(digest[8], 16) % 2 == 0 else -1.0
            x[i, idx] += sign
        norm = np.linalg.norm(x[i])
        if norm > 0:
            x[i] /= norm
    return x


def svd_project(x_train, others, n_comp=N_COMP):
    mu = x_train.mean(axis=0)
    centered = x_train - mu
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(s > 1e-8))
    k = max(1, min(n_comp, rank, centered.shape[0] - 1 if centered.shape[0] > 1 else 1))
    basis = vt[:k].T

    def project(x):
        z = (x - mu) @ basis
        return np.concatenate([z, np.ones((z.shape[0], 1))], axis=1)

    return project(x_train), [project(x) for x in others], k


def fit_ce(x, targets):
    n, d = x.shape
    k = targets.shape[1]

    def loss(w):
        prob = softmax(x @ w.reshape(d, k))
        p = np.clip(prob, 1e-12, 1.0)
        ce = float(-(targets * np.log(p)).sum(axis=1).mean())
        weight = w.reshape(d, k)
        pen = 0.5 * L2 * float(np.sum(weight[:-1] ** 2))
        return ce + pen

    result = minimize(loss, np.zeros(d * k), method="L-BFGS-B", options={"maxiter": MAXITER, "ftol": 1e-10})
    return result.x.reshape(d, k), {
        "success": bool(result.success),
        "nit": int(result.nit),
        "fun": float(result.fun),
        "convex_in_weights": True,
        "optimizer": "L-BFGS-B",
        "start": "zero",
        "maxiter": MAXITER,
        "l2": L2,
        "optimum_guaranteed": False,
    }


def per_item_losses(prob, ratings, indices):
    ce, br, rp = [], [], []
    for row, i in enumerate(indices):
        stats = item_mean_losses(prob[row], ratings[i])
        ce.append(stats["ce"])
        br.append(stats["brier"])
        rp.append(stats["rps"])
    return np.asarray(ce), np.asarray(br), np.asarray(rp)


def bootstrap(ce, br, rp, seed=BOOT_SEED, draws=B):
    rng = np.random.default_rng(seed)
    n = ce.shape[0]
    idx = rng.integers(0, n, size=(draws, n))

    def pack(v):
        if not np.isfinite(v).all():
            return {"mean": float("inf"), "lo": None, "hi": None, "draws": draws, "unit": "test prompt"}
        samples = v[idx].mean(axis=1)
        lo, hi = percentile_interval(samples)
        return {
            "mean": float(np.mean(v)),
            "lo": lo,
            "hi": hi,
            "draws": draws,
            "level": 0.95,
            "interval": "item-clustered percentile",
            "unit": "test prompt",
        }

    rps_mean = pack(rp)
    if rps_mean["mean"] == float("inf") or rps_mean["lo"] is None:
        sqrt_rps = {"mean": float("inf"), "lo": None, "hi": None}
    else:
        samples = np.sqrt(np.maximum(rp[idx].mean(axis=1), 0.0))
        lo, hi = percentile_interval(samples)
        sqrt_rps = {"mean": float(np.sqrt(rps_mean["mean"])), "lo": lo, "hi": hi,
                    "draws": draws, "level": 0.95, "interval": "item-clustered percentile of sqrt(mean RPS)"}
    return {"ce": pack(ce), "brier": pack(br), "rps": rps_mean, "sqrt_rps": sqrt_rps}


def histograms(ratings, indices, k):
    rows = []
    for i in indices:
        counts = np.bincount(np.asarray(ratings[i], dtype=int), minlength=k).astype(float)
        rows.append(counts / counts.sum())
    return np.stack(rows)


def transforms(z, seed=0):
    rng = np.random.default_rng(seed)
    k = z.shape[1]
    reversal = np.eye(k)[::-1]
    cycle = np.eye(k)[:, list(range(1, k)) + [0]]
    gauss = rng.normal(size=(k, k))
    q, _ = np.linalg.qr(gauss)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    shear = np.eye(k)
    if k >= 2:
        shear[0, 1] = 0.85
        shear[1, 0] = -0.35
    if k >= 3:
        shear[0, 2] = 0.4
    out = {
        "identity": z,
        "sign_flip": -z,
        "reversal_permutation": z @ reversal,
        "cycle_permutation": z @ cycle,
        "orthogonal": z @ q,
        "shear": z @ shear,
    }
    return out, {"reversal": reversal, "cycle": cycle, "orthogonal": q, "shear": shear}


def alignment(z, human, values):
    prob = softmax(z)
    pred = prob @ values
    hist_mean = human @ values
    gamma = median_gamma(z)
    return {
        "linear_cka": linear_cka(z, human),
        "normalized_opd": normalized_opd(z, human),
        "cca_sq": cca_sq(z, human),
        "pearson_expected_rating": pearson(pred, hist_mean),
        "rbf_cka": rbf_cka(z, human, gamma),
        "rbf_gamma_frozen_on_this_matrix": gamma,
    }


def choose_temperature(logits, ratings, indices, values):
    best_t, best_ce = 1.0, float("inf")
    for t in TEMPS:
        prob = softmax(logits / t)
        ce, _, _ = per_item_losses(prob, ratings, indices)
        risk = unbiased_risk(ce)
        if risk < best_ce:
            best_ce, best_t = risk, t
    return best_t, best_ce


def conditional_collapse(prob, ratings, indices, values):
    """Replace within-prompt orthogonal residual by zero.

    X = mu(c) + z, with z mean-zero inside the prompt and exactly orthogonal to
    the centered human ratings, and with mean square equal to the judge's
    predictive variance. Collapsing sets z = 0. Proper scores compare the
    calibrated distribution with a point mass at the nearest class to mu.
    """
    xs, xcol, ys = [], [], []
    ce_q, ce_d, br_q, br_d, rp_q, rp_d = [], [], [], [], [], []
    n_with_noise = 0
    for row, i in enumerate(indices):
        y_idx = np.asarray(ratings[i], dtype=int)
        y = values[y_idx]
        q = prob[row]
        mu = float(q @ values)
        var = float(q @ (values ** 2) - mu ** 2)
        var = max(var, 0.0)
        n = y.shape[0]
        z = np.zeros(n)
        if n >= 2 and var > 1e-12:
            direction = np.zeros(n)
            direction[0] = 1.0
            direction[1] = -1.0
            yc = y - y.mean()
            if float(np.dot(yc, yc)) > 1e-12:
                direction = direction - yc * (np.dot(direction, yc) / np.dot(yc, yc))
            if float(np.dot(direction, direction)) > 1e-12:
                direction = direction * np.sqrt(var * n / np.dot(direction, direction))
                z = direction
                n_with_noise += 1
        xs.append(mu + z)
        xcol.append(np.full(n, mu))
        ys.append(y)
        nearest = int(np.argmin(np.abs(values - mu)))
        delta = np.zeros_like(q)
        delta[nearest] = 1.0
        sq, sd = item_mean_losses(q, y_idx), item_mean_losses(delta, y_idx)
        ce_q.append(sq["ce"]); ce_d.append(sd["ce"])
        br_q.append(sq["brier"]); br_d.append(sd["brier"])
        rp_q.append(sq["rps"]); rp_d.append(sd["rps"])
    x = np.concatenate(xs).reshape(-1, 1)
    xc = np.concatenate(xcol).reshape(-1, 1)
    y = np.concatenate(ys).reshape(-1, 1)
    return {
        "n_prompts_with_orthogonal_residual": n_with_noise,
        "n_prompts": int(len(indices)),
        "linear_cka_uncollapsed": linear_cka(x, y),
        "linear_cka_collapsed": linear_cka(xc, y),
        "normalized_opd_uncollapsed": normalized_opd(x, y),
        "normalized_opd_collapsed": normalized_opd(xc, y),
        "pearson_uncollapsed": pearson(x, y),
        "pearson_collapsed": pearson(xc, y),
        "proper_uncollapsed": bootstrap(np.asarray(ce_q), np.asarray(br_q), np.asarray(rp_q)),
        "proper_pointmass": bootstrap(np.asarray(ce_d), np.asarray(br_d), np.asarray(rp_d)),
    }


def subset(data, indices):
    indices = np.asarray(indices, dtype=int)
    return {
        "texts": [data.texts[i] for i in indices],
        "ratings": [data.ratings[i] for i in indices],
        "groups": [data.groups[i] for i in indices],
    }


def prepare(data):
    train, val, test = split_by_group(data.groups, seed=SPLIT_SEED)
    features = hash_features(data.texts)
    x_train, (x_val, x_test), rank = svd_project(features[train], [features[val], features[test]])
    k = data.n_classes
    targets = []
    for i in train:
        counts = np.bincount(np.asarray(data.ratings[i], dtype=int), minlength=k).astype(float)
        targets.append(counts / counts.sum())
    weight, fit_info = fit_ce(x_train, np.stack(targets))
    fit_info["svd_components"] = rank
    logits = {"train": x_train @ weight, "val": x_val @ weight, "test": x_test @ weight}
    t, val_ce = choose_temperature(logits["val"], data.ratings, val, data.class_values)
    fit_info["temperature"] = t
    fit_info["validation_ce"] = val_ce
    z = logits["test"] / t
    human = histograms(data.ratings, test, k)
    base_prob = softmax(z)
    ce, br, rp = per_item_losses(base_prob, data.ratings, test)
    mats, factors = transforms(z, seed=0)
    # Freeze RBF gamma on the identity logits so later matrices are not retuned.
    gamma = median_gamma(z)
    rows = {}
    for name, matrix in mats.items():
        prob = softmax(matrix)
        ce_t, br_t, rp_t = per_item_losses(prob, data.ratings, test)
        pred = prob @ data.class_values
        hist_mean = human @ data.class_values
        rows[name] = {
            "linear_cka": linear_cka(matrix, human),
            "normalized_opd": normalized_opd(matrix, human),
            "cca_sq": cca_sq(matrix, human),
            "pearson_expected_rating": pearson(pred, hist_mean),
            "rbf_cka": rbf_cka(matrix, human, gamma),
            "proper": bootstrap(ce_t, br_t, rp_t),
        }
    rows["identity"]["proper"] = bootstrap(ce, br, rp)
    collapse = conditional_collapse(base_prob, data.ratings, test, data.class_values)
    return {
        "summary": data.summary(),
        "n_train": int(train.size),
        "n_val": int(val.size),
        "n_test": int(test.size),
        "fit": fit_info,
        "rbf_gamma_frozen_on_identity_logits": gamma,
        "transforms": rows,
        "collapse": collapse,
        "notes": data.notes,
    }


def load_veribench():
    ratings_dir = VERIBENCH_ROOT / "ratings"
    packet = VERIBENCH_ROOT / "rating_task_packet" / "main_rating_task"
    if not ratings_dir.is_dir():
        return None, (
            f"VeriBench experiment 09 ratings directory is not at {ratings_dir}. "
            "No file named R_v1 was found in the shallow clone of "
            "https://github.com/brando90/trustworthy-theorem-judges "
            f"at commit {VERIBENCH_COMMIT}."
        )
    import csv
    bucket = {}
    n_scored = 0
    for path in sorted(ratings_dir.glob("*native_main*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                raw = (row.get("final_score_0_5") or "").strip()
                if raw == "":
                    continue
                score = int(float(raw))
                if score < 0 or score > 5:
                    continue
                item = row["item_id"]
                candidate = row.get("candidate_file") or ""
                lean = packet / candidate
                text = lean.read_text(errors="replace") if lean.is_file() else f"{row.get('task_folder','')} {candidate}"
                rec = bucket.setdefault(item, {"text": text, "group": item, "y": [], "rater": []})
                rec["y"].append(score)
                rec["rater"].append(f"{row.get('rater_name','')}::{row.get('rater_id','')}")
                n_scored += 1
    from data_io import JudgingSet
    texts, groups, ratings, raters = [], [], [], []
    for rec in bucket.values():
        if not rec["y"]:
            continue
        texts.append(rec["text"][:8000])
        groups.append(rec["group"])
        ratings.append(np.asarray(rec["y"], dtype=int))
        raters.append(rec["rater"])
    notes = (
        "VeriBench functional semantic correctness scores from trustworthy-theorem-judges "
        "experiment 09 native-main CSVs, column final_score_0_5, integer grades 0 through 5. "
        f"Clone commit {VERIBENCH_COMMIT}. No artifact in that clone is named R_v1; "
        "this is the live experiment 09 numeric collection, not a frozen file called R_v1. "
        f"Scored rows kept: {n_scored}. rater_id is the roster id stored with rater_name. "
        "Split group is item_id. The same person can rate items that land in different splits."
    )
    data = JudgingSet(
        "veribench_fsc_exp09",
        "ordinal",
        [0, 1, 2, 3, 4, 5],
        texts,
        groups,
        ratings,
        raters,
        notes,
    )
    return data, notes


def aspect_view(expert, aspect: str):
    from data_io import JudgingSet
    keep = [i for i, text in enumerate(expert.texts) if text.startswith(f"aspect {aspect}.")]
    data = JudgingSet(
        f"summeval_expert_{aspect}",
        expert.kind,
        expert.class_values,
        [expert.texts[i] for i in keep],
        [expert.groups[i] for i in keep],
        [expert.ratings[i] for i in keep],
        [expert.rater_ids[i] for i in keep],
        f"SummEval expert, aspect {aspect} only. " + expert.notes,
    )
    return data


def main():
    jobs = []
    expert = load_summeval("expert")
    for aspect in ASPECTS:
        jobs.append(aspect_view(expert, aspect))
    jobs.append(_chaos_items("chaosNLI_snli.jsonl", False))
    jobs.append(_chaos_items("chaosNLI_mnli_m.jsonl", False))
    jobs.append(_chaos_items("chaosNLI_alphanli.jsonl", True))
    jobs.append(_load_dices("990"))
    veribench, veribench_note = load_veribench()
    if veribench is not None and veribench.n_items >= 30:
        jobs.append(veribench)
    results = {
        "protocol": {
            "split": "prompt group, fractions 0.6/0.2/0.2, seed 0",
            "bootstrap": "item-clustered percentile 95%, B=1000, seed 20261126",
            "proper_risk": "mean over test prompts of the mean loss inside the prompt",
            "judge": "CE-trained softmax linear head, temperature chosen on validation CE",
            "veribench_note": veribench_note,
        },
        "datasets": {},
    }
    for data in jobs:
        print(f"fitting {data.name} items={data.n_items}", flush=True)
        results["datasets"][data.name] = prepare(data)
        ident = results["datasets"][data.name]["transforms"]["identity"]
        print(
            f"  test CE {ident['proper']['ce']['mean']:.4f} "
            f"CKA {ident['linear_cka']:.4f}",
            flush=True,
        )
    out = ROOT / "results.json"
    out.write_text(json.dumps(results, indent=2, default=_json_default))
    write_tables(results)
    print(f"wrote {out}")


def _json_default(obj):
    if isinstance(obj, float):
        if obj == float("inf"):
            return "inf"
        if obj != obj:
            return None
    raise TypeError(type(obj))


def _fmt(block, key):
    b = block[key]
    if b.get("lo") is None:
        return f"{b['mean']}"
    return f"{b['mean']:.4f} [{b['lo']:.4f}, {b['hi']:.4f}]"


def write_tables(results):
    table_dir = ROOT / "tables"
    table_dir.mkdir(exist_ok=True)
    lines = [
        "# Invariance-group ladder on held-out test prompts",
        "",
        "Linear CKA and normalized OPD are computed on the test logit matrix against the test human histogram. "
        "Proper scores use `softmax` of the transformed logits. Lower CE, Brier, and sqrt(RPS) are better. "
        "Intervals are item-clustered percentile 95% intervals, B = 1000.",
        "",
        "| Dataset | Transform | Linear CKA | Normalized OPD | CCA sum of squares | Pearson of expected rating | Test CE | Test Brier | Test sqrt(RPS) |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for name, payload in results["datasets"].items():
        base_cka = payload["transforms"]["identity"]["linear_cka"]
        for transform, row in payload["transforms"].items():
            lines.append(
                "| {ds} | {tr} | {cka:.6f} | {opd:.6f} | {cca:.6f} | {pear:.4f} | {ce} | {br} | {sr} |".format(
                    ds=name,
                    tr=transform,
                    cka=row["linear_cka"],
                    opd=row["normalized_opd"],
                    cca=row["cca_sq"],
                    pear=row["pearson_expected_rating"],
                    ce=_fmt(row["proper"], "ce"),
                    br=_fmt(row["proper"], "brier"),
                    sr=_fmt(row["proper"], "sqrt_rps"),
                )
            )
        del base_cka
    (table_dir / "invariance_ladder.md").write_text("\n".join(lines) + "\n")
    collapse = [
        "# Conditional-spread collapse on held-out test ratings",
        "",
        "Rows are individual test ratings. The uncollapsed scalar is the judge's expected rating plus a within-prompt residual that is exactly orthogonal to the centered human ratings and has the judge's predictive variance. Collapsing sets that residual to zero. The point-mass forecast puts mass 1 on the class nearest the expected rating.",
        "",
        "| Dataset | CKA uncollapsed | CKA collapsed | OPD uncollapsed | OPD collapsed | Pearson uncollapsed | Pearson collapsed | CE distribution | CE point mass | Brier distribution | Brier point mass | sqrt(RPS) distribution | sqrt(RPS) point mass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for name, payload in results["datasets"].items():
        c = payload["collapse"]
        collapse.append(
            "| {ds} | {a:.4f} | {b:.4f} | {c:.4f} | {d:.4f} | {e:.4f} | {f:.4f} | {g} | {h} | {i} | {j} | {k} | {l} |".format(
                ds=name,
                a=c["linear_cka_uncollapsed"],
                b=c["linear_cka_collapsed"],
                c=c["normalized_opd_uncollapsed"],
                d=c["normalized_opd_collapsed"],
                e=c["pearson_uncollapsed"],
                f=c["pearson_collapsed"],
                g=_fmt(c["proper_uncollapsed"], "ce"),
                h=_fmt(c["proper_pointmass"], "ce"),
                i=_fmt(c["proper_uncollapsed"], "brier"),
                j=_fmt(c["proper_pointmass"], "brier"),
                k=_fmt(c["proper_uncollapsed"], "sqrt_rps"),
                l=_fmt(c["proper_pointmass"], "sqrt_rps"),
            )
        )
    (table_dir / "conditional_collapse.md").write_text("\n".join(collapse) + "\n")


if __name__ == "__main__":
    main()
