"""Matched retrospective selection, aggregate outputs only. See the dated protocol."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
EXPERIMENTS = next(p for p in HERE.parents if p.name == "experiments")
POOL_HASH = "fcbbd3332da9196aa8e285f182baf1c9c5081d7516fc6000cf1f644208b0a8f8"
RATINGS_HASH = "b3a05699a26d3ffd9f28e793178811219885df78319b3aa57f9d393c68724ba0"
OBJECTIVES = ("ce", "pearson", "rps")
METRICS = ("rps", "ce", "tvd", "pearson", "spearman", "mean_judge", "mean_human", "bias", "mae")
N_BOOT = 1000
FLOOR = 1e-6
ARCHIVE_SCALE = 10**8
R = np.arange(6)
BUDGETS = [(n, k) for n in (3, 7, 15) for k in (1, 2)]


def load(path_pattern, digest):
    path, = EXPERIMENTS.glob(path_pattern)
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest
    return json.loads(data)


def load_data():
    pool = load("10_frontier*/results/judge_forecasts.json", POOL_HASH)
    labels = load("02_exp06*/private/historical_exp06/cached_v1/dataset.json", RATINGS_HASH)
    items = pool["artifacts"]
    keys = [r["item"] for r in items]
    by_item = {r["item"]: r for r in labels["rows"]}
    assert len(by_item) == len(set(keys)) == 75 and set(keys) == set(by_item)
    assert all(by_item[r["item"]]["family"] == r["family"] for r in items)
    ratings = [np.asarray(by_item[key]["ratings"], int) for key in keys]
    assert sorted(map(len, ratings)) == [2] * 30 + [3] * 45
    assert all(np.isin(row, R).all() for row in ratings)
    names = [s["judge_id"] for s in pool["arm_a_specs"]]
    raw, failures = [], 0
    for name in names:
        records = pool["native_prob_forecasts"][name]
        mapped = {r["item"]: r for r in records}
        assert len(records) == len(mapped) == 75 and set(mapped) == set(keys)
        raw.append([mapped[key]["probabilities"] for key in keys])
        failures += sum(r["fallback_uniform_used"] for r in records)
    assert len(names) == 15 and failures == 27
    q = np.moveaxis(np.asarray(raw), 0, 1)
    # The archive serializes each of six components to eight decimals. Keep its
    # original values, allowing their cumulative rounding error in this check.
    np.testing.assert_allclose(q.sum(-1), 1, atol=3.1e-8, rtol=0)
    assert (q >= 0).all() and pool["probability_floor"] == FLOOR
    eps = pool["probability_floor"]
    return (1 - 6 * eps) * q + eps, items, ratings, names


def splits(items):
    """Replay analyze_all.build_250_grouped_splits without invoking its model clients."""
    families = sorted({r["family"] for r in items})
    ordinary = sorted({r["family"] for r in items if r["ordinary"]})
    placeholder = sorted({r["family"] for r in items if not r["ordinary"]})
    assert len(ordinary) == 14 and len(placeholder) == 5 and not set(ordinary) & set(placeholder)
    rng = np.random.default_rng(20261116)
    for _ in range(250):
        test = set(rng.choice(ordinary, 3, replace=False)) | set(rng.choice(placeholder, 1, replace=False))
        train = rng.permutation([f for f in families if f not in test])
        yield train, test


def histograms(ratings):
    return np.array([np.bincount(row, minlength=6) / len(row) for row in ratings])


def losses(q, p):
    """Artifact × candidate expected RPS, CE and empirical TVD."""
    target = p[:, None, :]
    Q, P = q.cumsum(axis=-1)[..., :-1], target.cumsum(axis=-1)[..., :-1]
    rps = ((Q - P)**2 + P * (1 - P)).mean(axis=-1)
    ce = -(target * np.log(q)).sum(axis=-1)
    tvd = np.abs(target - q).sum(axis=-1) / 2
    return rps, ce, tvd


def corr(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean()
    denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b)
    return np.divide(b @ a, denom, out=np.full(a.shape[1], np.nan), where=denom > 1e-12)


def forecast_mean_rank_keys(q):
    """The fixed affine smoothing preserves the order of the archived decimal means."""
    units = (q - FLOOR) / (1 - 6*FLOOR) * ARCHIVE_SCALE
    exact_units = np.rint(units).astype(np.int64)
    np.testing.assert_allclose(units, exact_units, rtol=0, atol=1e-6)
    return exact_units @ R


def observed_mean_rank_keys(ratings):
    """Scale observed means to one integer denominator to preserve exact ties."""
    common = int(np.lcm.reduce([len(y) for y in ratings]))
    return np.array([int(y.sum()) * (common // len(y)) for y in ratings])


def test_panel(q, p, judge_keys, rating_keys):
    means = q @ R
    observed = p @ R
    rps, ce, tvd = losses(q, p)
    correlations = [corr(means, observed), corr(rankdata(judge_keys, axis=0), rankdata(rating_keys))]
    return np.array([rps.mean(0), ce.mean(0), tvd.mean(0), *correlations, means.mean(0),
                     np.full(means.shape[1], observed.mean()), (means - observed[:, None]).mean(0),
                     abs(means - observed[:, None]).mean(0)]).T


def best_min(values):
    return np.argmax(values <= np.min(values, axis=-1, keepdims=True) + 1e-12, axis=-1)


def select(w, mu, h, ce, rps):
    """All Bayesian-bootstrap fits at once; row zero has equal family weights."""
    w = w / w.sum(axis=1, keepdims=True)
    mbar = w @ mu
    hbar = w @ h
    cov = w @ (mu * h[:, None]) - mbar * hbar[:, None]
    var_m = np.maximum(w @ (mu**2) - mbar**2, 0)
    var_h = np.maximum(w @ (h**2) - hbar**2, 0)
    # Positive family weights preserve which observed vectors are exactly constant.
    valid = (np.ptp(mu, axis=0) > 1e-12) & (np.ptp(h) > 1e-12)
    denom = np.sqrt(var_m * var_h[:, None])
    if np.any(valid):
        assert np.all(denom[:, valid] > 0)
    neg_corr = np.full_like(cov, np.inf)
    neg_corr[:, valid] = -cov[:, valid] / denom[:, valid]
    choices = np.column_stack([best_min(w @ ce), best_min(neg_corr), best_min(w @ rps)])
    direct_corr = corr(mu, h)
    direct_loss = [ce.mean(0), np.where(np.isfinite(direct_corr), -direct_corr, np.inf), rps.mean(0)]
    assert np.array_equal(choices[0], [best_min(v) for v in direct_loss])
    return choices, np.sum(~valid), int(not np.any(valid))


def save_csv(filename, rows):
    with (HERE / filename).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    q, items, ratings, names = load_data()
    family_names, fid = np.unique([r["family"] for r in items], return_inverse=True)
    family_col = family_names[fid]
    full = histograms(ratings)
    means = q @ R
    judge_keys = forecast_mean_rank_keys(q)
    rating_keys = observed_mean_rank_keys(ratings)
    full_rps = losses(q, full)[0]
    # Independent direct individual-rating reconstruction for every forecast and artifact.
    cuts = R[:-1]
    for i, ys in enumerate(ratings):
        direct = ((q[i].cumsum(-1)[:, None, :-1] - (ys[:, None] <= cuts))**2).mean(axis=(1, 2))
        np.testing.assert_allclose(direct, full_rps[i], atol=1e-14)
    boot = np.random.default_rng(20261123).exponential(size=(N_BOOT, len(family_names)))
    w = np.vstack([np.ones((1, len(family_names))), boot])
    by_budget = {b: [] for b in BUDGETS}
    estimates = {b: np.zeros((len(w), 3)) for b in BUDGETS}
    selections = {b: np.zeros((3, len(names)), dtype=int) for b in BUDGETS}
    counts = {b: [] for b in BUDGETS}
    for s, (train, test) in enumerate(splits(items)):
        assert len(train) == 15 and len(test) == 4 and not set(train) & test
        te = np.flatnonzero(np.isin(family_col, list(test)))
        wte = w[:, fid[te]]
        test_rps = (wte / wte.sum(1, keepdims=True)) @ full_rps[te]
        panel = test_panel(q[te], full[te], judge_keys[te], rating_keys[te])
        all_test_labels = sum(len(ratings[i]) for i in te)
        rng = np.random.default_rng(np.random.SeedSequence([20261121, 190, s]))
        perms = [rng.permutation(row) for row in ratings]
        for k in (1, 2):
            observed = histograms([row[:k] for row in perms])
            train_rps, train_ce, _ = losses(q, observed)
            h = observed @ R
            for n in (3, 7, 15):
                b = (n, k)
                tr = np.flatnonzero(np.isin(family_col, train[:n]))
                chosen, undefined, fallback = select(w[:, fid[tr]], means[tr], h[tr],
                                                     train_ce[tr], train_rps[tr])
                est = np.take_along_axis(test_rps, chosen, axis=1)
                np.testing.assert_allclose(est[0], panel[chosen[0], 0], atol=1e-12)
                estimates[b] += est
                selections[b][np.arange(3), chosen[0]] += 1
                marginal = (observed[tr] * k).sum(0) + 0.5
                marginal = marginal / marginal.sum()
                baseline = test_panel(np.broadcast_to(marginal, (len(te), 1, 6)), full[te],
                                      np.zeros((len(te), 1), dtype=int), rating_keys[te])
                baseline[0, 3:5] = np.nan  # exactly constant by construction
                by_budget[b].append(np.vstack([panel[chosen[0]], baseline]))
                counts[b].append((len(tr), k*len(tr), len(te), all_test_labels, undefined, fallback))
    rows, intervals, frequencies = [], [], []
    constant_contrasts = []
    for b in BUDGETS:
        n, k = b
        arr = np.array(by_budget[b])
        c = np.array(counts[b])
        assert arr.shape == (250, 4, len(METRICS)) and np.isfinite(arr[:, :, :3]).all()
        for j, name in enumerate((*OBJECTIVES, "constant")):
            record = {"train_families": n, "train_ratings_per_artifact": k, "selector": name,
                      "train_artifacts_mean": c[:, 0].mean(), "train_ratings_mean": c[:, 1].mean(),
                      "train_ratings_min": int(c[:, 1].min()), "train_ratings_max": int(c[:, 1].max()),
                      "test_artifacts_mean": c[:, 2].mean(), "test_ratings_mean": c[:, 3].mean(),
                      "test_artifacts_min": int(c[:, 2].min()), "test_artifacts_max": int(c[:, 2].max()),
                      "test_ratings_min": int(c[:, 3].min()), "test_ratings_max": int(c[:, 3].max()),
                      "train_pearson_undefined_candidates": int(c[:, 4].sum()),
                      "train_pearson_all_undefined_splits": int(c[:, 5].sum())}
            for col, metric in enumerate(METRICS):
                colv = arr[:, j, col]
                finite = np.isfinite(colv)
                record[metric] = colv[finite].mean() if finite.any() else ""
                record[metric + "_valid_splits"] = int(finite.sum())
            rows.append(record)
        scores = estimates[b] / 250
        for j, name in [(0, "ce"), (2, "rps")]:
            delta = scores[:, 1] - scores[:, j]
            assert np.isfinite(delta).all()
            lo, hi = np.quantile(delta[1:], [0.025, 0.975])
            intervals.append({"train_families": n, "train_ratings_per_artifact": k,
                              "subtrahend_selector": name, "pearson_selected_minus_other_test_rps": delta[0],
                              "family_bayes_95_lo": lo, "family_bayes_95_hi": hi, "draws": N_BOOT,
                              "matched_splits": 250})
        # Prespecified descriptive control: paired differences on equal-weight splits.
        pearson_minus_constant = arr[:, 1, 0] - arr[:, 3, 0]
        constant_contrasts.append({"train_families": n, "train_ratings_per_artifact": k,
                                   "subtrahend_selector": "constant",
                                   "pearson_selected_minus_other_test_rps": pearson_minus_constant.mean(),
                                   "family_bayes_95_lo": "", "family_bayes_95_hi": "",
                                   "draws": 0, "matched_splits": len(pearson_minus_constant)})
        for j, selector in enumerate(OBJECTIVES):
            assert selections[b][j].sum() == 250
            for judge, count in zip(names, selections[b][j]):
                frequencies.append({"train_families": n, "train_ratings_per_artifact": k, "selector": selector,
                                    "judge": judge, "times_chosen": int(count)})
    # Post hoc: compare the same Pearson-minus-CE gap across nested budgets.
    # Each paired bootstrap draw uses identical family weights at both budgets.
    comparisons = [((3, k), (15, k)) for k in (1, 2)] + [((n, 1), (n, 2)) for n in (3, 7, 15)]
    interactions = []
    for earlier, later in comparisons:
        earlier_scores = estimates[earlier] / 250
        later_scores = estimates[later] / 250
        interaction = (later_scores[:, 1] - later_scores[:, 0]) - (earlier_scores[:, 1] - earlier_scores[:, 0])
        lo, hi = np.quantile(interaction[1:], [0.025, 0.975])
        interactions.append({"from_train_families": earlier[0], "from_ratings_per_artifact": earlier[1],
                             "to_train_families": later[0], "to_ratings_per_artifact": later[1],
                             "change_in_pearson_minus_ce_test_rps": interaction[0],
                             "family_bayes_95_lo": lo, "family_bayes_95_hi": hi, "draws": N_BOOT})
    save_csv("exploratory_budget_interactions.csv", interactions)
    save_csv("scores.csv", rows)
    save_csv("paired_rps.csv", intervals + constant_contrasts)
    save_csv("selection_counts.csv", frequencies)
    print("250 splits x 6 budgets, 1,000 paired family reweightings with reselection.")
    for row in intervals:
        if row["subtrahend_selector"] == "ce":
            print(row["train_families"], row["train_ratings_per_artifact"],
                  *[f"{row[key]:+.6f}" for key in ("pearson_selected_minus_other_test_rps",
                                                  "family_bayes_95_lo", "family_bayes_95_hi")])


if __name__ == "__main__":
    main()
