"""Independently replay split membership, all point-estimate RPS and selected sample weights."""
import csv
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

import analyze as analysis

origin, = analysis.EXPERIMENTS.glob("10_frontier*/scripts")
sys.path.insert(0, str(origin))
from analyze_all import build_250_grouped_splits


def main():
    q, items, ys, names = analysis.load_data()
    family = np.array([item["family"] for item in items])
    families, codes = np.unique(family, return_inverse=True)
    original = build_250_grouped_splits(items)
    replay = list(analysis.splits(items))
    assert len(replay) == len(original) == 250
    all_rps = {b: [] for b in analysis.BUDGETS}
    constant_rps = {b: [] for b in analysis.BUDGETS}
    all_spearman = {b: [] for b in analysis.BUDGETS}
    # Independent exact-order reference from archived decimal strings and raw labels.
    pool = analysis.load("10_frontier*/results/judge_forecasts.json", analysis.POOL_HASH)
    rows_by_judge = [{r["item"]: r["probabilities"] for r in pool["native_prob_forecasts"][name]}
                     for name in names]
    decimal_mean = np.array([[sum(Decimal(str(p))*i for i, p in enumerate(rows[item["item"]]))
                              for rows in rows_by_judge] for item in items], dtype=object)
    fraction_mean = np.array([Fraction(int(y.sum()), len(y)) for y in ys], dtype=object)
    # scipy.rankdata uses object comparisons for exact ties. These whole-data ranks
    # preserve the same ties and ordering when restricted and reranked within a split.
    human_rank = rankdata(fraction_mean)
    judge_rank = rankdata(decimal_mean, axis=0)
    np.testing.assert_array_equal(rankdata(analysis.observed_mean_rank_keys(ys)), human_rank)
    np.testing.assert_array_equal(rankdata(analysis.forecast_mean_rank_keys(q), axis=0), judge_rank)
    C = q.cumsum(-1)[..., :-1]
    h = q @ np.arange(6)
    test_rps = np.array([((C[i][:, None] - (y[:, None] <= np.arange(5)))**2).mean((1, 2))
                         for i, y in enumerate(ys)])
    weights = np.random.default_rng(20261123).exponential(size=(3, len(families)))
    for s, (hist, (train, test)) in enumerate(zip(original, replay)):
        assert np.array_equal(hist["train_families_ordered"], train)
        assert set(hist["test_families"]) == test
        te = np.isin(family, list(test))
        panel_spearman = np.array([spearmanr(j, human_rank[te]).statistic
                                   if np.ptp(j) else np.nan for j in judge_rank[te].T])
        rng = np.random.default_rng(np.random.SeedSequence([20261121, 190, s]))
        draws = [rng.permutation(row) for row in ys]
        for n, k in analysis.BUDGETS:
            ids = np.flatnonzero(np.isin(family, train[:n]))
            obs = np.stack([draws[i][:k] for i in ids])
            ce = np.stack([-np.log(q[ids, :, obs[:, j]]) for j in range(k)]).mean(0)
            rps = ((C[ids, :, None, :] - (obs[:, None, :, None] <= np.arange(5)))**2).mean((2, 3))
            means = h[ids]
            labels = obs.mean(1)
            w_list = [np.ones(len(ids))]
            if s in (0, 99, 249):
                w_list.extend(weights[:, codes[ids]])
            winners = []
            for w in w_list:
                t = labels - np.average(labels, weights=w)
                u = means - np.average(means, axis=0, weights=w)
                denominator = np.sqrt(np.sum(w * t**2) * np.sum(w[:, None] * u**2, axis=0))
                numerator = np.sum((w*t)[:, None] * u, axis=0)
                pr = np.divide(numerator, denominator, out=np.full(len(names), -np.inf),
                               where=(np.ptp(means, axis=0) > 1e-12) & (np.ptp(labels) > 1e-12))
                metrics = (np.average(ce, axis=0, weights=w), -pr, np.average(rps, axis=0, weights=w))
                winners.append([np.flatnonzero(v <= min(v) + 1e-12)[0] for v in metrics])
            all_rps[n, k].append(test_rps[te].mean(0)[winners[0]])
            all_spearman[n, k].append(panel_spearman[winners[0]])
            # Reconstruct the constant from each retained individual training rating.
            counts = np.bincount(obs.ravel(), minlength=6) + 0.5
            cdf = np.cumsum(counts / counts.sum())[:-1]
            per_artifact = [np.mean((cdf - (ys[i][:, None] <= np.arange(5)))**2)
                            for i in np.flatnonzero(te)]
            constant_rps[n, k].append(np.mean(per_artifact))
            if len(w_list) > 1:
                actual, _, _ = analysis.select(np.array(w_list), means, labels, ce, rps)
                assert np.array_equal(winners, actual)
    path = Path(__file__).with_name("scores.csv")
    for row in csv.DictReader(path.open()):
        if row["selector"] in analysis.OBJECTIVES:
            b = int(row["train_families"]), int(row["train_ratings_per_artifact"])
            j = analysis.OBJECTIVES.index(row["selector"])
            np.testing.assert_allclose(float(row["rps"]), np.mean(all_rps[b], axis=0)[j], atol=1e-13, rtol=0)
            np.testing.assert_allclose(float(row["spearman"]), np.nanmean(all_spearman[b], axis=0)[j],
                                       atol=1e-13, rtol=0)
        elif row["selector"] == "constant":
            b = int(row["train_families"]), int(row["train_ratings_per_artifact"])
            np.testing.assert_allclose(float(row["rps"]), np.mean(constant_rps[b]), rtol=0, atol=1e-13)
    seen = set()
    with Path(__file__).with_name("paired_rps.csv").open() as stream:
        for row in csv.DictReader(stream):
            b = int(row["train_families"]), int(row["train_ratings_per_artifact"])
            name = row["subtrahend_selector"]
            scores = np.asarray(all_rps[b])
            if name == "constant":
                other = constant_rps[b]
                assert not row["family_bayes_95_lo"] and not row["family_bayes_95_hi"]
                assert int(row["draws"]) == 0
            else:
                other = scores[:, analysis.OBJECTIVES.index(name)]
                assert int(row["draws"]) == analysis.N_BOOT
            assert int(row["matched_splits"]) == 250 and (b, name) not in seen
            seen.add((b, name))
            np.testing.assert_allclose(float(row["pearson_selected_minus_other_test_rps"]),
                                       np.mean(scores[:, 1] - other), atol=1e-13, rtol=0)
    assert seen == {(b, name) for b in analysis.BUDGETS for name in ("ce", "rps", "constant")}
    print("6 constant baselines and 18 paired contrasts reproduced from individual ratings.")
    print("250 split schedules matched original code. All 18 held-out RPS point estimates reproduced.")
    print("18 held-out Spearman means reproduced from exact decimal forecasts and rational raw-label means.")
    print("54 non-uniform family-weight settings x 3 selectors reproduced.")


if __name__ == "__main__":
    main()
