"""Post hoc draw summaries and fixed-uniform forecast comparisons. Write aggregates only."""
import csv
from pathlib import Path

import numpy as np

import analyze as study
from smoothing_sensitivity import PAIRS, uniform_reference

DIRECTORY = Path(__file__).resolve().parent


def family_weights(items):
    names = np.array([item["family"] for item in items])
    unique, indices = np.unique(names, return_inverse=True)
    rng = np.random.default_rng(20261123)
    weights = np.vstack([np.ones(len(unique)), rng.exponential(size=(study.BOOT, len(unique)))])
    return names, weights[:, indices]


def normalized(weights):
    return weights / weights.sum(axis=1, keepdims=True)


def refit_draws(inputs, base, items, ratings):
    """Repeat every fit and keep weighted test losses until the fold averages are formed."""
    names, weights = family_weights(items)
    candidates = study.grid(base)
    candidate_means = candidates @ study.R
    histogram, moments, variance = study.target(ratings)
    losses = study.loss_panel(candidates, histogram, moments, variance)
    totals = {
        budget: np.zeros((len(weights), len(inputs), len(study.METHODS), len(study.LOSS)))
        for budget in study.BUDGETS
    }
    input_index = np.arange(len(inputs))[None, :, None]
    draw_index = np.arange(len(weights))[:, None, None]
    splits = list(study.history.splits(items))
    for split, (ordered, held_out) in enumerate(splits):
        test = np.isin(names, list(held_out))
        test_weight = normalized(weights[:, test])
        test_loss = study.avg(test_weight, losses[test])
        rng = np.random.default_rng(np.random.SeedSequence([20261121, 190, split]))
        shuffled = [rng.permutation(observed) for observed in ratings]
        for k in (1, 2):
            train_hist, train_moments, train_variance = study.target([row[:k] for row in shuffled])
            train_loss = study.loss_panel(candidates, train_hist, train_moments, train_variance)
            for n in (3, 7, 15):
                train = np.isin(names, ordered[:n])
                assert not np.any(train & test)
                train_weight = normalized(weights[:, train])
                selected, _, _ = study.select(
                    train_weight, candidate_means[train], train_hist[train] @ study.R,
                    train_loss[train, ..., study.CE], train_loss[train, ..., study.RPS],
                )
                total = totals[n, k]
                total[:, :, :3] += test_loss[draw_index, input_index, selected]
                total[:, :, 3] += test_loss[:, :, 0]
                marginal = (train_weight @ train_hist[train]) * (k * train.sum()) + 0.5
                constant_loss = study.constant_losses(
                    normalized(marginal), test_weight, histogram[test], moments[test], variance[test]
                )
                total[:, :, 4] += constant_loss[:, None, :]
    return {budget: total / len(splits) for budget, total in totals.items()}


def uniform_draws(items, ratings):
    """Score a fixed uniform vector directly against ratings with identical family weights."""
    uniform = np.full(6, 1 / 6)
    thresholds = np.arange(5)
    cdf = np.cumsum(uniform)[:5]
    powers = np.arange(1, 6)
    moments = uniform @ ((np.arange(6)[:, None] / 5) ** powers)
    variance = moments[1] - moments[0] ** 2
    by_artifact = []
    for observed in ratings:
        normalized_ratings = observed / 5
        by_artifact.append([
            np.mean((cdf - (observed[:, None] <= thresholds)) ** 2),
            np.log(6),
            *np.mean((moments - normalized_ratings[:, None] ** powers) ** 2, axis=0),
            (variance - np.var(normalized_ratings, ddof=1)) ** 2,
        ])
    scores = np.asarray(by_artifact)
    family_names, weights = family_weights(items)
    total = np.zeros((len(weights), len(study.LOSS)))
    splits = list(study.history.splits(items))
    for _, test_families in splits:
        test = np.isin(family_names, list(test_families))
        total += normalized(weights[:, test]) @ scores[test]
    return total / len(splits)


def describe(differences):
    point, draws = differences[0], differences[1:]
    low, high = np.quantile(draws, [0.025, 0.975])
    return dict(
        difference=point, draws=len(draws), draw_mean=draws.mean(),
        draws_le_zero=np.count_nonzero(draws <= 0), share_le_zero=np.mean(draws <= 0),
        percentile_lower_95=low, percentile_upper_95=high,
        reflected_lower_95=2 * point - high, reflected_upper_95=2 * point - low,
    )


def main():
    inputs, base, items, ratings, _ = study.load()
    uniform = uniform_draws(items, ratings)
    for metric, score in uniform_reference(items, ratings).items():
        np.testing.assert_allclose(uniform[0, study.LOSS.index(metric)], score, atol=1e-13)
    paired = refit_draws(inputs, base, items, ratings)

    def summarize(group, n, k, method, reference, metric):
        metric_index = study.LOSS.index(metric)
        by_method = paired[n, k][:, inputs.index(group), :, metric_index]
        left = by_method[:, study.METHODS.index(method)]
        right = uniform[:, metric_index] if reference == "uniform" else by_method[:, study.METHODS.index(reference)]
        return dict(input=group, n=n, k=k, method=method, subtract=reference, metric=metric,
                    method_score=left[0], reference_score=right[0], **describe(left - right))

    rows = []
    with (DIRECTORY / "contrasts.csv").open() as stream:
        for original in csv.DictReader(stream):
            summary = summarize(original["input"], int(original["n"]), int(original["k"]),
                                original["method"], original["subtract"], original["metric"])
            for source, repeated in (("difference", "difference"), ("lower_95", "percentile_lower_95"),
                                     ("upper_95", "percentile_upper_95")):
                np.testing.assert_allclose(float(original[source]), summary[repeated], atol=1e-13, rtol=0)
            rows.append(summary)
    original_count = len(rows)
    for group in PAIRS:
        for n, k in study.BUDGETS:
            for method in study.METHODS:
                for metric in study.LOSS:
                    rows.append(summarize(group, n, k, method, "uniform", metric))
    with (DIRECTORY / "bootstrap_sensitivity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Checked {original_count} original contrasts; added {len(rows) - original_count} fixed-uniform contrasts.")


if __name__ == "__main__":
    main()
