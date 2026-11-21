"""Descriptive pooled, system-ranking, and leave-one-human-vote-out diagnostics on grades 1–5."""
import numpy as np
from scipy.stats import kendalltau, rankdata

GRADES = np.arange(1, 6)
TIE_TOLERANCE = 1e-12


def vote_counts(counts):
    values = np.asarray(counts, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 5 or not len(values) or
            not np.isfinite(values).all() or (values < 0).any() or
            not np.equal(values, np.floor(values)).all() or (values.sum(axis=1) == 0).any()):
        raise ValueError("Expected a nonempty array of five nonnegative integer counts per item with positive total.")
    return values


def tie_levels(values):
    order = np.argsort(values)
    levels = np.empty_like(order)
    levels[order] = np.r_[0, np.cumsum(np.diff(values[order]) > TIE_TOLERANCE)]
    return levels


def correlation(x, y):
    if np.ptp(x) <= TIE_TOLERANCE or np.ptp(y) <= TIE_TOLERANCE:
        return np.nan
    xc, yc = x - x.mean(), y - y.mean()
    return float((xc @ yc) / (np.linalg.norm(xc) * np.linalg.norm(yc)))


def mode_weights(rows):
    ties = rows >= rows.max(axis=1, keepdims=True) - TIE_TOLERANCE
    return ties / ties.sum(axis=1, keepdims=True)


def cumulative_ece(q, h):
    """Equal weight per item and cumulative threshold; endpoints 0 and 1 included."""
    forecast = np.clip(q.cumsum(axis=1)[:, :-1], 0, 1)
    observed = h.cumsum(axis=1)[:, :-1]
    bins = np.minimum((10 * forecast).astype(int), 9)
    total = 0.0
    for threshold in range(4):
        residual = observed[:, threshold] - forecast[:, threshold]
        signed_sums = np.bincount(bins[:, threshold], weights=residual, minlength=10)
        total += np.abs(signed_sums).sum()
    return float(total / forecast.size)


def system_scores(x, y, system):
    """Rank systems on their item-averaged grades; give each unordered pair equal weight."""
    ids = np.asarray(system)
    if ids.shape != x.shape:
        raise ValueError("Expected one system identifier for every item.")
    names, index = np.unique(ids, return_inverse=True)
    group_sizes = np.bincount(index)
    sx = np.bincount(index, weights=x) / group_sizes
    sy = np.bincount(index, weights=y) / group_sizes
    rx, ry = tie_levels(sx), tie_levels(sy)
    left, right = np.triu_indices(len(names), k=1)
    predicted = np.sign(rx[left] - rx[right])
    target = np.sign(ry[left] - ry[right])
    agreement = np.where(predicted == target, 1.0, np.where((predicted == 0) | (target == 0), 0.5, 0.0))
    return dict(system_spearman=correlation(rankdata(rx), rankdata(ry)),
                system_pair_accuracy=float(agreement.mean()) if len(agreement) else np.nan,
                system_mean_mae=float(np.abs(sx - sy).mean()))


def diagnostics(q, counts, system=None):
    """Average items equally regardless of rating count; mode ties split uniformly."""
    counts = vote_counts(counts)
    q = np.asarray(q, dtype=float)
    if (q.shape != counts.shape or not np.isfinite(q).all() or (q < 0).any()
            or not np.allclose(q.sum(axis=1), 1.0, atol=1e-10, rtol=0)):
        raise ValueError("Expected a finite five-category probability vector for every item.")
    h = counts / counts.sum(axis=1, keepdims=True)
    x, y = q @ GRADES, h @ GRADES
    predicted_modes, human_modes = mode_weights(q), mode_weights(h)
    rx, ry = tie_levels(x), tie_levels(y)
    xc, yc = x - x.mean(), y - y.mean()
    denominator = np.mean(xc**2) + np.mean(yc**2) + (x.mean() - y.mean())**2
    # A constant versus a varying series has zero Lin concordance; identical constants give 0/0.
    lin = float(2 * np.mean(xc * yc) / denominator) if denominator > TIE_TOLERANCE**2 else np.nan
    result = dict(
        pearson=correlation(x, y), spearman=correlation(rankdata(rx), rankdata(ry)),
        kendall=float(kendalltau(rx, ry, variant="b").statistic) if rx.max() and ry.max() else np.nan,
        lin=lin, forecast_mean=float(x.mean()), human_mean=float(y.mean()),
        mean_bias=float((x - y).mean()), mean_mae=float(np.abs(x - y).mean()),
        cumulative_ece=cumulative_ece(q, h),
        modal_agreement=float((predicted_modes * human_modes).sum(axis=1).mean()),
        map_accuracy=float((predicted_modes * h).sum(axis=1).mean()),
        forecast_mode_tie_fraction=float(((predicted_modes > 0).sum(axis=1) > 1).mean()),
        human_mode_tie_fraction=float(((human_modes > 0).sum(axis=1) > 1).mean()),
    )
    if system is not None:
        result.update(system_scores(x, y, system))
    return result


def human_reference(counts):
    """Predict each omitted rating from its item's other votes and half a count in every bin.

    Return per-item means averaged equally over items. Multiclass Brier is the sum
    over five bins (0–2). RPS is the mean over four cumulative thresholds (0–1).
    """
    counts = vote_counts(counts)
    totals = counts.sum(axis=1)
    measures = np.zeros((len(counts), 4))
    for category in range(5):
        available = counts[:, category] > 0
        if not available.any():
            continue
        retained = counts[available].copy()
        retained[:, category] -= 1
        remaining = retained + 0.5
        forecast = remaining / remaining.sum(axis=1, keepdims=True)
        observed = np.eye(5)[category]
        residual = forecast - observed
        loss = np.column_stack([
            np.mean(residual.cumsum(axis=1)[:, :-1]**2, axis=1),
            -np.log(forecast[:, category]), np.sum(residual**2, axis=1),
            mode_weights(forecast)[:, category],
        ])
        measures[available] += loss * (counts[available, category] / totals[available])[:, None]
    return dict(zip(("rps", "ce", "brier", "map_accuracy"), map(float, measures.mean(axis=0))))


def check_fixtures():
    counts = np.eye(5, dtype=int)
    perfect = diagnostics(counts, counts, system=np.arange(5))
    for metric in ("pearson", "spearman", "kendall", "lin", "map_accuracy", "modal_agreement",
                   "system_spearman", "system_pair_accuracy"):
        np.testing.assert_allclose(perfect[metric], 1)
    for metric in ("mean_bias", "mean_mae", "cumulative_ece", "system_mean_mae"):
        np.testing.assert_allclose(perfect[metric], 0)
    uniform = np.full((5, 5), 0.2)
    mixed = diagnostics(uniform, counts, system=np.arange(5))
    assert all(np.isnan(mixed[m]) for m in ("pearson", "spearman", "kendall", "system_spearman"))
    np.testing.assert_allclose([mixed["lin"], mixed["map_accuracy"], mixed["system_pair_accuracy"]], [0, .2, .5])
    tied = diagnostics([[.5, .5, 0, 0, 0]], [[1, 2, 1, 0, 0]])
    np.testing.assert_allclose([tied["map_accuracy"], tied["modal_agreement"]], [.375, .5])
    reverse = diagnostics(counts[::-1], counts, system=np.arange(5))
    np.testing.assert_allclose([reverse["spearman"], reverse["system_pair_accuracy"]], [-1, 0])
    assert np.isnan(diagnostics(uniform, np.ones((5, 5), dtype=int))["lin"])
    # With one vote per item, every other-vote histogram is empty, hence uniform.
    ref = human_reference(counts)
    np.testing.assert_allclose([ref[k] for k in ("rps", "ce", "brier", "map_accuracy")],
                               [.2, np.log(5), .8, .2])
    assert np.isclose(cumulative_ece(np.array([[1, 0, 0, 0, 0.]]), np.array([[0, 0, 0, 0, 1.]])), 1)
    # Direct literal leave-one-out enumeration with unequal numbers of votes and held-out classes.
    observed = [[0, 1, 1], [4, 4]]
    expected = []
    for votes in observed:
        rows = []
        for i, y in enumerate(votes):
            p = np.bincount(votes[:i] + votes[i + 1:], minlength=5) + .5
            p /= p.sum()
            error = p - np.eye(5)[y]
            rows.append([np.mean(np.cumsum(error)[:4]**2), -np.log(p[y]), error @ error,
                         (p[y] == p.max()) / np.count_nonzero(p == p.max())])
        expected.append(np.mean(rows, axis=0))
    ref = human_reference([np.bincount(v, minlength=5) for v in observed])
    np.testing.assert_allclose(list(ref.values()), np.mean(expected, axis=0))
    print("Passed synthetic fixtures for agreement, ranks, Lin concordance, calibration and leave-one-vote-out scores.")


if __name__ == "__main__":
    check_fixtures()
