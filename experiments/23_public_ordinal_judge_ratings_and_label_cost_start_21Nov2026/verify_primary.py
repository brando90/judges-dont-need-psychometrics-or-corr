"""Rebuild primary discrete fits, per-answer grade losses, and paired cluster bands from the two archives."""
import argparse
import csv
import hashlib
import itertools
from pathlib import Path

import numpy as np

from candidates import build as frozen_candidates
from design import draw, load

ROOT = Path(__file__).resolve().parent
METHODS = ("ce", "pearson", "uniform", "histogram", "calibrated_pearson", "original", "rps")
GRADES = np.arange(1, 6)
THRESHOLDS = GRADES[:-1]
EPSILON = 6e-6
TOL = 1e-10


def read(directory, filename):
    with (directory / filename).open() as stream:
        return list(csv.DictReader(stream))


def standardize(features, training):
    scale = np.std(features[training], axis=0)
    return np.clip((features - features[training].mean(axis=0)) / np.where(scale > 1e-12, scale, 1), -3, 3)


def forecasts(data):
    """Use direct powers and positive kernels; compute PCA by a covariance eigendecomposition."""
    probabilities, names = [], []
    if data.native is not None:
        origin = (1 - EPSILON) * data.native + EPSILON / 5
        u = (GRADES - 3) / 2
        grid = sorted(itertools.product((.5, 1., 2.), (-1, 0, 1), (-1, 0, 1)),
                      key=lambda p: (np.log2(p[0])**2 + p[1]**2 + p[2]**2, p))
        for exponent, slope, curve in grid:
            p = origin**exponent * np.exp(slope*u + curve*u*u)
            probabilities.append(p / p.sum(axis=1, keepdims=True))
            names.append(f"native:t={exponent},b={slope},c={curve}")
    else:
        z = standardize(data.X, data.train)
        _, vectors = np.linalg.eigh(np.cov(z[data.train], rowvar=False))
        directions = vectors[:, -2:][:, ::-1].copy()
        for direction in directions.T:
            direction *= np.sign(direction[np.argmax(np.abs(direction))])
        features = np.column_stack([z, standardize(z @ directions, data.train)])
        labels = list(data.features) + ["PC1", "PC2"]
        grid = list(itertools.product(range(len(labels)), (1, 3, 5), (-1.5, -.75, .75, 1.5), (.65, 1.25)))
        reference = (0, 3, .75, 1.25)
        for feature, center, slope, width in [reference] + [p for p in grid if p != reference]:
            offset = (GRADES - center - slope*features[:, [feature]]) / width
            p = np.exp(-offset**2 / 2)
            probabilities.append((1 - EPSILON) * p / p.sum(axis=1, keepdims=True) + EPSILON / 5)
            names.append(f"{labels[feature]}:center={center},slope={slope},width={width}")
    probabilities.append(np.full_like(probabilities[0], .2))
    return np.stack(probabilities, axis=1), names + ["uniform"]


def grade_losses(p):
    """Integrate the definition of each metric against raw individual grade frequencies."""
    cdf = np.cumsum(p, axis=-1)[..., :-1]
    rps = np.stack([((cdf - (y <= THRESHOLDS))**2).mean(axis=-1) for y in GRADES], axis=-1)
    return rps, -np.log(p)


def predict_bins(means, training, order, picked, test, prior):
    boundaries = np.unique(np.quantile(means[training], [.25, .5, .75]))
    bins = np.sum(means[:, None] >= boundaries, axis=1)
    q = np.empty((len(test), 5))
    for b in range(len(boundaries) + 1):
        votes = picked[bins[order] == b].sum(axis=0) + 8*prior
        q[bins[test] == b] = votes / votes.sum()
    return q


def first_minimum(values):
    return np.flatnonzero(values <= np.min(values) + TOL)[0]


def check_sampling(data):
    """Replay item order and the first three votes without replacement under each frozen seed."""
    grade_indicators = np.eye(5, dtype=int)
    for repeat in range(64):
        item_rng = np.random.default_rng([20261121, 23, repeat])
        order = item_rng.permutation(data.train)
        assert len(np.unique(order)) == len(data.train)
        assert not np.isin(data.group[order], data.group[data.test]).any()
        vote_rng = np.random.default_rng([20261121, 23, repeat, 1])
        sampled = np.zeros_like(data.counts[order])
        for depth in (1, 2, 3):
            remaining = data.counts[order] - sampled
            ticket = vote_rng.integers(remaining.sum(axis=1))
            grade = (remaining.cumsum(axis=1) > ticket[:, None]).argmax(axis=1)
            sampled += grade_indicators[grade]
            assert (sampled <= data.counts[order]).all()
            if depth in (1, 3):
                recorded_order, recorded_votes = draw(data, repeat, depth)
                np.testing.assert_array_equal(recorded_order, order)
                np.testing.assert_array_equal(recorded_votes, sampled)


def verify_depth(data, p, names, frozen_means, out, depth):
    budgets = np.array([6*2**k for k in range(11) if 6*2**k <= len(data.train)])
    selected = {(int(r["draw"]), int(r["budget"])): r for r in read(out, "selections.csv") if int(r["depth"]) == depth}
    draw_scores = {(int(r["draw"]), int(r["budget"])): r for r in read(out, "draw_rps.csv") if int(r["depth"]) == depth}
    targets = data.counts[data.test] / data.counts[data.test].sum(axis=1, keepdims=True)
    train_rps, train_ce = grade_losses(p[data.train])
    test_rps, _ = grade_losses(p[data.test])
    fixed_test_loss = np.einsum("nhg,ng->nh", test_rps, targets)
    means = p @ GRADES
    train_index = {item: j for j, item in enumerate(data.train)}
    pooled = np.zeros((len(budgets), len(METHODS), len(data.test)))
    errors = []
    alternative = np.zeros((len(budgets), len(data.test)))
    max_alternate_draw_error = 0.0
    for rep in range(64):
        order, picked = draw(data, rep, depth)
        assert len(set(order)) == len(data.train) and np.all(picked <= data.counts[order])
        assert not np.isin(data.group[order], data.group[data.test]).any()
        train_rows = [train_index[i] for i in order[:budgets[-1] // depth]]
        hist = picked[:len(train_rows)] / depth
        # One expected individual-grade loss for each training item and candidate.
        ce = np.einsum("nhg,ng->nh", train_ce[train_rows], hist).cumsum(axis=0)
        rps = np.einsum("nhg,ng->nh", train_rps[train_rows], hist).cumsum(axis=0)
        y = hist @ GRADES
        for index, budget in enumerate(budgets):
            n = budget // depth
            rows = order[:n]
            ce_head, rps_head = first_minimum(ce[n-1] / n), first_minimum(rps[n-1] / n)
            x_center = means[rows] - means[rows].mean(axis=0)
            y_center = y[:n] - y[:n].mean()
            x_var = np.mean(x_center**2, axis=0)
            y_var = np.mean(y_center**2)
            valid = (x_var > TOL) & (y_var > TOL)
            correlations = np.full(len(names), -np.inf)
            if valid.any():
                correlations[valid] = (y_center @ x_center[:, valid]) / (n * np.sqrt(y_var * x_var[valid]))
                pearson_head = first_minimum(-correlations)
                if n == 2:
                    signs = np.sign(means[rows[1]] - means[rows[0]]) * np.sign(y[1] - y[0])
                    np.testing.assert_allclose(correlations[valid], signs[valid], rtol=0, atol=1e-12)
            else:
                pearson_head = 0
            saved = selected[rep, budget]
            fitted = [names[i] for i in (ce_head, pearson_head, rps_head)]
            archived = [saved[k] for k in ("ce", "pearson", "rps")]
            assert fitted == archived, (rep, depth, budget, fitted, archived)
            assert int(saved["pearson_undefined"]) == int(not valid.any())
            histogram = picked[:n].sum(axis=0) + .5
            histogram /= histogram.sum()
            # Quartile boundary equality is sensitive to last-bit differences for discrete G-Eval probabilities.
            # Use frozen mean representations for exact bin replay, and calculate the direct-power version too.
            q = predict_bins(frozen_means[:, pearson_head], data.train, rows, picked[:n], data.test, histogram)
            alternate_q = predict_bins(means[:, pearson_head], data.train, rows, picked[:n], data.test, histogram)
            alternate_loss = np.sum(grade_losses(alternate_q)[0] * targets, axis=1)
            alternative[index] += alternate_loss / 64
            calibrated_losses = np.sum(grade_losses(q)[0] * targets, axis=1)
            losses = np.array([fixed_test_loss[:, ce_head], fixed_test_loss[:, pearson_head], fixed_test_loss[:, -1],
                               targets @ grade_losses(histogram)[0], calibrated_losses,
                               fixed_test_loss[:, 0], fixed_test_loss[:, rps_head]])
            max_alternate_draw_error = max(max_alternate_draw_error, abs(float((alternate_loss - calibrated_losses).mean())))
            scores = draw_scores[rep, budget]
            errors.extend(np.abs(losses.mean(axis=1) - [float(scores[m]) for m in METHODS]))
            pooled[index] += losses / 64
    max_draw_error = max(errors)
    assert max_draw_error < 1e-10, dict(zip(METHODS, np.max(np.array(errors).reshape(-1, len(METHODS)), axis=0)))
    row = verify_bands(data, pooled, budgets, depth, out, max_draw_error)
    row['max_alt_quartile_draw_rps_shift'] = max_alternate_draw_error
    row['max_alt_quartile_pooled_rps_shift'] = float(np.abs((alternative - pooled[:,4]).mean(axis=1)).max())
    pooled[:,4] = alternative
    alt = verify_bands(data, pooled, budgets, depth, out, max_draw_error, check=False)
    row['alt_quartile_sustained_joint'] = alt['sustained_joint']
    row['alt_quartile_sustained_all_four'] = alt['sustained_against_all_four']
    return row


def verify_bands(data, scores, budgets, depth, out, max_draw_error, check=True):
    group = data.group[data.test]
    differences = scores[:, 1:5] - scores[:, [0]]
    grouped = np.stack([differences[:, :, group == key].mean(axis=2) for key in sorted(set(group))])
    n = len(grouped)
    ordered = grouped.reshape(n, -1)
    expected = ordered.mean(axis=0)
    rng = np.random.default_rng(20261123)
    replicates = []
    for _ in range(40):
        mult = rng.multinomial(n, np.full(n, 1/n), size=100)
        replicates.append((mult @ ordered) / n)
    replicates = np.vstack(replicates)
    scale = np.std(replicates, axis=0, ddof=1)
    active = scale > 1e-12
    maxima = np.max(np.abs(replicates[:, active] - expected[active]) / scale[active], axis=1)
    radius = np.sort(maxima)[int(np.ceil(.975 * (len(maxima)-1)))] * scale
    lower = (expected - radius).reshape(len(budgets), 4)
    upper = (expected + radius).reshape(len(budgets), 4)
    expected = expected.reshape(len(budgets), 4)
    recorded = {(int(r["budget"]), r["comparator"]): r for r in read(out, "contrasts.csv") if int(r["depth"]) == depth}
    error = []
    for i, b in enumerate(budgets):
        for j, name in enumerate(METHODS[1:5]):
            row = recorded[b, name]
            error.extend(np.abs([expected[i,j] - float(row["rps_gain"]), lower[i,j] - float(row["simultaneous_lo"]),
                                 upper[i,j] - float(row["simultaneous_hi"])]))
    if check:
        assert max(error) < 1e-10, max(error)
    passes = np.all(lower[:, :2] > 0, axis=1)
    deployment = np.all(lower > 0, axis=1)
    sustained = lambda flags: next((int(b) for i,b in enumerate(budgets) if flags[i:].all()), None)
    return dict(dataset=out.name, depth=depth, heldout_items=len(group), groups=n,
                budgets=len(budgets), training_draws=64, checked_method_draw_scores=len(budgets)*64*len(METHODS),
                checked_paired_contrasts=len(recorded), max_abs_draw_rps_error=max_draw_error,
                max_abs_point_or_band_error=max(error), jointly_passing_budgets=",".join(map(str, budgets[passes])),
                sustained_joint=sustained(passes), sustained_against_all_four=sustained(deployment))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "results/primary_verification.csv")
    parser.add_argument("--pair-out", type=Path, default=ROOT / "results/two_item_pearson_check.csv")
    args = parser.parse_args()
    reports = []
    for name, filename in (("gpt4_consistency", "summeval_consistency.npz"), ("image_quality", "koniq.npz")):
        data = load(args.data_directory / filename)
        check_sampling(data)
        if data.native is not None:
            ordered = sorted(set(data.group), key=lambda g: hashlib.sha256(f"corrce23-split|{g}".encode()).digest())
            assert set(data.group[data.test]) == set(ordered[:40])
        else:
            with np.load(args.data_directory / filename, allow_pickle=False) as source:
                assert np.all(source["split"][data.test] == "test") and np.all(source["split"][data.train] == "train")
        candidates, names = forecasts(data)
        frozen, frozen_names = frozen_candidates(data, "native" if data.native is not None else "features")
        assert names == frozen_names
        np.testing.assert_allclose(candidates, frozen, rtol=1e-12, atol=1e-12)
        frozen_means = frozen @ GRADES.astype(float)
        del frozen
        for depth in (1, 3):
            row = verify_depth(data, candidates, names, frozen_means, ROOT / "results" / name, depth)
            reports.append(row)
            print(row, flush=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=reports[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(reports)
    contrasts = verify_pairs(args.data_directory)
    with args.pair_out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=contrasts[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(contrasts)


def verify_pairs(directory):
    """Compare the former raw-moment computation with the exact two-point correlation."""
    results = []
    tracks = [
        ("gpt4_consistency", "summeval_consistency", "native"),
        ("gpt4_relevance", "summeval_relevance", "native"),
        ("image_quality", "koniq", "features"),
        ("summary_metrics", "summeval_consistency", "features"),
        ("jeffreys_consistency", "summeval_consistency", "smoothed"),
        ("jeffreys_relevance", "summeval_relevance", "smoothed"),
    ]
    for track, dataset, mode in tracks:
        path = directory / f"{dataset}.npz"
        data = load(path)
        if mode == "smoothed":
            with np.load(path) as archive:
                votes = archive["native_valid_votes"][:, None]
            data.native = (np.rint(data.native * votes) + .5) / (votes + 2.5)
        q, names = frozen_candidates(data, "native" if mode == "smoothed" else mode)
        means = q @ GRADES.astype(float)
        targets = data.counts[data.test] / data.counts[data.test].sum(axis=1, keepdims=True)
        test_losses = np.einsum("nhg,ng->nh", grade_losses(q[data.test])[0], targets).mean(axis=0)
        saved = read(ROOT / "results" / track, "selections.csv")
        selections = {int(r["draw"]): r for r in saved if r["depth"] == "3" and r["budget"] == "6"}
        changed = []
        differences = []
        for rep in range(64):
            order, votes = draw(data, rep, 3)
            x = means[order[:2]]
            y = (votes[:2] / 3) @ GRADES.astype(float)
            # Preserve the arithmetic and reduction order of the originally frozen raw-moment implementation.
            total = x.sum(axis=0)
            observed_ss = np.dot(y, y) - y.sum()**2 / 2
            predicted_ss = np.einsum("ih,ih->h", x, x) - total**2 / 2
            covariance = np.einsum("i,ih->h", y, x) - total*y.sum()/2
            eligible = (predicted_ss > 2*TOL) & (observed_ss > 2*TOL)
            old = 0
            if eligible.any():
                correlations = np.full(len(names), -np.inf)
                correlations[eligible] = covariance[eligible] / np.sqrt(observed_ss * predicted_ss[eligible])
                old = first_minimum(-correlations)
            delta_x = x[1] - x[0]
            delta_y = y[1] - y[0]
            eligible = ((delta_x / 2)**2 > TOL) & ((delta_y / 2)**2 > TOL)
            exact = 0
            if eligible.any():
                signs = np.where(eligible, np.sign(delta_x) * np.sign(delta_y), -np.inf)
                exact = np.argmax(signs)
            assert names[exact] == selections[rep]["pearson"], (track, rep)
            if old != exact:
                changed.append(rep)
            ce = names.index(selections[rep]["ce"])
            differences.append([test_losses[old] - test_losses[ce], test_losses[exact] - test_losses[ce]])
        old_gap, exact_gap = np.mean(differences, axis=0)
        results.append(dict(track=track, training_votes=6, training_items=2, training_draws=64,
                            changed_pearson_selections=len(changed), changed_draws=",".join(map(str, changed)),
                            old_test_rps_pearson_minus_ce=old_gap, exact_test_rps_pearson_minus_ce=exact_gap))
    return results


if __name__ == "__main__":
    main()
