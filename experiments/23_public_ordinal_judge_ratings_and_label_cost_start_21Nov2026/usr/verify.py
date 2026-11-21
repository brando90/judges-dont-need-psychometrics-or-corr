"""Independent centered fits and individual-grade score replay for the dialogue extension."""
import argparse
import csv
from pathlib import Path

import numpy as np

from run import HERE, TRACKS, fetch, prepare
from design import draw
from verify_primary import (GRADES, TOL, check_sampling, first_minimum, forecasts, grade_losses,
                            predict_bins, verify_bands)


def saved(directory, name):
    with (directory / name).open() as stream:
        return list(csv.DictReader(stream))


def verify(data, path):
    # Rebuild the library by covariance eigendecomposition and normalize kernels directly.
    p, names = forecasts(data)
    mean = p @ GRADES
    train_rps, train_ce = grade_losses(p)
    normed = data.counts[data.test] / 3
    grade_weighted_rps = np.einsum("nhg,ng->nh", train_rps[data.test], normed)
    by_draw = saved(path, "draw_rps.csv")
    selections = saved(path, "selections.csv")
    score_rows = {(int(r["depth"]), int(r["draw"]), int(r["budget"])): r for r in by_draw}
    heads = {(int(r["depth"]), int(r["draw"]), int(r["budget"])): r for r in selections}
    budgets = sorted({int(r["budget"]) for r in by_draw})
    metrics = ("ce", "pearson", "uniform", "histogram", "calibrated_pearson", "original", "rps")
    matched_losses = np.zeros((len(budgets), 2, len(data.test), 3))
    check_sampling(data)
    for depth in (1, 3):
        pooled = np.zeros((len(budgets), 7, len(data.test)))
        peak_error = 0.
        for repeat in range(64):
            order, sample = draw(data, repeat, depth)
            for index, budget in enumerate(budgets):
                n = budget // depth
                ids = order[:n]
                votes = sample[:n]
                observed = votes / depth
                y = observed @ GRADES
                centered_y = y - y.mean()
                centered_x = mean[ids] - mean[ids].mean(axis=0)
                eligible = (centered_x.var(axis=0) > TOL) & (centered_y.var() > TOL)
                correlations = np.full(len(names), -np.inf)
                if eligible.any():
                    x = centered_x[:, eligible]
                    correlations[eligible] = (centered_y @ x)/(np.linalg.norm(centered_y) * np.linalg.norm(x, axis=0))
                    pearson = first_minimum(-correlations)
                else:
                    pearson = 0
                ce = first_minimum(np.einsum("nhg,ng->h", train_ce[ids], observed) / n)
                rps = first_minimum(np.einsum("nhg,ng->h", train_rps[ids], observed) / n)
                record = heads[depth, repeat, budget]
                assert all(names[j] == record[k] for j,k in [(ce,"ce"), (pearson,"pearson"), (rps,"rps")])
                prior = (votes.sum(axis=0) + .5) / (budget + 2.5)
                calibrated = predict_bins(mean[:, pearson], data.train, ids, votes, data.test, prior)
                if depth == 1:
                    calibrated_ce = predict_bins(mean[:, ce], data.train, ids, votes, data.test, prior)
                    for method, prediction in enumerate((calibrated_ce, calibrated)):
                        ordinal, log = grade_losses(prediction)
                        matched_losses[index, method, :, 0] += (normed * ordinal).sum(axis=1) / 64
                        matched_losses[index, method, :, 1] += (normed * log).sum(axis=1) / 64
                        matched_losses[index, method, :, 2] += abs(normed - prediction).sum(axis=1) / 128
                losses = np.array([grade_weighted_rps[:,ce], grade_weighted_rps[:,pearson], grade_weighted_rps[:,-1],
                                   normed @ grade_losses(prior)[0],
                                   (normed * grade_losses(calibrated)[0]).sum(axis=1),
                                   grade_weighted_rps[:,0], grade_weighted_rps[:,rps]])
                expected = score_rows[depth, repeat, budget]
                diff = losses.mean(axis=1) - [float(expected[k]) for k in metrics]
                peak_error = max(peak_error, float(np.max(np.abs(diff))))
                pooled[index] += losses/64
        if peak_error >= 1e-10:
            raise AssertionError((data.name, depth, peak_error))
        report = verify_bands(data, pooled, np.array(budgets), depth, path, peak_error)
        print(data.name, depth, report["checked_method_draw_scores"], report["checked_paired_contrasts"],
              report["sustained_joint"], report["sustained_against_all_four"],
              report["max_abs_draw_rps_error"], report["max_abs_point_or_band_error"], flush=True)
        yield report
    delta = matched_losses[:, 1] - matched_losses[:, 0]
    groups = data.group[data.test]
    grouped = np.stack([delta[:, groups == group].mean(axis=1) for group in sorted(set(groups))])
    n = len(grouped)
    rng = np.random.default_rng(20261124)
    bootstrap = np.vstack([(rng.multinomial(n, np.full(n, 1/n), 100) @ grouped.reshape(n, -1)) / n
                           for _ in range(40)])
    bands = np.quantile(bootstrap, [.025, .975], axis=0).reshape(2, len(budgets), 3)
    paired = {(int(r["budget"]), r["metric"]): r for r in saved(path, "symmetric_calibration.csv")}
    for i,budget in enumerate(budgets):
        for j,metric in enumerate(("rps", "ce", "tvd")):
            row = paired[budget, metric]
            observed = [float(row[c]) for c in ("calibrated_ce", "calibrated_pearson", "pearson_minus_ce",
                                                "paired_context_lower95", "paired_context_upper95")]
            expected = [*matched_losses[i, :, :, j].mean(axis=1), delta[i, :, j].mean(), *bands[:, i, j]]
            np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/corr-ce-public23/usr"))
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    fetch(args.cache)
    records = []
    for track in TRACKS:
        data, _ = prepare(args.cache, track)
        records.extend(verify(data, args.out / track))
    with (args.out / "verification.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
