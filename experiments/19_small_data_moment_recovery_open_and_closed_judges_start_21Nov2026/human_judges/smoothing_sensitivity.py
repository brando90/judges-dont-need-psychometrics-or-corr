"""Post hoc input-smoothing sensitivity on the two pairs at three families × one or two ratings.

Reproduce with OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 smoothing_sensitivity.py.
Normalize each original six-bin input p, then smooth it before calibration using
p_eps(r) = (1 - 6 * epsilon) * p(r) + epsilon. This mixes with the uniform
forecast with weight 6 * epsilon. The fitted output may fall below epsilon.

All rows use 250 overlapping splits, four held-out task families per split,
and 1,000 shared Bayesian family reweightings with full training reselection.
RPS (ranked probability score, 0–1) and cross-entropy (CE, nats) are
individual-rating losses, averaged within each artifact, then across artifacts
and splits. Positive paired differences favor CE fitting; intervals are pointwise
and conditional on this historical panel. Each replay refits in a temporary
directory. The only new output is smoothing_sensitivity.csv.
"""
import csv
import math
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

import analyze as study

DIRECTORY = Path(__file__).resolve().parent
EPSILONS = (1e-6, 1e-4, 1e-3, 1e-2)
PAIRS = ("open_pair", "closed_pair")
PUBLISHED = ("scores.csv", "contrasts.csv", "choices.csv", "accounting.csv")


def uniform_reference(items, ratings):
    """Mean individual-rating loss under q(r)=1/6, with the original artifact and split weights."""
    families = [item["family"] for item in items]
    thresholds = np.arange(5)
    uniform_cdf = (thresholds + 1) / 6
    losses = np.array([
        np.mean((uniform_cdf - (observations[:, None] <= thresholds))**2)
        for observations in ratings
    ])
    split_rps = [
        np.mean(losses[np.isin(families, list(test_families))])
        for _, test_families in study.history.splits(items)
    ]
    return {"rps": float(np.mean(split_rps)), "ce": math.log(6)}


def paired_rows(replay_dir, epsilon, uniform):
    with (replay_dir / "scores.csv").open() as stream:
        scores = {
            (row["input"], row["k"], row["method"], row["metric"]): float(row["mean"])
            for row in csv.DictReader(stream)
            if row["n"] == "3" and row["k"] in {"1", "2"} and row["input"] in PAIRS
            and row["metric"] in uniform
        }
    with (replay_dir / "contrasts.csv").open() as stream:
        for row in csv.DictReader(stream):
            pair, metric = row["input"], row["metric"]
            if (row["n"], row["method"], row["subtract"]) != ("3", "pearson", "ce"):
                continue
            if pair not in PAIRS or metric not in uniform:
                continue
            estimates = {
                method: scores[pair, row["k"], method, metric]
                for method in ("ce", "pearson", "identity", "constant")
            }
            delta = float(row["difference"])
            np.testing.assert_allclose(estimates["pearson"] - estimates["ce"], delta, atol=1e-12, rtol=0)
            yield {
                "input": pair, "epsilon": epsilon,
                "training_families": 3, "ratings_per_training_artifact": int(row["k"]), "test_metric": metric,
                "ce_fitted": estimates["ce"], "pearson_fitted": estimates["pearson"],
                "uncalibrated_input": estimates["identity"], "training_constant": estimates["constant"],
                "uniform": uniform[metric], "pearson_fitted_minus_ce_fitted": delta,
                "lower_95": row["lower_95"], "upper_95": row["upper_95"],
                "paired_family_weight_draws": row["draws"],
            }


def main():
    _, _, items, ratings, _ = study.load()
    uniform = uniform_reference(items, ratings)
    rows = []
    for epsilon in EPSILONS:
        with tempfile.TemporaryDirectory(prefix="judge-input-smoothing-") as workspace:
            replay_dir = Path(workspace)
            shutil.copyfile(DIRECTORY / "models.json", replay_dir / "models.json")
            # The frozen loader normalizes each row, then uses (1−6ε)p+ε before fitting.
            with patch.object(study, "ROOT", replay_dir), patch.object(study, "EPS", epsilon):
                study.run()
            if epsilon == 1e-6:
                for filename in PUBLISHED:
                    assert (replay_dir / filename).read_bytes() == (DIRECTORY / filename).read_bytes(), filename
            rows.extend(paired_rows(replay_dir, epsilon, uniform))
            print(f"Completed input epsilon {epsilon:g}", flush=True)
    assert len(rows) == len(EPSILONS) * 2 * len(PAIRS) * len(uniform)
    with (DIRECTORY / "smoothing_sensitivity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
