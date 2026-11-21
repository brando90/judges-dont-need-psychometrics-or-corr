"""Reproduce the conditional-shape by annotation-budget sweep on a single CPU."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import itertools
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import kendalltau, rankdata

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results"
N_TRAIN = (8, 32, 128)
REPEATS = (1, 4, 16)
REPLICATIONS = 256
TEST_ARTIFACTS = 384
SHAPES = np.array([0, 1, -1])
RATING = np.arange(6)
MEAN_DIRECTION = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
SHAPE_DIRECTION = np.array([1, 0, -1, -1, 0, 1])
CANDIDATES = list(itertools.product(range(4), range(3)))


@dataclass
class Design:
    probability: np.ndarray  # mean curve × shape × artifact × rating level
    mean: np.ndarray  # mean curve × artifact
    uniforms: np.ndarray  # artifact × repeated draw

    def ratings(self, curve, shape_index):
        cdf = self.probability[curve, shape_index].cumsum(axis=-1)[:, :-1]
        return (self.uniforms[:, :, None] > cdf[:, None, :]).sum(axis=-1)


def design(rng, artifacts, repeats):
    x = rng.uniform(-np.pi, np.pi, size=artifacts)
    z = rng.choice([-1, 1], size=artifacts)
    signals = np.array([np.sin(x), np.cos(x), np.sin(2*x), np.cos(2*x)])
    tilts = 0.06 * signals[:, None, :, None] * MEAN_DIRECTION
    shapes = 0.08 * SHAPES[:, None, None] * z[:, None] * SHAPE_DIRECTION
    return Design(1 / 6 + tilts + shapes, 2.5 + 0.42 * signals,
                  rng.uniform(size=(artifacts, repeats)))


def histogram(ratings):
    return (ratings[..., None] == RATING).mean(axis=1)


def rps(q, observations):
    """Average individual-rating normalized RPS, integrating over each histogram."""
    forecast_cdf = np.cumsum(q, axis=-1)[..., :-1]
    observed_cdf = np.cumsum(observations, axis=-1)[..., :-1]
    return (forecast_cdf**2 + (1 - 2 * forecast_cdf) * observed_cdf).mean(axis=(-2, -1))


def pearson(a, b):
    u, v = a - a.mean(), b - b.mean()
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    return np.nan if norm == 0 else float(u @ v / norm)


def minimum_ties(loss):
    return np.flatnonzero(loss.ravel() <= np.min(loss) + 1e-12)


def fits(train, train_hist, val, val_hist, true_shape):
    n, val_n = len(train_hist), len(val_hist)
    train_ce = -(np.log(train.probability[:, :, :n]) * train_hist).sum(axis=-1).mean(axis=-1)
    ce_set = minimum_ties(train_ce)
    artifact_mean = train_hist @ RATING
    undefined = bool(np.ptp(artifact_mean) == 0)
    if undefined:
        pearson_set = np.arange(len(CANDIDATES))
    else:
        corr = np.array([pearson(m[:n], artifact_mean) for m in train.mean])
        pearson_set = minimum_ties(-np.repeat(corr, len(SHAPES)))
    val_loss = rps(val.probability[:, :, :val_n], val_hist).ravel()

    def val_choice(candidates):
        return candidates[minimum_ties(val_loss[candidates])[0]]

    best_mean = CANDIDATES[pearson_set[0]][0]
    choices = {
        "ce": ce_set[0], "pearson": pearson_set[0],
        "ce_val_rps": val_choice(ce_set), "pearson_val_rps": val_choice(pearson_set),
        "pearson_oracle_shape": CANDIDATES.index((best_mean, true_shape))
    }
    return choices, undefined, len(ce_set), len(pearson_set)


def distribution_metrics(q, obs, truth):
    return {
        "rps": float(rps(q, obs)),
        "ce": float(-(obs * np.log(q)).sum(axis=-1).mean()),
        "empirical_tvd": float(0.5 * np.abs(q - obs).sum(axis=-1).mean()),
        "truth_tvd": float(0.5 * np.abs(q - truth).sum(axis=-1).mean()),
    }


def mean_metrics(predicted, observed, true_mean, constant=False):
    error = predicted - observed
    return {
        "pearson_empirical": np.nan if constant else pearson(predicted, observed),
        "spearman_empirical": np.nan if constant else pearson(rankdata(predicted), rankdata(observed)),
        "kendall_empirical": np.nan if constant else float(kendalltau(predicted, observed).statistic),
        "pearson_truth": np.nan if constant else pearson(predicted, true_mean),
        "spearman_truth": np.nan if constant else pearson(rankdata(predicted), rankdata(true_mean)),
        "mean_absolute_error": float(np.abs(error).mean()),
        "signed_bias": float(error.mean()),
        "forecast_mean": float(predicted.mean()), "observed_mean": float(observed.mean()),
    }


def test_metrics(test, curve, shape, test_hist, obs_mean):
    truth = test.probability[curve, shape]
    mean_scores = [mean_metrics(m, obs_mean, test.mean[curve]) for m in test.mean]
    return [{**distribution_metrics(test.probability[g, s], test_hist, truth), **mean_scores[g]}
            for g, s in CANDIDATES]


def simulation(rep):
    rng = np.random.default_rng(np.random.SeedSequence([20261121, rep]))
    curve = int(rng.integers(4))
    varying_shape = int(rng.integers(1, 3))
    train = design(rng, max(N_TRAIN), max(REPEATS))
    val = design(rng, max(N_TRAIN) // 4, max(REPEATS))
    test = design(rng, TEST_ARTIFACTS, 3)
    for name, shape in (("neutral", 0), ("varying", varying_shape)):
        train_r, val_r = train.ratings(curve, shape), val.ratings(curve, shape)
        test_r = test.ratings(curve, shape)
        test_hist = histogram(test_r)
        # Direct integer totals preserve exact ties for midranks and Kendall tau-b.
        test_means = test_r.mean(axis=1)
        metrics = test_metrics(test, curve, shape, test_hist, test_means)
        true_index = CANDIDATES.index((curve, shape))
        for n, k in itertools.product(N_TRAIN, REPEATS):
            counts = np.bincount(train_r[:n, :k].ravel(), minlength=6)
            constant = (counts + 0.5) / (counts.sum() + 3)
            choices, undefined, ce_ties, pearson_ties = fits(
                train, histogram(train_r[:n, :k]), val, histogram(val_r[:n//4, :k]), shape)
            choices["oracle"] = true_index
            choices["constant"] = None
            for method, selected in choices.items():
                if selected is None:
                    m = np.full(TEST_ARTIFACTS, constant @ RATING)
                    scores = {**distribution_metrics(constant, test_hist, test.probability[curve, shape]),
                              **mean_metrics(m, test_means, test.mean[curve], constant=True)}
                    selected_curve = selected_shape = ""
                else:
                    scores = metrics[selected]
                    selected_curve, index = CANDIDATES[selected]
                    selected_shape = int(SHAPES[index])
                yield dict(rep=rep, scenario=name, n=n, k=k, method=method,
                           true_curve=curve, true_shape=int(SHAPES[shape]),
                           selected_curve=selected_curve, selected_shape=selected_shape,
                           train_pearson_undefined=int(undefined), ce_ties=ce_ties,
                           pearson_ties=pearson_ties, **scores)


def main():
    protocol_hash = hashlib.sha256((ROOT / "PROTOCOL.md").read_bytes()).hexdigest()
    expected_hash = (OUTPUT / "protocol.sha256").read_text().split()[0]
    if protocol_hash != expected_hash:
        raise RuntimeError("Protocol differs from the pre-measurement digest.")
    path = OUTPUT / "raw.csv.gz"
    temp = OUTPUT / "raw.partial.gz"
    with temp.open("wb") as file, gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as gz:
        with io.TextIOWrapper(gz, encoding="utf-8", newline="") as stream:
            writer = None
            count = 0
            for rep in range(REPLICATIONS):
                for row in simulation(rep):
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=row, lineterminator="\n")
                        writer.writeheader()
                    writer.writerow(row)
                    count += 1
    if count != REPLICATIONS * 2 * len(N_TRAIN) * len(REPEATS) * 7:
        raise RuntimeError(f"Incomplete run: {count} rows")
    temp.replace(path)
    receipt = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                   full_replications=REPLICATIONS, total_rows=count, protocol_sha256=protocol_hash,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   uncompressed_csv_sha256=hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest())
    (OUTPUT / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
