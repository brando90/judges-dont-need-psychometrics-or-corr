"""Independent scalar loss and fit reconstruction plus complete-output checks."""
import csv
import gzip
import hashlib
import json

import numpy as np
from scipy.stats import pearsonr, spearmanr

from run import (CANDIDATES, N_TRAIN, OUTPUT, RATING, REPEATS, REPLICATIONS, SHAPES,
                 design, fits, histogram, rps, simulation)


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def verify():
    rng = np.random.default_rng(2701)
    train = design(rng, 16, 4)
    val = design(rng, 4, 4)
    ratings = train.ratings(2, 1)
    hist = histogram(ratings)
    vhist = histogram(val.ratings(2, 1))
    close(train.probability.sum(axis=-1), 1)
    close(train.probability @ RATING, np.broadcast_to(train.mean[:, None], (4, 3, 16)))
    assert train.probability.min() > 0
    ce, pearson, rps_validation = [], [], []
    for curve, shape in CANDIDATES:
        q = train.probability[curve, shape]
        ce.append(np.mean([-np.log(q[i, y]) for i, row in enumerate(ratings) for y in row]))
        pearson.append(pearsonr(q @ RATING, ratings.mean(axis=1)).statistic)
        individual = [np.mean((q[i].cumsum()[:-1] - (y <= RATING[:-1]))**2)
                      for i, row in enumerate(ratings) for y in row]
        close(rps(q, hist), np.mean(individual))
        qv = val.probability[curve, shape]
        rps_validation.append(np.mean([np.mean((qv[i].cumsum()[:-1] - (y <= RATING[:-1]))**2)
                                       for i, row in enumerate(val.ratings(2, 1)) for y in row]))
    choices, undefined, _, size = fits(train, hist, val, vhist, 1)
    assert not undefined and size == 3
    assert choices["ce"] == np.argmin(ce)
    tied = np.flatnonzero(np.abs(pearson - np.max(pearson)) < 1e-12)
    assert choices["pearson"] == tied[0]
    assert choices["pearson_val_rps"] == tied[np.argmin(np.array(rps_validation)[tied])]
    truth = train.probability[2, 1]
    close(rps(train.probability[2, 0], truth) - rps(truth, truth), .00512)
    _, undefined, _, size = fits(train, np.tile([1, 0, 0, 0, 0, 0], (16, 1)), val, vhist, 1)
    assert undefined and size == 12

    path = OUTPUT / "raw.csv.gz"
    if not path.exists():
        print("Pre-run checks passed: direct individual scoring, exact fit, shape invariance, undefined fallback.")
        return
    with gzip.open(path, "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    unique = {(r["rep"], r["scenario"], r["n"], r["k"], r["method"]) for r in rows}
    assert len(rows) == len(unique) == REPLICATIONS*2*len(N_TRAIN)*len(REPEATS)*7
    for row in rows:
        assert 0 <= float(row["rps"]) <= 1 and float(row["ce"]) >= 0
        assert 0 <= float(row["empirical_tvd"]) <= 1 and 0 <= float(row["truth_tvd"]) <= 1
        if row["method"] == "oracle":
            close(float(row["truth_tvd"]), 0)
        if row["method"] == "constant":
            assert np.isnan(float(row["pearson_empirical"]))
    for index in (0, REPLICATIONS - 1):
        saved = [r for r in rows if int(r["rep"]) == index]
        for actual, regenerated in zip(saved, simulation(index), strict=True):
            assert actual == {key:str(value) for key, value in regenerated.items()}
    # Audit the tie-aware rank metric against raw integer-rating totals.
    regen = np.random.default_rng(np.random.SeedSequence([20261121, 0]))
    c = int(regen.integers(4))
    regen.integers(1, 3)
    design(regen, 128, 16)
    design(regen, 32, 16)
    test = design(regen, 384, 3)
    total = test.ratings(c, 0).sum(axis=1)
    oracle = next(r for r in rows if r["method"] == "oracle" and r["scenario"] == "neutral")
    close(float(oracle["spearman_empirical"]), spearmanr(test.mean[c], total).statistic)
    raw_hash = hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()
    assert raw_hash == json.loads((OUTPUT / "run.json").read_text())["uncompressed_csv_sha256"]
    print(f"Complete grid and all metrics checked, first and last replications regenerated; {len(rows)} rows.")


if __name__ == "__main__":
    verify()
