"""Enumerate all states and rating outcomes of the notebook's input-subset example."""
import csv
import math
from fractions import Fraction as F
from itertools import accumulate, product
from pathlib import Path
from statistics import mean
import sys


STATES = tuple(product((-1, 1), repeat=3))
RATINGS = range(6)
SUBSETS = {
    "x,c,g": (0, 1, 2),
    "x,c": (0, 1),
    "x": (0,),
    "c,g": (1, 2),
    "none (constant)": (),
}
MEAN_DIRECTION = tuple(F(2 * r - 5, 5) for r in RATINGS)
SHAPE_DIRECTION = (1, 0, -1, -1, 0, 1)


def distribution(state):
    """x is the task, c the candidate, g the reference. Each is a fair sign."""
    x, c, g = state
    signal = F(45 * x + 35 * x * c, 1000)
    shape = F(6 * c * g, 100)
    return tuple(F(1, 6) + signal * u + shape * v
                 for u, v in zip(MEAN_DIRECTION, SHAPE_DIRECTION))


def conditional_forecast(state, observed, truth):
    compatible = [probs for other, probs in truth.items()
                  if all(other[i] == state[i] for i in observed)]
    return tuple(mean(p[r] for p in compatible) for r in RATINGS)


def rps(q, rating):
    return mean((cdf - int(rating <= cut)) ** 2
                for cut, cdf in enumerate(tuple(accumulate(q))[:-1]))


def pearson(xs, ys):
    dx = [x - mean(xs) for x in xs]
    dy = [y - mean(ys) for y in ys]
    ssx, ssy = sum(x*x for x in dx), sum(y*y for y in dy)
    if ssx == 0 or ssy == 0:
        return None
    return float(sum(x*y for x, y in zip(dx, dy))) / math.sqrt(float(ssx * ssy))


def mean_rating(p):
    return sum(r * p[r] for r in RATINGS)


def run():
    truth = {s: distribution(s) for s in STATES}
    target_means = [mean_rating(p) for p in truth.values()]
    assert all(min(p) > 0 and sum(p) == 1 for p in truth.values())
    assert sum(SHAPE_DIRECTION) == mean_rating(SHAPE_DIRECTION) == 0
    assert mean_rating(MEAN_DIRECTION) == 7
    rows = {}
    for label, subset in SUBSETS.items():
        ce, scores, distances, means, cumulative_gaps = [], [], [], [], []
        for state, p in truth.items():
            q = conditional_forecast(state, subset, truth)
            assert min(q) > 0 and sum(q) == 1
            ce.append(-sum(float(p[r]) * math.log(float(q[r])) for r in RATINGS))
            scores.append(sum(p[r] * rps(q, r) for r in RATINGS))
            distances.append(sum(abs(a - b) for a, b in zip(q, p)) / 2)
            means.append(mean_rating(q))
            cumulative_gaps.append(mean((a-b)**2 for a, b in zip(
                tuple(accumulate(q))[:-1], tuple(accumulate(p))[:-1])))
        rows[label] = (mean(ce), mean(scores), mean(distances), pearson(means, target_means))
        # Independently check the exact excess RPS identity for every input mask.
        assert rows[label][1] - rows["x,c,g"][1] == mean(cumulative_gaps)

    assert rows["x,c,g"][2] == 0
    assert math.isclose(rows["x,c"][3], 1.0)
    assert rows["x,c"][2] == F(12, 100)
    assert rows["x,c"][1] - rows["x,c,g"][1] == F(36, 12500)
    assert rows["c,g"][3] is None and rows["none (constant)"][3] is None
    assert rows["c,g"][0] < rows["none (constant)"][0]
    assert rows["c,g"][1] < rows["none (constant)"][1]
    assert math.isclose(rows["x"][3], 45 / math.sqrt(45**2 + 35**2))
    return rows


def write(rows, file):
    writer = csv.writer(file, lineterminator="\n")
    writer.writerow(("visible_inputs", "population_ce_nats", "population_rps",
                     "population_tvd_to_full_truth", "pearson_vs_full_means"))
    for label, values in rows.items():
        writer.writerow([label] + ["undefined" if v is None else f"{float(v):.9f}" for v in values])


if __name__ == "__main__":
    results = run()
    write(results, sys.stdout)
    with Path(__file__).with_name("population_scores.csv").open("w", newline="") as f:
        write(results, f)
