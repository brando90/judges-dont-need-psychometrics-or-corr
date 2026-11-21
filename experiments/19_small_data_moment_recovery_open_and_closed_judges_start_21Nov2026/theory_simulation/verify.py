"""Independently reproduce histogram counts using ordered individual-rating sequences."""
import csv
import itertools
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
# Probabilities for R=1,2,3 times 10, written independently from the generator.
p = np.array([[[3,6,1],[2,6,2],[1,6,3]], [[5,2,3],[4,2,4],[3,2,5]],
              [[1,6,3],[3,6,1],[2,6,2]], [[3,2,5],[5,2,3],[4,2,4]]])
names = ["A_low", "A_high", "B_low", "B_high"]
means = (p @ np.array([1,2,3])) / 10
assert np.all(p.sum(-1) == 10)
with (ROOT / "exact_selection_mass.csv").open() as f:
    records = list(csv.DictReader(f))
assert len(records) == 8 * 4 * 3 * 2 * 2
for row in records:
    assert sum(int(row[tag]) for tag in names) == int(row["common_mass_denominator"])
    assert 0 <= int(row["undefined_pearson_mass"]) <= int(row["common_mass_denominator"])

for k in (1,2):
    n = 3*k
    x = np.repeat(np.arange(3), k)
    sequences = np.array(list(itertools.product(range(3), repeat=n)))
    probnum = p[:, x, sequences]
    mass = probnum.prod(-1, dtype=int)
    loglik = np.log(probnum / 10).sum(axis=-1)
    empirical = (sequences + 1).reshape(len(sequences), 3, k).mean(-1)
    rho = np.empty((4, len(sequences)))
    under = np.ptp(empirical, axis=-1) == 0
    for a, b in itertools.product(range(4), range(len(sequences))):
        rho[a,b] = 0 if under[b] else np.corrcoef(means[a], empirical[b])[0,1]
    for row in records:
        if int(row["total_train_ratings"]) != n:
            continue
        j = names.index(row["true_law"])
        order = [0,1,2,3] if row["tie_policy"]=="low_first" else [1,0,3,2]
        mode = row["known_information"]
        options = [a for a in order if (mode!="mean_known" or a//2==j//2) and
                   (mode!="spread_known" or a%2==j%2)]
        S = (rho if row["selector"]=="pearson" else loglik)[options]
        chosen = np.array(options)[(S >= S.max(axis=0)-1e-12).argmax(axis=0)]
        assert int(row["undefined_pearson_mass"]) == sum(mass[j,under])
        for a in range(4):
            assert int(row[names[a]]) == sum(mass[j,chosen==a])
print("For k=1,2 all 27 and 729 ordered data sets replayed with direct Pearson and log loss.")
with (ROOT / "population_risk.csv").open() as f:
    risks=list(csv.DictReader(f))
assert len(risks)==8*4*3*3*2
for r in risks:
    if r["known_information"]=="mean_known":
        assert float(r["mean_mse"]) == 0
        assert math.isclose(float(r["variance_mse"]),.16*(1-float(r["recover_spread"])),abs_tol=1e-13)
    if r["known_information"]=="spread_known":
        assert float(r["fixed_center_second_moment_mse"])==0
    assert math.isclose(float(r["mean_mse"]),.08*(1-float(r["recover_mean"])),abs_tol=1e-13)
    if r["tie_policy"]=="random_half" and r["known_information"]!="spread_known" and r["selector"]=="pearson":
        assert math.isclose(float(r["recover_spread"]),.5,abs_tol=1e-13)
print(f"All {len(records)} exact selection rows normalized, {len(risks)} population-risk rows checked.")

# Distributional bounds on each fully specified pair, independent of training size.
for truth, judge, x in itertools.product(range(4),range(4),range(3)):
    target = p[truth,x] / 10
    forecast = p[judge,x] / 10
    tv = abs(target - forecast).sum() / 2
    kl = np.sum(target * np.log(target / forecast))
    assert tv**2 <= kl / 2 + 1e-14
    for degree in range(1,6):
        f = np.arange(1,4, dtype=float)**degree
        assert abs((target-forecast) @ f) <= np.ptp(f) * tv + 1e-12
    var_p = target @ np.arange(1,4)**2 - (target @ np.arange(1,4))**2
    var_q = forecast @ np.arange(1,4)**2 - (forecast @ np.arange(1,4))**2
    assert abs(var_p - var_q) <= 4 * tv + 1e-14  # on the effective two-point-wide support
with (ROOT / "known_mean_prior_threshold.csv").open() as stream:
    boundary = next(csv.DictReader(stream))
    assert int(boundary["total_train_ratings"]) == 1
    for name, ref in (("ce_wrong_given_low", .4), ("ce_wrong_given_high", .2),
                      ("high_spread_prior_for_equal_risk", 1/3)):
        assert math.isclose(float(boundary[name]), ref, abs_tol=1e-15)
print("All 48 law-pair/input checks satisfy the total-variation, moment and KL bounds.")
print("One-label error rates 2/5 and 1/5 and prevalence threshold 1/3 independently verified.")
