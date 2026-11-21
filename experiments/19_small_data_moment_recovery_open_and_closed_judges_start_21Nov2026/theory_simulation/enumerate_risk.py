"""Enumerate finite training histograms with exact integer likelihoods and masses."""
from itertools import product
from math import comb
from pathlib import Path
import csv

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parent
CURVES = np.array([[-1, 0, 1], [1, -1, 0]])  # conditional mean of R minus 2, times 5
WORLDS = tuple(product(range(2), range(2)))  # mean A/B, spread low/high
NAMES = ("A_low", "A_high", "B_low", "B_high")
SETTINGS = ("joint", "mean_known", "spread_known")
ORDERS = {"low_first": (0, 1, 2, 3), "high_first": (1, 0, 3, 2)}
# Integer numerators of P(R = 1, 2, 3 | X); denominator ten.
P10 = np.array([[[2 + 2*s - m, 6 - 4*s, 2 + 2*s + m] for m in CURVES[c]]
                for c, s in WORLDS])
SIZES = (1, 2, 3, 4, 6, 8, 12)


def write_csv(filename, rows):
    with (ROOT / filename).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def candidates(truth, setting, order):
    return [j for j in ORDERS[order] if
            (setting != "mean_known" or WORLDS[j][0] == WORLDS[truth][0]) and
            (setting != "spread_known" or WORLDS[j][1] == WORLDS[truth][1])]


def totals_template():
    return {(j, setting, order, objective): [0] * len(WORLDS)
            for j, setting, order, objective in product(range(4), SETTINGS, ORDERS, ("ce", "pearson"))}


def accumulate(totals, likelihoods, pearson, multiplicity):
    """Maximum over a finite ordered candidate set. Every likelihood is an integer."""
    mass = likelihoods * multiplicity
    # Fixed set and priority determine winners, so share work between true laws.
    winners = {}
    for (j, setting, order, objective), counts in totals.items():
        allowed = tuple(candidates(j, setting, order))
        key = objective, allowed
        if key not in winners:
            score = likelihoods if objective == "ce" else pearson
            winners[key] = np.asarray(allowed)[np.argmax(score[list(allowed)], axis=0)]
        selected = winners[key]
        for choice in allowed:
            counts[choice] += int(mass[j][selected == choice].sum())


def one_label():
    # The 3 input choices × 3 outcomes have probability P10/30. With one
    # observation there is no defined sample Pearson coefficient.
    obs = P10.reshape(4, 9).astype(object)
    totals = totals_template()
    accumulate(totals, obs, np.zeros_like(obs, dtype=int), 1)
    return totals, [30] * 4, 30, 9


def balanced(k):
    # For each of the 3 input types a trinomial histogram replaces 3**k sequences.
    hist = [(left, k - right - left, right) for left in range(k + 1) for right in range(k - left + 1)]
    likelihood = np.array([[[int(a)**l * int(b)**c * int(d)**h for l, c, h in hist]
                            for a, b, d in law] for law in P10], dtype=object)
    counts = np.array([comb(k, l) * comb(k-l, h) for l, _, h in hist], dtype=object)
    signed = np.array([h - l for l, _, h in hist])
    yz_likelihood = likelihood[:, 1, :, None] * likelihood[:, 2, None, :]
    yz_counts = counts[:, None] * counts[None, :]
    yz_scores = np.array([CURVES[c, 1] * signed[:, None] + CURVES[c, 2] * signed[None, :]
                          for c, _ in WORLDS])
    totals = totals_template()
    undefined = [0] * 4
    for i, d in enumerate(signed):
        lik = likelihood[:, 0, i, None, None] * yz_likelihood
        multiplicity = counts[i] * yz_counts
        score = yz_scores + np.array([CURVES[c, 0] * d for c, _ in WORLDS])[:, None, None]
        # Both mean vectors are centered and share their squared norm. When all
        # observed means are equal both integer dot products are zero as required.
        const = (signed[:, None] == d) & (signed[None, :] == d)
        assert np.all(score[:, const] == 0)
        for j in range(4):
            undefined[j] += int((lik[j][const] * multiplicity[const]).sum())
        accumulate(totals, lik, score, multiplicity)
    return totals, undefined, 10**(3*k), len(hist)**3


def population_scores():
    # Compute targets on all three inputs and all six categories of the project scale.
    ratings = np.arange(6)
    q = np.zeros((4, 3, 6))
    q[:, :, 1:4] = P10 / 10
    mean = q @ ratings
    second = q @ (ratings**2)
    variance = second - mean**2
    spread = q @ ((ratings - 2)**2)
    cdf = q.cumsum(axis=-1)[:, :, :-1]
    scores = {}
    for j, h in product(range(4), repeat=2):
        F, Q = cdf[j], cdf[h]
        metrics = {
            "mean_mse": np.mean((mean[j] - mean[h])**2),
            "second_moment_mse": np.mean((second[j] - second[h])**2),
            "variance_mse": np.mean((variance[j] - variance[h])**2),
            "fixed_center_second_moment_mse": np.mean((spread[j] - spread[h])**2),
            "rps": np.mean((Q - F)**2 + F*(1 - F)),
            "ce_nats": -np.mean(np.sum(q[j, :, 1:4] * np.log(q[h, :, 1:4]), axis=-1)),
            "population_tvd": np.mean(np.sum(abs(q[j] - q[h]), axis=-1) / 2),
            "mean_pearson": np.corrcoef(mean[j], mean[h])[0, 1],
            "mean_spearman": np.corrcoef(rankdata(mean[j]), rankdata(mean[h]))[0, 1],
            "recover_mean": float(WORLDS[j][0] == WORLDS[h][0]),
            "recover_spread": float(WORLDS[j][1] == WORLDS[h][1]),
            "recover_variance_function": float(j == h),
        }
        # Enforce exact invariances only where mathematical equality is known;
        # tiny binary-roundoff artifacts would otherwise masquerade as positive MSE.
        if WORLDS[j][0] == WORLDS[h][0]:
            metrics["mean_mse"] = 0.0
        if WORLDS[j][1] == WORLDS[h][1]:
            metrics["fixed_center_second_moment_mse"] = 0.0
        scores[j, h] = metrics
    return scores


def summarize(total_n, k, counts, denominator, undefined, patterns, scores):
    cells = []
    for j, setting, order, objective in counts:
        mass = counts[j, setting, order, objective]
        assert all(x >= 0 for x in mass) and sum(mass) == denominator
        cells.append({"total_train_ratings":total_n, "ratings_per_type":k, "true_law":NAMES[j],
                      "known_information":setting, "tie_policy":order, "selector":objective,
                      "histogram_patterns":patterns, "common_mass_denominator":denominator,
                      "undefined_pearson_mass":undefined[j], **dict(zip(NAMES, mass))})
    risk = []
    for j, setting, objective in product(range(4), SETTINGS, ("ce","pearson")):
        low = np.array(counts[j,setting,"low_first",objective],dtype=object)
        high = np.array(counts[j,setting,"high_first",objective],dtype=object)
        for order, freq, den in (("low_first",low,denominator), ("high_first",high,denominator),
                                 ("random_half",low+high,2*denominator)):
            weights = np.array([int(x)/den for x in freq])
            row = {"total_train_ratings":total_n, "ratings_per_type":k, "true_law":NAMES[j],
                   "known_information":setting, "tie_policy":order, "selector":objective,
                   "undefined_train_pearson_prob":undefined[j]/denominator}
            for key in scores[j,j]:
                row[key] = float(weights @ [scores[j,h][key] for h in range(4)])
            risk.append(row)
    return cells, risk


def main():
    assert np.all(P10.sum(axis=-1) == 10) and np.all(P10 > 0)
    score = population_scores()
    exact, risk, thresholds = [], [], []
    for k in (0, *SIZES):
        n = 3*k if k else 1
        cnt, undefined, den, pat = balanced(k) if k else one_label()
        cells, risks = summarize(n, k or "one_random_type", cnt, den, undefined, pat, score)
        exact.extend(cells)
        risk.extend(risks)
        # Equal prior on mean curves, varying prior on high spread. Known curve
        # makes spread misclassification exactly proportional to variance MSE.
        rates = {}
        for shape, which_error in ((0, [1,3]), (1,[0,2])):
            misclassified = 0
            for curve in (0,1):
                j = 2*curve + shape
                misclassified += sum(cnt[j,"mean_known","low_first","ce"][s] for s in which_error)
            rates[shape] = misclassified / (2*den)
        e_low, e_high = rates[0], rates[1]
        thresholds.append({"total_train_ratings":n, "ce_wrong_given_low":e_low,
                           "ce_wrong_given_high":e_high,
                           "high_spread_prior_for_equal_risk":e_low/(1+e_low-e_high)})
        print(f"N={n}, {pat:,} histograms. Exact mass {den}.", flush=True)
    write_csv("exact_selection_mass.csv",exact)
    write_csv("population_risk.csv",risk)
    write_csv("known_mean_prior_threshold.csv",thresholds)


if __name__ == "__main__":
    main()
