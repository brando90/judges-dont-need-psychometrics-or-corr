"""Explore all prespecified pseudocounts using the original fixed-grid methods and test article bootstrap."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from analyze import COMPARATORS, METHODS, METRICS, REPEATS, bands, save
from candidates import R, best, bin_distribution, bin_indices, build, loss_by_grade, pearson_from_totals, per_item
from design import draw, load

ALPHAS = (0, .01, .05, .1, .25, .5, 1, 2)
BUDGETS = (6, 12, 24, 48, 96, 192, 384, 768)


def original_results(criterion, alpha):
    prefix = "gpt4" if alpha == 0 else "jeffreys"
    folder = HERE.parent / "results" / (prefix + "_" + criterion)
    def read(filename):
        with (folder / filename).open() as f:
            return [r for r in csv.DictReader(f) if r["depth"] == "1"]
    return ({(int(r["budget"]), r["method"]): r for r in read("scores.csv")},
            {(int(r["budget"]), r["comparator"]): r for r in read("contrasts.csv")},
            {(int(r["draw"]), int(r["budget"])): r for r in read("selections.csv")})


def run(data, votes, criterion, alpha):
    data.native = (votes + alpha) / (votes.sum(axis=1, keepdims=True) + 5*alpha)
    q, names = build(data, "native")
    means = q @ R
    qlog, ordinal = loss_by_grade(q)
    actual = data.counts[data.test] / data.counts[data.test].sum(axis=1, keepdims=True)
    head_loss = np.stack([per_item(p, actual) for p in q[data.test].transpose(1, 0, 2)])
    partitions = [bin_indices(m[data.train], m) for m in means.T]
    cube = np.zeros((len(BUDGETS), len(METHODS), len(data.test), len(METRICS)))
    anchor_scores, anchor_contrasts, anchor_heads = original_results(criterion,alpha) if alpha in (0,.5) else ({},{},{})
    for rep in range(REPEATS):
        order, picked = draw(data, rep, 1)
        realized = picked.argmax(axis=1)
        cum_log = qlog[order, realized].cumsum(axis=0)
        cum_rps = ordinal[order, realized].cumsum(axis=0)
        for j,budget in enumerate(BUDGETS):
            tr = order[:budget]
            y = realized[:budget] + 1.
            m = means[tr]
            ce = best(cum_log[budget-1] / budget)
            rps = best(cum_rps[budget-1] / budget)
            pearson, undefined = pearson_from_totals(m.sum(axis=0), np.einsum("ih,ih->h",m,m),
                                                      y @ m, y.sum(), np.dot(y,y), budget)
            if anchor_heads:
                gold = anchor_heads[rep,budget]
                assert [names[i] for i in (ce,pearson,rps)] == [gold[k] for k in ("ce","pearson","rps")]
                assert undefined == bool(int(gold["pearson_undefined"]))
            prior = (picked[:budget].sum(axis=0) + .5) / (budget+2.5)
            histogram = per_item(np.broadcast_to(prior, actual.shape), actual)
            bins = partitions[pearson]
            posterior = bin_distribution(bins,tr,picked[:budget],prior)
            calibrated = per_item(posterior[bins[data.test]],actual)
            losses = [head_loss[ce],head_loss[pearson],head_loss[-1],histogram,calibrated,head_loss[0],head_loss[rps]]
            cube[j] += np.stack(losses)
    cube /= REPEATS
    point, lo, hi, point_lo, point_hi, pvals = bands(cube,data.group[data.test])
    summary, contrasts = [], []
    for j,budget in enumerate(BUDGETS):
        for mi,method in enumerate(METHODS):
            row = dict(criterion=criterion, alpha=alpha, budget=budget, method=method,
                       test_summaries=len(data.test), test_articles=len(np.unique(data.group[data.test])))
            row.update(zip(METRICS, cube[j,mi].mean(axis=0)))
            if anchor_scores:
                ref = anchor_scores[budget,method]
                np.testing.assert_allclose([row[k] for k in METRICS], [float(ref[k]) for k in METRICS],
                                           atol=1e-12,rtol=0)
            summary.append(row)
        for ci,comparator in enumerate(COMPARATORS):
            row = dict(criterion=criterion,alpha=alpha,budget=budget,comparator=comparator,subtract="ce",
                       rps_gain=point[j,ci],simultaneous_lo=lo[j,ci],simultaneous_hi=hi[j,ci],
                       pointwise_lo=point_lo[j,ci],pointwise_hi=point_hi[j,ci],within_setting_max_t_p=pvals[j,ci])
            if anchor_contrasts:
                ref = anchor_contrasts[budget,comparator]
                keys = ("rps_gain","simultaneous_lo","simultaneous_hi","pointwise_lo","pointwise_hi")
                np.testing.assert_allclose([row[k] for k in keys],[float(ref[k]) for k in keys],
                                           atol=1e-12,rtol=0)
                assert row["within_setting_max_t_p"] == float(ref["within_benchmark_max_t_p"])
            contrasts.append(row)
    threshold = dict(criterion=criterion,alpha=alpha)
    tests = {name: lo[:,i] > 0 for i,name in enumerate(COMPARATORS)}
    tests["pearson_and_uniform"] = (lo[:,:2] > 0).all(axis=1)
    tests["all_four"] = (lo > 0).all(axis=1)
    for name, passed in tests.items():
        threshold[name] = next((b for i,b in enumerate(BUDGETS) if passed[i:].all()), None)
    print(criterion, alpha, threshold, flush=True)
    return summary, contrasts, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory",type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    scores, contrasts, thresholds = [], [], []
    for criterion in ("consistency","relevance"):
        file = args.data_directory / f"summeval_{criterion}.npz"
        data = load(file)
        with np.load(file) as source:
            valid = source["native_valid_votes"][:,None]
        votes = np.rint(data.native * valid)
        np.testing.assert_array_equal(votes.sum(axis=1,keepdims=True), valid)
        for alpha in ALPHAS:
            absolute, paired, first = run(data,votes,criterion,alpha)
            scores.extend(absolute)
            contrasts.extend(paired)
            thresholds.append(first)
    args.out.mkdir(exist_ok=True,parents=True)
    save(args.out / "scores.csv",scores)
    save(args.out / "rps_contrasts.csv",contrasts)
    save(args.out / "sustained_thresholds.csv",thresholds)


if __name__ == "__main__":
    main()
