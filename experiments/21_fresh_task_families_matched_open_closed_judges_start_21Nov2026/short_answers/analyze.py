"""Run the frozen homework-family comparison on saved prompt-matched forecasts."""
import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, t as student_t

from metrics import COEFFICIENTS, GRADE, METRICS, TOL, calibrations, rating_histograms, score

JUDGES = ("qwen", "claude")
METHODS = ("ce_fit", "pearson_fit", "identity", "uniform", "train_hist")
BUDGETS = (4, 8, 16, 32, 64, 128, 256)
REPEATS = 200
SIGNS = np.array(list(itertools.product((-1, 1), repeat=10)))


def write_csv(out, name, rows):
    with (out / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def schedule(family, held_out, repeat):
    rng = np.random.default_rng(np.random.SeedSequence([20261121, int(held_out), repeat]))
    groups = [rng.permutation(np.flatnonzero(family == a)) for a in np.unique(family) if a != held_out]
    assert len(groups) == 9 and all(len(g) == 16 for g in groups)
    order = [groups[group][position] for position in range(16) for group in rng.permutation(9)]
    graders = rng.integers(0, 2, size=len(order))
    return np.array(order), graders


def select(q, h, y):
    """Exhaustive search, keeping the earliest closest-to-identity tie."""
    loss = -(np.log(q) * h[:, None, None, :]).sum(axis=-1).mean(axis=0)
    ce = (loss <= loss.min(axis=-1, keepdims=True) + TOL).argmax(axis=-1)
    m = q @ GRADE
    centered = m - m.mean(axis=0)
    target = y - y.mean()
    norm = np.linalg.norm(centered, axis=0) * np.linalg.norm(target)
    valid = (np.ptp(m, axis=0) > TOL) & (np.ptp(y) > TOL)
    rho = np.divide((centered * target[:, None, None]).sum(axis=0), norm,
                    out=np.full_like(norm, -np.inf), where=valid & (norm > 0))
    pearson = (rho >= rho.max(axis=-1, keepdims=True) - TOL).argmax(axis=-1)
    pearson[~valid.any(axis=-1)] = 0
    return ce, pearson, ~valid.any(axis=-1)


def simple_corr(a, b):
    x, y = a-a.mean(), b-b.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return np.dot(x, y) / den if np.ptp(a) > TOL and np.ptp(b) > TOL else np.nan


def diagnostics(q, target):
    means = q @ GRADE
    return np.array([simple_corr(means, target), simple_corr(rankdata(means), rankdata(target)),
                     means.mean(), target.mean(), (means-target).mean(), np.abs(means-target).mean()])


def interval(values):
    """Nadeau–Bengio corrected fold t interval (approximate; ten assignments)."""
    assert len(values) == 10
    point = float(np.mean(values))
    se = float(np.std(values, ddof=1)) * np.sqrt(1 / 10 + 1 / 9)
    radius = student_t.ppf(.975, 9) * se
    if se == 0:
        pvalue = float(point <= 0)
    else:
        pvalue = float(student_t.sf(point / se, 9))
    return point, point-radius, point+radius, pvalue


def run(manifest, label_file, forecast_file, out):
    items = [json.loads(l) for l in manifest.read_bytes().splitlines()]
    by_id = {r["id"]: r["ratings"] for r in map(json.loads, label_file.read_bytes().splitlines())}
    assert len(by_id) == len(items) == 160 and set(by_id) == {r["id"] for r in items}
    with forecast_file.open() as f:
        rows = list(csv.DictReader(f))
    forecasts = {(r["id"], r["judge"]): [float(r[f"p{k}"]) for k in GRADE] for r in rows}
    assert len(rows) == len(forecasts) == 320
    base = np.array([[forecasts[item["id"], judge] for judge in JUDGES] for item in items])
    assert np.isfinite(base).all() and base.min() >= 0
    np.testing.assert_allclose(base.sum(axis=-1), 1, atol=1e-12)
    q = calibrations(base)
    ratings = np.array([by_id[r["id"]] for r in items])
    assert ratings.shape == (160, 2) and (ratings >= 0).all() and (ratings <= 5).all()
    family = np.array([r["assignment"] for r in items])
    test_truth = rating_histograms(ratings)
    test_losses = score(q, test_truth[:, None, None, :])
    # [draw, judge, method, metric] for each held-out fold and exact rating budget.
    by_fold, diag_rows, counts, undefined = {}, [], Counter(), Counter()
    for held in np.unique(family):
        mask = family == held
        test_hist = test_truth[mask]
        test_mean = test_hist @ GRADE
        loss = test_losses[mask].mean(axis=0)
        diag = np.array([[diagnostics(q[mask, j, head], test_mean) for head in range(27)] for j in range(2)])
        fold = defaultdict(list)
        fold_diagnostics = defaultdict(list)
        uniform = np.full(6, 1 / 6)
        uniform_loss = score(uniform, test_hist).mean(axis=0)
        for rep in range(REPEATS):
            order, graders = schedule(family, held, rep)
            assert len(set(order)) == 144 and not np.any(family[order] == held)
            for k in (1, 2):
                labels = ratings[order, graders][:, None] if k == 1 else ratings[order]
                h = rating_histograms(labels)
                for budget in BUDGETS:
                    n = budget // k
                    if n > len(order):
                        continue
                    chosen = select(q[order[:n]], h[:n], h[:n] @ GRADE)
                    hist = (k*h[:n].sum(axis=0) + .5) / (budget + 3)
                    hist_loss = score(hist, test_hist).mean(axis=0)
                    pair = []
                    diags = []
                    for j in range(2):
                        heads = [int(chosen[0][j]), int(chosen[1][j]), 0]
                        for method, head in zip(METHODS[:2], heads):
                            counts[k, budget, JUDGES[j], method, head] += 1
                        undefined[k, budget, JUDGES[j]] += int(chosen[2][j])
                        pair.append([*loss[j, heads], uniform_loss, hist_loss])
                        # Constants have undefined correlations; their scalar agreement is still defined.
                        constant_diags = [[np.nan, np.nan, p@GRADE, test_mean.mean(),
                                           p@GRADE-test_mean.mean(), np.abs(p@GRADE-test_mean).mean()]
                                          for p in (uniform, hist)]
                        diags.append([*diag[j, heads], *constant_diags])
                    fold[k, budget].append(pair)
                    fold_diagnostics[k, budget].append(diags)
        for key, arrays in fold.items():
            by_fold[held, *key] = np.asarray(arrays)
            ds = np.array(fold_diagnostics[key])
            for j, judge in enumerate(JUDGES):
                for m, method in enumerate(METHODS):
                    row = dict(held_assignment=held, ratings_per_answer=key[0], training_ratings=key[1],
                               judge=judge, method=method)
                    for v, name in enumerate(("pearson", "spearman", "forecast_mean", "human_mean", "bias", "mae")):
                        values = ds[:, j, m, v]
                        valid = np.isfinite(values)
                        row[name] = values[valid].mean() if valid.any() else ""
                        row[name+"_valid_draws"] = int(valid.sum())
                    diag_rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    score_rows, delta_rows, fold_rows, decisions = [], [], [], []
    for k in (1, 2):
        for j, judge in enumerate(JUDGES):
            gate_open = True
            for budget in reversed(BUDGETS):
                if (1, k, budget) not in by_fold:
                    continue
                values = np.stack([by_fold[a, k, budget][:, j] for a in range(1, 11)])
                # Ten fold means determine uncertainty. Fit-draw variability is reported separately.
                means = values.mean(axis=1)
                for m, method in enumerate(METHODS):
                    for d, metric in enumerate(METRICS):
                        point, lo, hi, _ = interval(means[:, m, d])
                        score_rows.append(dict(judge=judge, ratings_per_answer=k, training_ratings=budget,
                                               method=method, metric=metric, mean=point, lower95=lo, upper95=hi,
                                               mean_within_fold_draw_sd=values[:, :, m, d].std(axis=1, ddof=1).mean()))
                    for a in range(10):
                        fold_rows.append(dict(judge=judge, ratings_per_answer=k, training_ratings=budget,
                                              assignment=a+1, method=method, **dict(zip(METRICS, means[a,m]))))
                passed = {}
                for ref in ("pearson_fit", "uniform", "train_hist", "identity"):
                    m = METHODS.index(ref)
                    for d, metric in enumerate(METRICS):
                        diffs = means[:, m, d] - means[:, 0, d]
                        point, lo, hi, pval = interval(diffs)
                        pflip = float(np.mean((SIGNS*diffs).mean(axis=1) >= point - 1e-14))
                        delta_rows.append(dict(judge=judge, ratings_per_answer=k, training_ratings=budget,
                                               comparator=ref, subtract="ce_fit", metric=metric,
                                               improvement=point, lower_one_sided_975=lo, upper_two_sided_95=hi,
                                               p_one_sided_t=pval, p_one_sided_signflip=pflip,
                                               folds_favor_ce=int(np.count_nonzero(diffs > 1e-12)),
                                               mean_within_fold_paired_draw_sd=(values[:,:,m,d]-values[:,:,0,d]).std(axis=1,ddof=1).mean()))
                        if metric == "rps":
                            passed[ref] = bool(lo > 0)
                local = passed["pearson_fit"] and passed["uniform"]
                status = "secondary" if k == 1 else ("past_stop" if not gate_open else ("pass" if local else "stop"))
                decisions.append(dict(judge=judge, ratings_per_answer=k, training_ratings=budget,
                                      local_both_pass=int(local), formal_status=status,
                                      histogram_lower_positive=int(passed["train_hist"])))
                gate_open = gate_open and local
    write_csv(out, "scores.csv", score_rows)
    write_csv(out, "contrasts.csv", delta_rows)
    write_csv(out, "families.csv", fold_rows)
    write_csv(out, "decisions.csv", decisions)
    write_csv(out, "diagnostics.csv", diag_rows)
    selections = [dict(ratings_per_answer=k, training_ratings=b, judge=j, method=m,
                       t=COEFFICIENTS[h][0], b=COEFFICIENTS[h][1], c=COEFFICIENTS[h][2],
                       selected=v) for (k,b,j,m,h),v in sorted(counts.items())]
    write_csv(out, "selections.csv", selections)
    write_csv(out, "undefined_pearson.csv", [dict(ratings_per_answer=k,training_ratings=b,judge=j,
                undefined_training_fits=v,total_fits=10*REPEATS) for (k,b,j),v in sorted(undefined.items())])
    for d in decisions:
        if d["ratings_per_answer"] == 2:
            print(d["judge"], d["training_ratings"], d["formal_status"], "hist", d["histogram_lower_positive"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "label_file", "forecast_file", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
