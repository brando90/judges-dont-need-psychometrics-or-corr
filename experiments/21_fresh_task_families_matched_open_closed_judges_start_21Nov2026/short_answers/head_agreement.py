"""Post hoc replay of joint calibration-head choices at the two primary budget endpoints."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from analyze import JUDGES, REPEATS, schedule, select
from metrics import COEFFICIENTS, GRADE, TOL, calibrations, rating_histograms, score
from prepare import SOURCE_SHA

ROOT = Path(__file__).resolve().parent
BUDGETS = (4, 256)


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def load(source):
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
    manifest = [json.loads(line) for line in (ROOT / 'design/manifest.jsonl').read_bytes().splitlines()]
    ratings = []
    with ZipFile(source) as archive:
        for item in manifest:
            question, position = item['id'].split('/')
            ratings.append([float(archive.read(f'data/scores/{question}/{grader}').splitlines()[int(position)])
                            for grader in ('other', 'me')])
    rows = read_csv(ROOT / 'forecasts.csv')
    forecasts = {(r['id'], r['judge']): [float(r[f'p{k}']) for k in GRADE] for r in rows}
    assert len(manifest) == len(ratings) == 160 and len(rows) == len(forecasts) == 320
    forecasts = np.array([[forecasts[item['id'], judge] for judge in JUDGES] for item in manifest])
    return np.array([item['assignment'] for item in manifest]), np.array(ratings), calibrations(forecasts)


def replay(assignments, ratings, candidates):
    labels = rating_histograms(ratings)
    losses = score(candidates, labels[:, None, None])[:, :, :, 0]
    counters = defaultdict(Counter)
    differences = defaultdict(list)
    for held in np.unique(assignments):
        held_loss = losses[assignments == held].mean(axis=0)
        for draw in range(REPEATS):
            order, _ = schedule(assignments, held, draw)
            for budget in BUDGETS:
                train = order[:budget // 2]
                targets = labels[train] @ GRADE
                ce, pearson, undefined = select(candidates[train], labels[train], targets)
                if budget == 4:
                    assert len(train) == 2 and len(set(assignments[train])) == 2
                    # For two points, every defined Pearson value equals the sign of this product.
                    candidate_means = candidates[train] @ GRADE
                    predicted_difference = candidate_means[1] - candidate_means[0]
                    valid = (np.abs(predicted_difference) > TOL) & (np.ptp(targets) > TOL)
                    signs = np.where(valid, np.sign(predicted_difference * (targets[1] - targets[0])), -np.inf)
                    best = (signs == signs.max(axis=1, keepdims=True)) & valid
                    np.testing.assert_array_equal(pearson, best.argmax(axis=1))
                for j, judge in enumerate(JUDGES):
                    count = counters[judge, budget]
                    count['ce_fit', ce[j]] += 1
                    count['pearson_fit', pearson[j]] += 1
                    count['undefined'] += int(undefined[j])
                    count['equal_target'] += int(np.ptp(targets) <= TOL)
                    if budget == 4:
                        count['defined_multiple_maximizers'] += int(best[j].sum() > 1)
                    same = ce[j] == pearson[j]
                    count['same'] += int(same)
                    count['both_identity'] += int(same and ce[j] == 0)
                    count['same_dominant'] += int(same and COEFFICIENTS[ce[j]] == (0.5, 1, 1))
                    gap = float(held_loss[j, pearson[j]] - held_loss[j, ce[j]])
                    assert not same or gap == 0
                    differences[judge, budget, 'all'].append(gap)
                    if not same:
                        differences[judge, budget, 'different'].append(gap)
    return counters, differences


def published_cells(filename):
    return [r for r in read_csv(ROOT / 'results' / filename)
            if int(r['ratings_per_answer']) == 2 and int(r['training_ratings']) in BUDGETS]


def format_head(index):
    return '(' + ','.join(f'{x:g}' for x in COEFFICIENTS[index]) + ')'


def summarize(source):
    assignments, ratings, candidates = load(source)
    counts, gaps = replay(assignments, ratings, candidates)
    for row in published_cells('selections.csv'):
        head = COEFFICIENTS.index((float(row['t']), int(row['b']), int(row['c'])))
        cell = counts[row['judge'], int(row['training_ratings'])]
        assert cell[row['method'], head] == int(row['selected'])
    for row in published_cells('undefined_pearson.csv'):
        assert counts[row['judge'], int(row['training_ratings'])]['undefined'] == int(row['undefined_training_fits'])
    contrasts = {(r['judge'], int(r['training_ratings'])): float(r['improvement'])
                 for r in published_cells('contrasts.csv')
                 if (r['comparator'], r['subtract'], r['metric']) == ('pearson_fit', 'ce_fit', 'rps')}
    diagnostics = published_cells('diagnostics.csv')
    records = []
    for judge in JUDGES:
        for budget in BUDGETS:
            count = counts[judge, budget]
            all_gaps = gaps[judge, budget, 'all']
            differing_gaps = gaps[judge, budget, 'different']
            total = len(all_gaps)
            assert total == 10 * REPEATS and total - count['same'] == len(differing_gaps)
            np.testing.assert_allclose(np.mean(all_gaps), contrasts[judge, budget], rtol=0, atol=1e-12)
            record = dict(judge=judge, training_ratings=budget, training_answers=budget // 2, draws=total,
                          grade_count=ratings.size, count_grade_5=int((ratings == 5).sum()),
                          mean_human_grade=ratings.mean(), equal_training_target_means=count['equal_target'],
                          all_training_pearsons_undefined=count['undefined'],
                          defined_multiple_best_heads=count['defined_multiple_maximizers'] if budget == 4 else '',
                          both_same_head=count['same'], both_identity=count['both_identity'],
                          both_t05_b1_c1=count['same_dominant'], different_heads=len(differing_gaps),
                          test_rps_gap_all_draws=np.mean(all_gaps),
                          test_rps_gap_given_different_heads=np.mean(differing_gaps))
            for method in ('ce_fit', 'pearson_fit'):
                freq = [count[method, index] for index in range(len(COEFFICIENTS))]
                assert sum(freq) == total
                mode = int(np.argmax(freq))
                record.update({f'{method}_identity': freq[0], f'{method}_mode': format_head(mode),
                               f'{method}_mode_count': freq[mode],
                               f'{method}_t05_b1_c1': freq[COEFFICIENTS.index((0.5, 1, 1))]})
                folds = [row for row in diagnostics if (row['judge'], int(row['training_ratings']), row['method'])
                         == (judge, budget, method)]
                assert len(folds) == 10
                for measure in ('forecast_mean', 'bias', 'pearson'):
                    record[f'{method}_test_{measure}'] = np.mean([float(row[measure]) for row in folds])
            records.append(record)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Original ShortAnswerGrading_v2.0.zip')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/head_agreement.csv')
    args = parser.parse_args()
    rows = summarize(args.source)
    with args.out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row['judge'], row['training_ratings'], 'same', row['both_same_head'], 'of', row['draws'],
              'test RPS gap across all draws', row['test_rps_gap_all_draws'])
