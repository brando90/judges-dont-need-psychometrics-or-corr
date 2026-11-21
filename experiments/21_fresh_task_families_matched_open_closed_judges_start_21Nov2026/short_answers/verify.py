"""Independent replay of the one- and two-grader procedures. Use --fixture without study labels."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from zipfile import ZipFile

import numpy as np
from scipy.stats import t


BUDGETS = (4, 8, 16, 32, 64, 128, 256)
METHODS = ("ce_fit", "pearson_fit", "uniform", "identity", "train_hist")
JUDGES = ("qwen", "claude")
METRICS = ("rps", "ce", "tvd")
DEGREES = sorted(itertools.product((.5, 1, 2), (-1, 0, 1), (-1, 0, 1)),
                 key=lambda x: (math.log2(x[0])**2+x[1]**2+x[2]**2, x))


def grid(p):
    probs = []
    for power, slope, spread in DEGREES:
        unscaled = []
        for k in range(6):
            u = (k-2.5)/2.5
            unscaled.append(((1-6e-6)*p[k]+1e-6)**power * math.exp(slope*u + spread*u*u))
        probs.append([mass/math.fsum(unscaled) for mass in unscaled])
    return np.asarray(probs)


def losses(prob, grades):
    """Compute individual threshold/observed-category losses and empirical histogram distance."""
    cdf = np.cumsum(prob, axis=-1)[..., :5]
    grade1, grade2 = grades
    rps = sum(((cdf[..., k] - (grade1 <= k))**2 + (cdf[..., k] - (grade2 <= k))**2)
              for k in range(5))/10
    ce = -(np.log(prob[..., grade1]) + np.log(prob[..., grade2]))/2
    tvd = sum(abs(prob[..., k] - ((int(grade1 == k) + int(grade2 == k))/2))
              for k in range(6))/2
    return np.stack((rps, ce, tvd), axis=-1)


def train_order(assignments, held, draw):
    rng = np.random.default_rng([20261121, held, draw])
    other = sorted(set(assignments)-{held})
    groups = [list(rng.permutation([i for i, group in enumerate(assignments) if group == a]))
              for a in other]
    order = [groups[j][round_] for round_ in range(16) for j in rng.permutation(len(other))]
    assert len(order) == len(set(order)) == 144 and all(assignments[i] != held for i in order)
    for endpoint in range(9,145,9):
        assert set(assignments[i] for i in order[endpoint-9:endpoint]) == set(other)
    return order, rng.integers(0, 2, size=144)


def first_max(scores):
    return np.argmax(scores >= np.max(scores, axis=-1, keepdims=True)-1e-12, axis=-1)


def replay(base, labels, assignments, repeats, n_graders=2):
    # [answer,judge,head]. Compute each score against the two actual grade
    # observations here, independently of any production histogram scorer.
    all_q = np.array([[grid(p) for p in both] for both in base])
    scores = np.array([losses(q, pair) for q, pair in zip(all_q, labels)])
    means = all_q @ np.arange(6)
    log_q = np.log(all_q)
    budgets = [b for b in BUDGETS if b//n_graders <= 144]
    target = np.asarray(labels).mean(axis=1)
    uniform = np.full(6, 1/6)
    family_results = {}
    order_digest = hashlib.sha256()
    for held in range(1, 11):
        test = np.flatnonzero(assignments == held)
        assert len(test) == 16
        evaluation = scores[test].mean(axis=0)
        uniform_eval = np.mean([losses(uniform, labels[i]) for i in test], axis=0)
        # Score all training histograms as a batch after selections are complete.
        for draw in range(repeats):
            order, graders = train_order(assignments, held, draw)
            order_digest.update(np.asarray(order,dtype="<i2").tobytes())
            for budget in budgets:
                train = order[:budget//n_graders]
                assert len(train)*n_graders == budget
                # Equal weighting of both observed grades. Pearson is a centered
                # scalar product between the two item-mean vectors.
                if n_graders == 2:
                    observed = labels[train].ravel()
                    ce_head = first_max(-scores[train, :, :, 1].mean(axis=0))
                    observed_means = target[train]
                else:
                    observed = labels[train, graders[:len(train)]]
                    ce_head = first_max(log_q[train, :, :, observed].mean(axis=0))
                    observed_means = observed
                m = means[train]
                mc, yc = m-m.mean(axis=0), observed_means-observed_means.mean()
                norm = np.sqrt(np.sum(mc*mc, axis=0)*np.dot(yc, yc))
                allowed = (np.ptp(m, axis=0)>1e-12) & (np.ptp(observed_means)>1e-12) & (norm>0)
                rho = np.divide(np.einsum("ijk,i->jk", mc, yc), norm,
                                out=np.full(norm.shape,-np.inf), where=allowed)
                corr_head = first_max(rho)
                hist = (np.bincount(observed,minlength=6)+.5)/(budget+3)
                cells = family_results.setdefault((held,budget), {"result":[],"hist":[]})
                cells["hist"].append(hist)
                cells["result"].append([[evaluation[j,ce_head[j]], evaluation[j,corr_head[j]],
                                         uniform_eval, evaluation[j,0]] for j in range(2)])
        for budget in budgets:
            cells=family_results[held,budget]
            histories = np.stack(cells["hist"])
            hist_eval = sum(losses(histories,labels[index]) for index in test)/len(test)
            primary = np.mean(cells["result"],axis=0)
            family_results[held,budget] = np.concatenate(
                (primary, np.broadcast_to(hist_eval.mean(axis=0),(2,1,3))), axis=1)
    return family_results, order_digest.hexdigest()


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def interval(values):
    delta = statistics.mean(values)
    se = statistics.stdev(values)*math.sqrt(1/10+1/9)
    return delta, delta - t.isf(.025,9)*se, delta + t.isf(.025,9)*se


def verify_outputs(result, out, n_graders=2):
    def primary(name):
        return [r for r in read_csv(out / name) if int(r["ratings_per_answer"])==n_graders]
    scores = primary("scores.csv")
    families = primary("families.csv")
    contrasts = primary("contrasts.csv")
    decisions = primary("decisions.csv")
    budgets = [b for b in BUDGETS if b//n_graders <= 144]
    n_budget = len(budgets)
    assert [len(rows) for rows in (scores,families,contrasts,decisions)] == [
        n*n_budget for n in (30,100,24,2)]

    def check(observed, expected):
        assert math.isclose(float(observed), expected, abs_tol=6e-12, rel_tol=6e-12), (observed,expected)

    def coordinates(r):
        return JUDGES.index(r["judge"]), int(r["training_ratings"])
    for r in families:
        j,n = coordinates(r)
        ref = result[int(r["assignment"]),n][j,METHODS.index(r["method"])]
        for k, name in enumerate(METRICS):
            check(r[name],ref[k])
    for r in scores:
        j,n=coordinates(r)
        idx=(j,METHODS.index(r["method"]),METRICS.index(r["metric"]))
        vals=[result[a,n][idx] for a in range(1,11)]
        for key, expected in zip(("mean","lower95","upper95"),interval(vals)):
            check(r[key],expected)
    bounds = {}
    signs = np.asarray(list(itertools.product((1,-1), repeat=10)))
    for r in contrasts:
        j,n=coordinates(r)
        m=METHODS.index(r["comparator"])
        k=METRICS.index(r["metric"])
        vals=[result[a,n][j,m,k] - result[a,n][j,0,k] for a in range(1,11)]
        triplet=interval(vals)
        for key, expected in zip(("improvement","lower_one_sided_975","upper_two_sided_95"),triplet):
            check(r[key],expected)
        sd = statistics.stdev(vals)*math.sqrt(1/10+1/9)
        check(r["p_one_sided_t"], t.sf(triplet[0]/sd,9) if sd else float(triplet[0]<=0))
        sign_p = np.count_nonzero((signs @ vals)/10 >= triplet[0]-1e-14)/len(signs)
        check(r["p_one_sided_signflip"],sign_p)
        assert int(r["folds_favor_ce"])==sum(value>1e-12 for value in vals)
        if k==0:
            bounds[j,n,m]=triplet
    for j,judge in enumerate(JUDGES):
        open_gate = True
        for n in reversed(budgets):
            local = all(bounds[j,n,m][1] > 0 for m in (1,2))
            status = ("pass" if local else "stop") if open_gate else "past_stop"
            if n_graders == 1:
                status = "secondary"
            r, = [row for row in decisions if coordinates(row)==(j,n)]
            assert status==r["formal_status"] and int(local)==int(r["local_both_pass"])
            assert int(bounds[j,n,4][1]>0)==int(r["histogram_lower_positive"])
            print(judge,n,status,"Pearson−CE",tuple(map(float,bounds[j,n,1])),
                  "uniform−CE",tuple(map(float,bounds[j,n,2])))
            open_gate &= local
    print(f"Verified {len(families)} assignment/method rows, {len(scores)} absolute score/interval rows, "
          f"{len(contrasts)} paired differences with bounds, t/sign-flip p-values and fold wins, "
          f"and {len(decisions)} decision rows.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture",action="store_true")
    parser.add_argument("--source", type=Path, help="Optional original UNT v2.0 zip for grade provenance.")
    for name in ("manifest", "labels", "forecasts", "results"):
        parser.add_argument("--"+name,type=Path)
    args=parser.parse_args()
    if args.fixture:
        rng=np.random.default_rng(17)
        base=rng.dirichlet(np.ones(6),size=(160,2))
        labels=rng.integers(0,6,size=(160,2))
        assignments=np.repeat(np.arange(1,11),16)
        results,_=replay(base,labels,assignments,2)
        assert len(results)==70 and all(np.isfinite(v).all() for v in results.values())
        # Perfect one-hot forecasts have zero RPS/TVD and zero CE on identical ratings.
        np.testing.assert_allclose(losses(np.array([0,0,0,1.,0,0]), [3,3]),0)
        print("Synthetic fixture passed all 70 fold/budget cells.")
        return
    if not all((args.manifest,args.labels,args.forecasts,args.results)):
        parser.error("Supply all four study paths or --fixture.")
    items=[json.loads(line) for line in args.manifest.read_bytes().splitlines()]
    by_id={r["id"]:r["ratings"] for r in map(json.loads,args.labels.read_bytes().splitlines())}
    if args.source:
        assert hashlib.sha256(args.source.read_bytes()).hexdigest() == (
            "5abee7ee804a2c5ae802b226230fb14c5bdb894f0f33bab6e20b4e7f1ed89247")
        with ZipFile(args.source) as raw:
            for item_id, observed in by_id.items():
                question, position = item_id.split("/")
                original = [float(raw.read(f"data/scores/{question}/{grader}")
                                  .splitlines()[int(position)]) for grader in ("other", "me")]
                assert original == observed
        print("Both grades for each of the 160 answers match the original release archive.")
    rows=read_csv(args.forecasts)
    forecasts={(r["id"],r["judge"]):[float(r["p"+str(k)]) for k in range(6)] for r in rows}
    assert len(items)==len(by_id)==160 and len(rows)==len(forecasts)==320
    base=np.array([[forecasts[i["id"],j] for j in JUDGES] for i in items])
    labels=np.asarray([by_id[i["id"]] for i in items])
    assert labels.shape==(160,2) and np.isin(labels,range(6)).all()
    assignments=np.array([i["assignment"] for i in items])
    result, digest=replay(base,labels,assignments,200)
    # Matched independently against the production scheduler using the blind manifest.
    assert digest == "5b3255826731f09dc722bb769532fd0c88e568e57c4dbe97683b0ea7c6c1b647"
    print("Verified 2,000 held-assignment/draw training orders; SHA256 (little-endian int16):", digest)
    verify_outputs(result,args.results)
    secondary, secondary_digest = replay(base,labels,assignments,200,n_graders=1)
    assert secondary_digest==digest
    verify_outputs(secondary,args.results,n_graders=1)


if __name__ == "__main__":
    main()
