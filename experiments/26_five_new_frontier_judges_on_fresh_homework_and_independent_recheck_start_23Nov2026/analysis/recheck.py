"""Second implementation of the experiment 21 / 26 homework CE-vs-Pearson analysis (11-23-2026).

Written from the protocol text. The author read experiment 21's schedule and select functions first, so
this is a rewrite with different structure (per-fold arrays, explicit budget loops), not a blind one; its value is
catching transcription errors and running the same procedure on new judges. Usage:
  python3 recheck.py FORECASTS.csv LABELS.jsonl MANIFEST.jsonl OUT_DIR --judges qwen claude [--alpha 0.025]
FORECASTS.csv columns: id, judge, p0..p5 (long format; failures already replaced by uniform).
"""
import argparse, csv, itertools, json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

G = np.arange(6)
BUDGETS = (4, 8, 16, 32, 64, 128, 256)
DRAWS = 200
TOL = 1e-12
GRID = sorted(itertools.product((0.5, 1.0, 2.0), (-1, 0, 1), (-1, 0, 1)),
              key=lambda v: (np.log2(v[0]) ** 2 + v[1] ** 2 + v[2] ** 2, v))
T_CRIT = student_t.ppf(0.975, 9)


def transforms(p):
    """p: (n,6) judge probabilities -> (n,27,6) calibrated distributions."""
    p = (1 - 6e-6) * p + 1e-6
    u = (G - 2.5) / 2.5
    out = np.empty((len(p), len(GRID), 6))
    for k, (t, b, c) in enumerate(GRID):
        z = t * np.log(p) + b * u + c * u * u
        z = np.exp(z - z.max(axis=1, keepdims=True))
        out[:, k] = z / z.sum(axis=1, keepdims=True)
    return out


def metrics(q, hist):
    """q, hist broadcast to (...,6); returns rps, ce, tvd averaged over the rating distribution hist."""
    F, H = np.cumsum(q, -1)[..., :5], np.cumsum(hist, -1)[..., :5]
    # expected squared cumulative error when the observed rating is drawn from hist
    rps = (F ** 2 - 2 * F * H + H).mean(-1)
    ce = -(hist * np.log(q)).sum(-1)
    tvd = 0.5 * np.abs(q - hist).sum(-1)
    return rps, ce, tvd


def schedule(fam, held, rep):
    rng = np.random.default_rng(np.random.SeedSequence([20261121, held, rep]))
    groups = [rng.permutation(np.flatnonzero(fam == a)) for a in sorted(set(fam)) if a != held]
    order = []
    for pos in range(16):
        for g in rng.permutation(len(groups)):
            order.append(groups[g][pos])
    return np.array(order), rng.integers(0, 2, size=len(order))


def pick(loss_or_corr, valid, maximise):
    """Earliest grid entry within TOL of the best value; all-invalid -> identity (0)."""
    x = loss_or_corr.copy()
    if maximise:
        x[~valid] = -np.inf
        best = x.max(1, keepdims=True)
        sel = (x >= best - TOL).argmax(1)
    else:
        best = x.min(1, keepdims=True)
        sel = (x <= best + TOL).argmax(1)
    if maximise:
        sel[~valid.any(1)] = 0
    return sel


def analyse(P, Y, fam, judge_label):
    """P (160,6) forecasts; Y (160,2) ratings; fam (160,) assignment. Returns dict[(rpa,B,method)] -> (10,3) fold means."""
    Q = transforms(P)                       # (160,27,6)
    logQ = np.log(Q)
    M = Q @ G                               # predicted means (160,27)
    hist_all = (Y[..., None] == G).mean(1)  # (160,6) two-grader histogram
    folds = sorted(set(fam))
    res = {}
    arange27 = np.arange(len(GRID))
    for fi, held in enumerate(folds):
        test = np.flatnonzero(fam == held)
        sched = [schedule(fam, held, r) for r in range(DRAWS)]
        orders = np.stack([s[0] for s in sched])    # (200,144)
        coin = np.stack([s[1] for s in sched])
        for rpa in (2, 1):
            for B in BUDGETS:
                n = B // 2 if rpa == 2 else B
                if n > 144:
                    continue
                idx = orders[:, :n]                                       # (200,n)
                if rpa == 2:
                    ratings = Y[idx]                                       # (200,n,2)
                else:
                    ratings = Y[idx, coin[:, :n]][..., None]               # (200,n,1)
                # training CE loss per draw and transform
                ce_loss = np.zeros((DRAWS, 27))
                for j in range(ratings.shape[2]):
                    ce_loss += -logQ[idx[:, :, None], arange27[None, None, :], ratings[:, :, j, None]].sum(1)
                ce_loss /= n * ratings.shape[2]            # mean individual-rating CE, the scale the 1e-12 tie rule refers to
                ce_sel = pick(ce_loss, np.ones_like(ce_loss, bool), False)
                # Pearson with mean grade of the answer's training ratings
                ybar = ratings.mean(2)                                      # (200,n)
                m = M[idx]                                                  # (200,n,27)
                mc = m - m.mean(1, keepdims=True)
                yc = ybar - ybar.mean(1, keepdims=True)
                num = (mc * yc[:, :, None]).sum(1)
                den = np.sqrt((mc ** 2).sum(1)) * np.sqrt((yc ** 2).sum(1, keepdims=True))
                valid = (np.ptp(m, 1) > TOL) & (np.ptp(ybar, 1, keepdims=True) > TOL) & (den > 0)
                rho = np.where(valid, num / np.where(den > 0, den, 1), -np.inf)
                pe_sel = pick(rho, valid, True)
                # training histogram
                counts = np.zeros((DRAWS, 6))
                for j in range(ratings.shape[2]):
                    for k in range(6):
                        counts[:, k] += (ratings[:, :, j] == k).sum(1)
                th = (counts + 0.5) / (counts + 0.5).sum(1, keepdims=True)
                h = hist_all[test]                                          # (16,6)
                out = {}
                out["ce_fit"] = np.stack(metrics(Q[test][:, ce_sel, :].transpose(1, 0, 2), h[None]), -1).mean((0, 1))
                out["pearson_fit"] = np.stack(metrics(Q[test][:, pe_sel, :].transpose(1, 0, 2), h[None]), -1).mean((0, 1))
                out["identity"] = np.stack(metrics(Q[test][:, 0, :], h), -1).mean(0)
                out["uniform"] = np.stack(metrics(np.full((16, 6), 1 / 6), h), -1).mean(0)
                out["train_hist"] = np.stack(metrics(th[:, None, :], h[None]), -1).mean((0, 1))
                for name, v in out.items():
                    res.setdefault((rpa, B, name), np.zeros((len(folds), 3)))[fi] = v
    return res


def interval(x):
    """x: ten fold values -> mean, lower, upper (95% two-sided, overlap-adjusted t), one-sided 97.5% lower."""
    mu, sd = x.mean(), x.std(ddof=1)
    se = sd * np.sqrt(1 / 10 + 1 / 9)
    return mu, mu - T_CRIT * se, mu + T_CRIT * se, se


def write(out, judge, res):
    scores, contrasts, decisions = [], [], []
    for (rpa, B, name), v in sorted(res.items()):
        for mi, metric in enumerate(("rps", "ce", "tvd")):
            mu, lo, hi, _ = interval(v[:, mi])
            scores.append(dict(judge=judge, ratings_per_answer=rpa, training_ratings=B, method=name, metric=metric,
                               mean=mu, lower95=lo, upper95=hi))
    for rpa in (2, 1):
        for B in BUDGETS:
            if (rpa, B, "ce_fit") not in res:
                continue
            for comp in ("pearson_fit", "uniform", "identity", "train_hist"):
                d = res[(rpa, B, comp)] - res[(rpa, B, "ce_fit")]      # comparator minus CE fit: positive = CE better
                for mi, metric in enumerate(("rps", "ce", "tvd")):
                    mu, lo, hi, se = interval(d[:, mi])
                    contrasts.append(dict(judge=judge, ratings_per_answer=rpa, training_ratings=B, comparator=comp,
                                          metric=metric, improvement=mu, lower_975=mu - T_CRIT * se, upper_95=hi,
                                          folds_favor_ce=int((d[:, mi] > 0).sum())))
    return scores, contrasts


def decisions(contrasts, judge, alpha):
    """Descending sequential rule on primary two-rating RPS versus Pearson-fit and uniform."""
    crit = student_t.ppf(1 - alpha, 9)
    rows, stopped = [], False
    lookup = {(c["training_ratings"], c["comparator"]): c for c in contrasts
              if c["judge"] == judge and c["ratings_per_answer"] == 2 and c["metric"] == "rps"}
    for B in sorted(BUDGETS, reverse=True):
        lb = {}
        for comp in ("pearson_fit", "uniform", "train_hist"):
            c = lookup[(B, comp)]
            se = (c["improvement"] - c["lower_975"]) / T_CRIT
            lb[comp] = c["improvement"] - crit * se
        local = lb["pearson_fit"] > 0 and lb["uniform"] > 0
        status = "not_tested_after_failure" if stopped else ("pass" if local else "fail")
        if not stopped and not local:
            stopped = True
        rows.append(dict(judge=judge, training_ratings=B, alpha=alpha, lb_vs_pearson=lb["pearson_fit"],
                         lb_vs_uniform=lb["uniform"], lb_vs_histogram=lb["train_hist"],
                         local_both_pass=int(local), formal_status=status))
    return rows


def dump(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("forecasts", type=Path); ap.add_argument("labels", type=Path)
    ap.add_argument("manifest", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--judges", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.025)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ids = [json.loads(l)["id"] for l in a.manifest.read_text().splitlines()]
    fam = np.array([json.loads(l)["assignment"] for l in a.manifest.read_text().splitlines()])
    lab = {r["id"]: r["ratings"] for r in map(json.loads, a.labels.read_text().splitlines())}
    Y = np.array([lab[i] for i in ids])
    rows = list(csv.DictReader(a.forecasts.open()))
    S, C, D = [], [], []
    for j in a.judges:
        by = {r["id"]: [float(r[f"p{k}"]) for k in range(6)] for r in rows if r["judge"] == j}
        P = np.array([by[i] for i in ids])
        s, c = write(a.out, j, analyse(P, Y, fam, j))
        S += s; C += c
        D += decisions(C, j, a.alpha)
        print(j, "done", flush=True)
    dump(a.out / "scores.csv", S); dump(a.out / "contrasts.csv", C); dump(a.out / "decisions.csv", D)


if __name__ == "__main__":
    main()
