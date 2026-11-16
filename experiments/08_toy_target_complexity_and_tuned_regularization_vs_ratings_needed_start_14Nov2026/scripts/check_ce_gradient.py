"""Post-hoc check (added 11-16-2026, Mega QA finding QA1-F1): does the inconsistent cross-entropy (CE) gradient in
experiment 13's frozen objective code change any saved result of experiments 07, 08 or 13?

The defect: experiment 13's `toy.ce_loss_grad` clips all six softmax probabilities at 1e-6 and renormalizes them before
taking the loss value, but returns (renormalized clipped probabilities - target histogram) as the gradient. Whenever any
category probability (observed or unobserved) falls below 1e-6, renormalization shifts the loss value on observed bins
and the returned gradient (~+1e-6 on a clipped unobserved bin, or Q-P on an observed bin) disagrees with the value's derivative.

The check: swap in an exact CE in memory -- value -sum_r p_r log softmax(z)_r computed with log-sum-exp, no floor,
and its exact gradient softmax(z) - p (the CE objective is then smooth and convex in the weights) -- and refit every
fit that uses CE, with the same data, starts, ridge, optimizer and budgets as the saved run. Held-out test metrics
are still computed with experiment 13's `toy.metrics` (its 1e-6 evaluation floor is part of the metric definition and
is unchanged). Each refit is compared with the saved row. No frozen file is modified and no saved output is
rewritten; the comparison is written to results/ce_gradient_check.json.

Parts (all rows of each saved grid):
  exp08  CE-trained judges with ridge c = 1 (formulation 4) and with c chosen by 4-fold validation RPS
         (formulation 3), 4 scenarios x 11 sizes x 100 seeds. Scenarios A and B also cover experiment 13's CE fits:
         they regenerate the same data, and the saved values agree to 4e-14.
  exp13  experiment 13's full run_one (CE, RPS, Pearson and the calibrated CE head on the Pearson score) and its
         dense-grid references for CE and the head, each run with the exact and with the frozen CE on this machine.
         The saved experiment 13 rows came from another machine, where the non-convex Pearson fits (and the head on
         their score) differ at about the 1e-3 level, so the CE effect is measured as exact versus frozen here; the
         exact CE fits are also compared with the saved rows.
  exp07  experiment 07's CE fits (400/600 and 2,000/4,000 budgets) and the calibrated head under its three conditions.
Also counted along the exact-CE trajectories: objective evaluations in which any category probability fell below 1e-6
(where the frozen code clips, renormalizes and produces a value/gradient mismatch), and the subset where that category
was observed in training.

Run from the repository root (about 15 minutes with 8 processes on a 10-core laptop):
  python3 experiments/08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/scripts/check_ce_gradient.py [processes]
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import softmax  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXPS = os.path.abspath(os.path.join(HERE, "..", ".."))
E08 = os.path.dirname(HERE)
E07 = os.path.join(EXPS, "07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives_start_13Nov2026")
E13 = os.path.join(EXPS, "archive", "13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(E07, "scripts"))
import complexity as cx  # noqa: E402  (imports experiment 13's frozen toy module)
import restarts  # noqa: E402  (experiment 07; uses the same toy module object)

toy = cx.toy
assert restarts.toy is toy
FROZEN_CE = toy.ce_loss_grad
FLOOR = toy.FLOOR
STATS = {"evals": 0, "any_below_floor": 0, "observed_below_floor": 0, "fits": 0, "converged": 0}


def exact_ce_loss_grad(Z, P):
    """Exact CE per row and its gradient with respect to the logits Z (rows of P sum to 1)."""
    Zs = Z - Z.max(1, keepdims=True)
    logQ = Zs - np.log(np.exp(Zs).sum(1, keepdims=True))
    Q = np.exp(logQ)
    STATS["evals"] += 1
    STATS["any_below_floor"] += int(np.any(Q < FLOOR))
    STATS["observed_below_floor"] += int(np.any((P > 0) & (Q < FLOOR)))
    return -(P * logQ).sum(1), Q - P


toy.ce_loss_grad = exact_ce_loss_grad  # in memory only; children re-import this module and patch again
_minimize = toy.minimize


def logged_minimize(*a, **kw):
    """scipy's minimize, counting fits and scipy's convergence flag (used only where every fit is a CE fit)."""
    res = _minimize(*a, **kw)
    STATS["fits"] += 1
    STATS["converged"] += bool(res.success)
    return res


K, NTEST = 3, 500
METRICS = ("rps_excess", "ce_excess", "pearson", "spearman", "tvd_from_constant")


def reset():
    for k in STATS:
        STATS[k] = 0


def exp08_row(args):
    """Experiment 08's CE fits for one (scenario, n, seed), in the same order and with the same data as cx.run_one."""
    sc, n, seed = args
    reset()
    rng = np.random.default_rng(1000003 * seed + 7919 * n + cx.CODE[sc])
    x = rng.uniform(-1, 1, n)
    P = np.array([rng.multinomial(K, p) for p in cx.truth(x, sc)]) / K
    Phi = toy.phi(x)
    W0 = toy.marginal_W0(Phi, P, seed)
    xt = rng.uniform(-1, 1, NTEST)
    Pt = cx.truth(xt, sc)
    Phit = toy.phi(xt)
    fold = np.empty(n, int)
    fold[np.random.default_rng([seed, n, cx.CODE[sc], 99]).permutation(n)] = np.arange(n) % cx.FOLDS
    Qconst = np.tile(P.mean(0), (NTEST, 1))
    toy.minimize = logged_minimize  # every fit below is a CE fit
    try:
        W = toy.fit(Phi, P, "ce", n, W0, c=1.0)
        out = {"ce_fixed": cx.test_metrics(softmax(Phit @ W, 1), Pt, xt, Qconst)}
        vs = [cx.val_scores(Phi, P, "ce", c, fold, seed) for c in cx.C_GRID]
        c = cx.pick(cx.C_GRID, [v[0] for v in vs], True)
        fallback = c is None
        c = 1.0 if fallback else c
        Wc = W if c == 1.0 else toy.fit(Phi, P, "ce", n, W0, c=c)
        out["ce_valrps"] = dict(cx.test_metrics(softmax(Phit @ Wc, 1), Pt, xt, Qconst), c=c, fallback=fallback)
    finally:
        toy.minimize = _minimize
    return sc, n, seed, out, dict(STATS)


def with_frozen_ce(fn, *a):
    """Run fn with experiment 13's original CE, then restore the exact CE."""
    toy.ce_loss_grad = FROZEN_CE
    try:
        return fn(*a)
    finally:
        toy.ce_loss_grad = exact_ce_loss_grad


def exp13_row(args):
    """Experiment 13's run_one with the exact CE and, on this machine, with the frozen CE. Its saved rows came from
    another machine, where the non-convex Pearson fits (and so the head on their score) differ at the 1e-3 level;
    the frozen-versus-exact comparison on one machine isolates the effect of the CE gradient."""
    sc, n, seed = args
    reset()
    _, _, _, exact = toy.run_one((sc, n, seed, K, NTEST))
    stats = dict(STATS)
    _, _, _, frozen = with_frozen_ce(toy.run_one, (sc, n, seed, K, NTEST))
    return sc, n, seed, {"exact": exact, "frozen": frozen}, stats


def exp07_row(args):
    """Experiment 07's CE fits and calibrated heads (same data, starts and budgets as restarts.run_one)."""
    sc, n, seed = args
    reset()
    rng = np.random.default_rng(1000003 * seed + 7919 * n + (sc == "B"))
    x = rng.uniform(-1, 1, n)
    P = np.array([rng.multinomial(K, p) for p in toy.truth(x, sc)]) / K
    Phi = toy.phi(x)
    W0 = toy.marginal_W0(Phi, P, seed)
    xt = rng.uniform(-1, 1, NTEST)
    Pt = toy.truth(xt, sc)
    Phit = toy.phi(xt)
    starts = [W0] + [W0 + restarts.START_SD * np.random.default_rng([seed, n, r, sc == "B"]).normal(size=W0.shape)
                     for r in range(1, restarts.N_STARTS)]
    out = {}
    W, log = restarts.fit_logged(Phi, P, "ce", n, W0, restarts.SHORT)
    out["ce"] = {"orig": dict(log, **restarts.test_metrics(softmax(Phit @ W, 1), Pt, xt))}
    W, log = restarts.fit_logged(Phi, P, "ce", n, W0, restarts.LONG)
    out["ce"]["long"] = dict(log, **restarts.test_metrics(softmax(Phit @ W, 1), Pt, xt))
    Wp, _ = restarts.fit_logged(Phi, P, "pearson", n, W0, restarts.SHORT)
    fits = [restarts.fit_logged(Phi, P, "pearson", n, S, restarts.LONG) for S in starts]
    best = min(range(len(fits)), key=lambda r: fits[r][1]["train_obj"])
    out["pearson_head"] = {
        "orig": restarts.head_metrics(Wp, Phi, P, Phit, Pt, xt, n, seed, restarts.SHORT),
        "long": restarts.head_metrics(fits[0][0], Phi, P, Phit, Pt, xt, n, seed, restarts.LONG),
        "best": restarts.head_metrics(fits[best][0], Phi, P, Phit, Pt, xt, n, seed, restarts.LONG)}
    return sc, n, seed, out, dict(STATS)


def load(path):
    return [json.loads(line) for line in open(path)]


def compare(saved, new, keys):
    """Largest absolute difference per metric and the keys whose values differ by more than 1e-9."""
    diff = {}
    for k in keys:
        a, b = saved.get(k), new.get(k)
        if a is None and b is None:
            continue
        if a is None or b is None or (isinstance(a, float) and np.isnan(a)) != (isinstance(b, float) and np.isnan(b)):
            diff[k] = float("inf")
        elif not (isinstance(a, float) and np.isnan(a)):
            diff[k] = abs(float(a) - float(b))
    return diff


def summarize(pairs, fits, keys):
    s = {}
    for fit in fits:
        maxd = {k: 0.0 for k in keys}
        changed_rows, choice_changes = 0, 0
        for saved, new in pairs:
            sv, nv = saved[fit], new[fit]
            d = compare(sv, nv, keys)
            for k, v in d.items():
                maxd[k] = max(maxd[k], v)
            changed_rows += any(v > 1e-9 for v in d.values())
            choice_changes += any(sv.get(k) != nv.get(k) for k in ("c", "fallback", "start") if k in sv)
        s[fit] = {"rows": len(pairs), "rows_with_a_metric_difference_above_1e-9": changed_rows,
                  "selection_changes": choice_changes, "max_abs_difference": maxd}
    return s


def boot_ci(v, rng, draws=10000):
    b = v[rng.integers(0, len(v), (draws, len(v)))].mean(1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


PAIRS = [("ce_fixed", "pearson_fixed"), ("ce_valrps", "pearson_valcorr"), ("ce_valrps", "pearson_fixed"),
         ("ce_valrps", "pearson_valrps"), ("ce_valrps", "constant"), ("pearson_fixed", "constant")]
JUDGES = ["constant", "ce_fixed", "ce_valrps", "rps_valrps", "pearson_fixed", "pearson_valcorr", "pearson_valrps"]


def reported(rows, ns):
    """Every table and figure-annotation quantity in make_report.py that involves a CE-trained judge, computed with
    make_report.py's exact sequential np.random.default_rng(0) bootstrap stream so saved and exact rows are compared
    under the exact report RNG trajectory."""
    cells = {}
    for r in rows:
        cells.setdefault((r["scenario"], r["n"]), []).append(r)
    for v in cells.values():
        v.sort(key=lambda r: r["seed"])
    rng = np.random.default_rng(0)

    def vals(sc, n, j, key):
        return np.array([r["fits"][j][key] for r in cells[(sc, n)]], dtype=float)

    def diff_ci(sc, n, a, b, key):
        x, y = vals(sc, n, a, key), vals(sc, n, b, key)
        if key in ("pearson", "spearman"):
            x, y = np.nan_to_num(x, nan=0.0), np.nan_to_num(y, nan=0.0)
        return boot_ci(x - y, rng)

    out = {
        "cell_means": {}, "ce_excess_cell_means": {}, "pearson_cell_means": {}, "tvd_from_constant_cell_means": {},
        "crossovers": {}, "tuned_ce_significantly_worse_at": {}, "h1_pearson_better_at": {},
        "h2_ce_lower_test_pearson_at": {}, "tuned_ce_ridge_mode": {},
        "figure_ce_excess_crossovers": {}, "figure_test_pearson_crossovers": {},
    }
    for sc in "ABCD":
        for n in ns:
            for fit in ("ce_fixed", "ce_valrps"):
                out["cell_means"][f"{sc}/{n}/{fit}"] = f"{vals(sc, n, fit, 'rps_excess').mean():.4f}"
                out["ce_excess_cell_means"][f"{sc}/{n}/{fit}"] = f"{vals(sc, n, fit, 'ce_excess').mean():.4f}"
                out["pearson_cell_means"][f"{sc}/{n}/{fit}"] = f"{np.nan_to_num(vals(sc, n, fit, 'pearson'), nan=0.0).mean():.3f}"
            cs, counts = np.unique([r["fits"]["ce_valrps"]["c"] for r in cells[(sc, n)]], return_counts=True)
            out["tuned_ce_ridge_mode"][f"{sc}/{n}"] = f"{cs[counts.argmax()]:g} ({counts.max()}%)"
        for n in ns[:6]:
            for fit in ("ce_fixed", "ce_valrps"):
                out["tvd_from_constant_cell_means"][f"{sc}/{n}/{fit}"] = f"{vals(sc, n, fit, 'tvd_from_constant').mean():.3f}"
        for a, b in PAIRS:
            better = [diff_ci(sc, n, b, a, "rps_excess")[1] > 0 for n in ns]
            if a.startswith("ce_"):
                out["crossovers"][f"{sc}/{a}_vs_{b}"] = next((ns[k] for k in range(len(ns)) if all(better[k:])), None)
        for b in ("pearson_fixed", "pearson_valcorr", "pearson_valrps"):
            out["tuned_ce_significantly_worse_at"][f"{sc}/vs_{b}"] = [n for n in ns if diff_ci(sc, n, "ce_valrps", b, "rps_excess")[1] > 0]
        out["h1_pearson_better_at"][sc] = [n for n in ns if diff_ci(sc, n, "ce_fixed", "pearson_fixed", "rps_excess")[1] > 0]
        for b in ("pearson_fixed", "pearson_valcorr"):
            out["h2_ce_lower_test_pearson_at"][f"{sc}/vs_{b}"] = [n for n in ns if diff_ci(sc, n, "ce_fixed", b, "pearson")[2] < 0]
    for keys in (["rps_excess", "ce_excess"], ["pearson"]):
        for sc in "ABCD":
            for key in keys:
                for jd in JUDGES:
                    if key == "pearson" and jd == "constant":
                        continue
                    _ = [boot_ci(np.nan_to_num(vals(sc, n, jd, key), nan=0.0), rng) for n in ns]
                if key != "pearson":
                    for a, b in (("ce_valrps", "pearson_valcorr"), ("ce_fixed", "pearson_fixed")):
                        better = [diff_ci(sc, n, b, a, key)[1] > 0 for n in ns]
                        if key == "ce_excess":
                            out["figure_ce_excess_crossovers"][f"{sc}/{a}_vs_{b}"] = next((ns[k] for k in range(len(ns)) if all(better[k:])), None)
                else:
                    for a, b in (("ce_fixed", "pearson_fixed"), ("ce_valrps", "pearson_valcorr")):
                        d = [diff_ci(sc, n, a, b, "pearson") for n in ns]
                        out["figure_test_pearson_crossovers"][f"{sc}/{a}_gt_{b}"] = next((ns[k] for k in range(len(ns)) if all(x[1] > 0 for x in d[k:])), None)
                        out["figure_test_pearson_crossovers"][f"{sc}/{b}_gt_{a}"] = next((ns[k] for k in range(len(ns)) if all(x[2] < 0 for x in d[k:])), None)
    return out


def run(fn, jobs, procs):
    with Pool(procs) as pool:
        return list(pool.imap(fn, jobs, chunksize=4))


def main():
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    t0 = time.time()
    report = {"what": __doc__.split("\n\n")[0], "exact_ce": "value -sum p log softmax(z) via log-sum-exp; gradient softmax(z) - p",
              "metric_tolerance": 1e-9}

    rows08 = load(os.path.join(E08, "results", "raw.jsonl"))
    res = run(exp08_row, [(r["scenario"], r["n"], r["seed"]) for r in rows08], procs)
    new08 = {(sc, n, s): o for sc, n, s, o, _ in res}
    pairs = [(r["fits"], new08[(r["scenario"], r["n"], r["seed"])]) for r in rows08]
    part = summarize(pairs, ["ce_fixed", "ce_valrps"], METRICS)
    part["exact_ce_fits"] = sum(st["fits"] for *_, st in res)
    part["exact_ce_fits_scipy_converged"] = sum(st["converged"] for *_, st in res)
    part["exact_ce_evaluations"] = sum(st["evals"] for *_, st in res)
    part["evaluations_with_any_category_below_1e-6"] = sum(st["any_below_floor"] for *_, st in res)
    part["rows_with_any_category_below_1e-6"] = sum(st["any_below_floor"] > 0 for *_, st in res)
    part["evaluations_with_an_observed_rating_below_1e-6"] = sum(st["observed_below_floor"] for *_, st in res)
    part["rows_with_such_an_evaluation"] = sum(st["observed_below_floor"] > 0 for *_, st in res)
    tuned = [(r, new08[(r["scenario"], r["n"], r["seed"])]["ce_valrps"]) for r in rows08]
    moved = [(r, nv) for r, nv in tuned if max(compare(r["fits"]["ce_valrps"], nv, METRICS).values()) > 1e-9]
    part["ce_valrps_rows_that_moved"] = {
        "by_chosen_ridge": {f"{c:g}": sum(nv["c"] == c for _, nv in moved) for c in cx.C_GRID},
        "by_training_tasks": {str(n): sum(r["n"] == n for r, _ in moved) for n in sorted({r["n"] for r, _ in moved})},
        "ridge_changes": [dict(scenario=r["scenario"], n=r["n"], seed=r["seed"], saved_c=r["fits"]["ce_valrps"]["c"], exact_c=nv["c"])
                          for r, nv in tuned if r["fits"]["ce_valrps"]["c"] != nv["c"]],
        "largest_test_rps_excess_change": max(abs(r["fits"]["ce_valrps"]["rps_excess"] - nv["rps_excess"]) for r, nv in tuned)}
    ns = sorted({r["n"] for r in rows08})
    patched = [dict(r, fits=dict(r["fits"], **new08[(r["scenario"], r["n"], r["seed"])])) for r in rows08]
    saved_rep, exact_rep = reported(rows08, ns), reported(patched, ns)
    part["reported_quantities_that_change"] = {
        group: {k: {"saved": saved_rep[group][k], "exact_ce": exact_rep[group][k]}
                for k in saved_rep[group] if saved_rep[group][k] != exact_rep[group][k]} for group in saved_rep}
    part["reported_quantities_compared"] = {group: len(v) for group, v in saved_rep.items()}
    part["crossovers_in_tasks"] = {k: v for k, v in exact_rep["crossovers"].items()}
    report["exp08"] = part
    print("exp08 done", round(time.time() - t0), "s", flush=True)

    rows13 = load(os.path.join(E13, "results", "raw.jsonl"))
    res = run(exp13_row, [(r["scenario"], r["n"], r["seed"]) for r in rows13], procs)
    new13 = {(sc, n, s): o for sc, n, s, o, _ in res}
    keys13 = ("rps", "rps_excess", "ce", "ce_excess", "tvd", "pearson", "spearman")
    part = {"ce_exact_vs_saved": summarize(
        [(r, new13[(r["scenario"], r["n"], r["seed"])]["exact"]) for r in rows13], ["ce"], keys13)["ce"]}
    part["exact_vs_frozen_ce_same_machine"] = summarize(
        [(o["frozen"], o["exact"]) for o in new13.values()], ["ce", "rps", "pearson", "pearson_head"], keys13)
    part["evaluations_with_any_category_below_1e-6"] = sum(st["any_below_floor"] for *_, st in res)
    part["rows_with_any_category_below_1e-6"] = sum(st["any_below_floor"] > 0 for *_, st in res)
    part["evaluations_with_an_observed_rating_below_1e-6"] = sum(st["observed_below_floor"] for *_, st in res)
    part["note_on_saved_pearson_rows"] = (
        "Not a CE effect: rerunning the frozen code on this machine reproduces the saved CE and RPS fits, but the "
        "non-convex Pearson fits (and the head fitted on their score) differ from the saved rows, which were "
        "computed on another machine; see the max differences below and experiment 07's README.")
    part["frozen_rerun_vs_saved_rows"] = summarize(
        [(r, new13[(r["scenario"], r["n"], r["seed"])]["frozen"]) for r in rows13], ["ce", "rps", "pearson", "pearson_head"], keys13)
    pop = json.load(open(os.path.join(E13, "results", "population.json")))
    part["dense_grid_reference"] = {}
    for sc in "AB":
        for kind in ("ce", "pearson_head"):
            exact = {k: float(v) for k, v in toy.population_limit(sc, kind).items()}
            frozen = {k: float(v) for k, v in with_frozen_ce(toy.population_limit, sc, kind).items()}
            part["dense_grid_reference"][f"{sc}/{kind}"] = {
                "exact_vs_frozen_same_machine": compare(frozen, exact, keys13),
                "exact_vs_saved": compare(pop[sc][kind], exact, keys13),
                "exact_rps_excess": exact["rps_excess"], "saved_rps_excess": pop[sc][kind]["rps_excess"]}
    report["exp13"] = part
    print("exp13 done", round(time.time() - t0), "s", flush=True)

    rows07 = load(os.path.join(E07, "results", "raw.jsonl"))
    res = run(exp07_row, [(r["scenario"], r["n"], r["seed"]) for r in rows07], procs)
    new07 = {(sc, n, s): o for sc, n, s, o, _ in res}
    part = {}
    for fit, conds in (("ce", ("orig", "long")), ("pearson_head", ("orig", "long", "best"))):
        for cond in conds:
            pairs = [({"x": r["fits"][fit][cond]}, {"x": new07[(r["scenario"], r["n"], r["seed"])][fit][cond]}) for r in rows07]
            s = summarize(pairs, ["x"], ("rps_excess", "ce_excess", "pearson", "spearman"))["x"]
            s["success_flag_changes"] = sum(a["x"]["success"] != b["x"]["success"] for a, b in pairs)
            part[f"{fit}/{cond}"] = s
    part["evaluations_with_any_category_below_1e-6"] = sum(st["any_below_floor"] for *_, st in res)
    part["rows_with_any_category_below_1e-6"] = sum(st["any_below_floor"] > 0 for *_, st in res)
    part["evaluations_with_an_observed_rating_below_1e-6"] = sum(st["observed_below_floor"] for *_, st in res)
    report["exp07"] = part
    report["runtime_seconds"] = round(time.time() - t0)
    out = os.path.join(E08, "results", "ce_gradient_check.json")
    with open(out, "w") as fh:
        fh.write(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
