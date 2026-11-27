"""Run the proof checks and the real multi-annotator experiments.

Pre-registered choices (see PROTOCOL.md), fixed before this process prints a test metric:
hash dimension 256, 24 principal components fit on train prompts only, L2 penalty 1e-2
on non-intercept weights, L-BFGS-B from the zero vector, at most 35 iterations,
split seed 0 with fractions 0.6/0.2/0.2 by prompt group, bootstrap seed 1 with
B=1000 item-clustered resamples, nested rating subsample seed 2.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from data_io import (
    assert_no_prompt_leakage,
    availability,
    load_all,
    load_certjudge_real_panels,
    load_chaos_alpha,
    load_summeval,
    split_by_group,
)
from fit_eval import (
    OBJECTIVES,
    constant_marginal,
    design_matrices,
    evaluate_constant,
    evaluate_pack,
    fit_objective,
    predict_distributions,
    rollout_from_probs,
)
from metrics import brier_one, ce_one, rps_one
from proofs_numeric import check_all

HERE = Path(__file__).resolve().parent
SMOKE = os.environ.get("SANMI36_SMOKE") == "1"
BOOT = 40 if SMOKE else 1000
KS = [1, 2, 4] if SMOKE else [1, 2, 4, 8, 16, 32]


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None if np.isnan(obj) else ("inf" if obj > 0 else "-inf")
        return obj
    return obj


def _subsample_indices(train_idx, n_keep, seed):
    rng = np.random.default_rng(seed)
    take = min(n_keep, train_idx.size)
    choice = rng.choice(train_idx, size=take, replace=False)
    return np.sort(choice)


def prepare(data):
    train, val, test = split_by_group(data.groups, seed=0)
    assert_no_prompt_leakage(data.groups, data.rater_ids, train, val, test)
    x_train, (x_val, x_test), rank = design_matrices(data.texts, train, [val, test])
    return {
        "train": train, "val": val, "test": test,
        "x_train": x_train, "x_val": x_val, "x_test": x_test, "rank": rank,
    }


def fit_and_eval(data, prep, objective, k_keep, seed, bootstrap=BOOT):
    pack = fit_objective(
        prep["x_train"], data.ratings, prep["train"], data.class_values,
        objective, k_keep, seed,
    )
    ev = evaluate_pack(pack, prep["x_test"], data.ratings, prep["test"], bootstrap=bootstrap, boot_seed=1)
    ev["optimizer"] = {
        "success": pack.get("success"),
        "nit": pack.get("nit"),
        "convex_claim": pack.get("convex_claim"),
        "closed_form": pack.get("closed_form"),
        "objective": objective,
        "k_keep": k_keep,
        "seed": seed,
    }
    return pack, ev


def matrix_for(data):
    print(f"[matrix] {data.name} items={data.n_items}", flush=True)
    prep = prepare(data)
    objectives = ("ce", "pearson") if SMOKE else OBJECTIVES
    rows = {}
    packs = {}
    for obj in objectives:
        t0 = time.time()
        pack, ev = fit_and_eval(data, prep, obj, None, seed=0)
        rows[obj] = ev
        packs[obj] = pack
        print(f"  {obj} test CE {ev['point']['ce']:.4f} in {time.time()-t0:.1f}s", flush=True)
    prob, mu = constant_marginal(data.ratings, prep["train"], data.n_classes, data.class_values)
    marginal = evaluate_constant(prob, mu, data.ratings, prep["test"], data.class_values, bootstrap=BOOT)
    uniform_p = np.full(data.n_classes, 1.0 / data.n_classes)
    uniform = evaluate_constant(uniform_p, float(uniform_p @ data.class_values), data.ratings, prep["test"], data.class_values, bootstrap=BOOT)
    # Shuffle control: break the alignment of train features and labels, refit CE.
    rng = np.random.default_rng(7)
    shuffled = prep["x_train"].copy()
    rng.shuffle(shuffled)
    prep_s = dict(prep)
    prep_s["x_train"] = shuffled
    _, shuffle_ev = fit_and_eval(data, prep_s, "ce", None, seed=0)
    trace = _trace(data, prep, packs.get("ce"))
    return {
        "summary": data.summary(),
        "n_train": int(prep["train"].size),
        "n_val": int(prep["val"].size),
        "n_test": int(prep["test"].size),
        "svd_rank_used": int(prep["rank"]),
        "rows": {k: {"point": v["point"], "ci": v["ci"], "optimizer": v["optimizer"]} for k, v in rows.items()},
        "marginal": marginal,
        "uniform": uniform,
        "shuffle_ce": shuffle_ev["point"]["ce"],
        "trace": trace,
        "_prep": prep,
        "_packs": packs,
    }


def _trace(data, prep, pack):
    if pack is None:
        return None
    i = int(prep["test"][0])
    prob, mu = predict_distributions(pack, prep["x_test"][:1], "ce" if pack.get("objective") != "mse" else "mse")
    q = prob[0]
    y = data.ratings[i]
    return {
        "item_index": i,
        "group": data.groups[i],
        "text_prefix": data.texts[i][:180],
        "ratings_class_index": y.tolist(),
        "q": [float(v) for v in q],
        "mu": float(mu[0]),
        "ce": float(np.mean([ce_one(q, int(t)) for t in y])),
        "brier": float(np.mean([brier_one(q, int(t)) for t in y])),
        "rps": float(np.mean([rps_one(q, int(t)) for t in y])),
    }


def crossover(data, label):
    print(f"[crossover] {label}", flush=True)
    prep = prepare(data)
    # Use a high percentile if the minimum is 1 because of a thin item, but never exceed the median.
    ks_avail = [int(np.median([r.size for r in data.ratings]))]
    cap = int(np.percentile([r.size for r in data.ratings], 20))
    use = [k for k in KS if k <= cap]
    if not use:
        use = [1]
    curves = {obj: [] for obj in ("ce", "pearson")}
    for k in use:
        for obj in ("ce", "pearson"):
            t0 = time.time()
            _pack, ev = fit_and_eval(data, prep, obj, k, seed=2)
            curves[obj].append({
                "k": k,
                "point": ev["point"],
                "ci": {m: ev["ci"][m] for m in ("ce", "brier", "sqrt_rps", "rps_norm", "pearson_r")},
            })
            print(f"  k={k} {obj} CE {ev['point']['ce']:.4f} ({time.time()-t0:.1f}s)", flush=True)
    return {"label": label, "k_cap_p20": cap, "median_k": ks_avail[0], "curves": curves}


def iso_budget(data, ks=(1, 4, 16, 32)):
    print("[iso-budget]", data.name, flush=True)
    prep = prepare(data)
    n_train = int(prep["train"].size)
    # Exact products B = N * k inside the train pool.
    budget = 1024 if n_train >= 1024 else max(64, (n_train // 16) * 16)
    plans = []
    for k in ks:
        n = budget // k
        if n >= 30 and n <= n_train and k <= int(np.percentile([r.size for r in data.ratings], 20)):
            plans.append((n, k, n * k))
    rows = []
    for n, k, b in plans:
        idx = _subsample_indices(prep["train"], n, seed=3 + k)
        # Refit the projection on the chosen prompts only.
        x_tr, (x_te,), rank = design_matrices(data.texts, idx, [prep["test"]])
        local = {"train": idx, "test": prep["test"], "x_train": x_tr, "x_test": x_te, "rank": rank}
        for obj in ("ce", "pearson"):
            _pack, ev = fit_and_eval(data, local, obj, k, seed=4)
            rows.append({
                "N": n, "k": k, "B": b, "objective": obj,
                "point": ev["point"],
                "ci": {m: ev["ci"][m] for m in ("ce", "brier", "sqrt_rps", "rps_norm")},
            })
            print(f"  N={n} k={k} {obj} CE {ev['point']['ce']:.4f}", flush=True)
    return {"budget_target": budget, "rows": rows}


def within_prompt(data):
    """Labeled oracle: half the raters of a test prompt estimate q, the other half are scored.

    This is not a deployable judge. It checks the anti-proper comparison on real ratings.
    """
    _train, _val, test = split_by_group(data.groups, seed=0)
    rng = np.random.default_rng(5)
    xs, ys, xds, ce_q, ce_c, br_q, br_c = [], [], [], [], [], [], []
    used = 0
    for i in test:
        y = np.asarray(data.ratings[i], dtype=int)
        if y.size < 4:
            continue
        order = rng.permutation(y.size)
        half = y.size // 2
        a = y[order[:half]]
        b = y[order[half:]]
        counts = np.bincount(a, minlength=data.n_classes).astype(float)
        # Add-one so the oracle distribution is positive.
        q = (counts + 1.0) / (counts.sum() + data.n_classes)
        mu = float(counts @ data.class_values / counts.sum())
        # One draw from q for each held-out rating, independent of that rating given the prompt.
        draws = rng.choice(data.n_classes, size=b.size, p=q)
        x_star = data.class_values[draws]
        x_delta = 0.1 * mu + 4.0
        collapsed = one_hot_round(mu, data.class_values)
        for yj, xsamp in zip(b, x_star):
            xs.append(float(xsamp))
            ys.append(float(data.class_values[int(yj)]))
            xds.append(float(x_delta))
            ce_q.append(ce_one(q, int(yj)))
            ce_c.append(ce_one(collapsed, int(yj)))
            br_q.append(brier_one(q, int(yj)))
            br_c.append(brier_one(collapsed, int(yj)))
        used += 1
    if len(xs) < 10:
        return {"ok": False, "reason": "too few within-prompt pairs", "n_items": used}
    xs, ys, xds = np.asarray(xs), np.asarray(ys), np.asarray(xds)
    def corr(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])
    return {
        "ok": True,
        "labeled": "within-prompt oracle, not a held-out judge",
        "n_items": used,
        "n_pairs": int(ys.size),
        "corr_sampled_halfA": corr(xs, ys),
        "corr_collapsed_affine": corr(xds, ys),
        "sqrt_gap": bool(corr(xds, ys) > corr(xs, ys)),
        "mean_ce_distributional": float(np.mean(ce_q)),
        "mean_ce_collapsed": float(np.mean(ce_c)),
        "mean_brier_distributional": float(np.mean(br_q)),
        "mean_brier_collapsed": float(np.mean(br_c)),
    }


def one_hot_round(mu, values):
    from metrics import one_hot
    idx = int(np.argmin(np.abs(values - mu)))
    return one_hot(len(values), idx, floor=1e-6)


def rollout_experiment(pack, x_test, ratings, test_idx):
    prob, _mu = predict_distributions(pack, x_test, pack["objective"])
    rng = np.random.default_rng(9)
    rows = []
    # Single pass: the softmax itself.
    rows.append(_score_probs("single_pass", prob, ratings, test_idx))
    for m in (1, 5, 20):
        acc = []
        for rep in range(8 if not SMOKE else 2):
            q = rollout_from_probs(prob, m, rng)
            # A miss at M=1 can put a zero on the realized class. Floor only the Monte Carlo
            # histogram so the table stays finite, and record the floor.
            q = (1 - 1e-6) * q + 1e-6 / q.shape[1]
            acc.append(_score_probs(f"M{m}", q, ratings, test_idx)["ce"])
        point = _score_probs(f"M{m}", rollout_from_probs(prob, m, np.random.default_rng(11 + m)), ratings, test_idx)
        point["ce_mean_over_reps"] = float(np.mean(acc))
        point["M"] = m
        rows.append(point)
    return rows


def _score_probs(name, prob, ratings, test_idx):
    ces, brs, rps = [], [], []
    for row, i in enumerate(test_idx):
        q = prob[row]
        y = ratings[i]
        ces.append(float(np.mean([ce_one(q, int(t)) for t in y])))
        brs.append(float(np.mean([brier_one(q, int(t)) for t in y])))
        rps.append(float(np.mean([rps_one(q, int(t)) for t in y])))
    return {
        "readout": name,
        "ce": float(np.mean(ces)),
        "brier": float(np.mean(brs)),
        "rps": float(np.mean(rps)),
        "sqrt_rps": float(np.sqrt(np.mean(rps))),
    }


def main():
    t0 = time.time()
    proofs = check_all()
    print("[proofs numeric] ok", flush=True)
    if SMOKE:
        sets = [load_summeval("expert"), load_chaos_alpha()]
    else:
        sets = load_all()
    by_name = {s.name: s for s in sets}
    print("[data]", {k: v.summary()["n_items"] for k, v in by_name.items()}, flush=True)
    matrix_names = ["summeval_expert", "chaosnli_alpha"] if SMOKE else [
        "summeval_expert", "summeval_expert_binary",
        "chaosnli_snli_mnli", "chaosnli_alpha",
        "dices990", "dices990_binary_yes",
        "popquorn_offensiveness", "popquorn_offensiveness_binary",
        "popquorn_politeness", "popquorn_politeness_binary",
        "veribench_fsc_rv1", "veribench_fsc_rv1_expert3", "veribench_fsc_rv1_binary",
        "veribench_fsc_native_main",
        "veribench_fsc_pooled", "veribench_fsc_pooled_binary",
    ]
    matrices = {}
    for name in matrix_names:
        matrices[name] = matrix_for(by_name[name])
    curves = {}
    if not SMOKE:
        curves["binary_alpha"] = crossover(by_name["chaosnli_alpha"], "ChaosNLI alphaNLI binary")
        curves["ordinal_summeval_crowd"] = crossover(by_name["summeval_crowd"], "SummEval crowd Likert 1-5")
        curves["ordinal_dices990"] = crossover(by_name["dices990"], "DICES-990 3-level ordinal")
        curves["chaos_3way"] = crossover(by_name["chaosnli_snli_mnli"], "ChaosNLI SNLI+MNLI 3-class")
        curves["ordinal_popquorn_offensiveness"] = crossover(by_name["popquorn_offensiveness"], "POPQUORN offensiveness 1-5")
        curves["ordinal_popquorn_politeness"] = crossover(by_name["popquorn_politeness"], "POPQUORN politeness 1-5")
        curves["ordinal_veribench_fsc_rv1"] = crossover(by_name["veribench_fsc_rv1"], "VeriBench FSC R_v1 (0-5, all5)")
        curves["ordinal_veribench_fsc_pooled"] = crossover(by_name["veribench_fsc_pooled"], "VeriBench FSC pooled (0-5)")
        iso = iso_budget(by_name["chaosnli_snli_mnli"])
        iso_extra = {
            "popquorn_offensiveness": iso_budget(by_name["popquorn_offensiveness"], ks=(1, 2, 4, 8)),
            "popquorn_politeness": iso_budget(by_name["popquorn_politeness"], ks=(1, 2, 4)),
        }
    else:
        curves["binary_alpha"] = crossover(by_name["chaosnli_alpha"], "ChaosNLI alphaNLI binary")
        curves["ordinal_summeval"] = crossover(by_name["summeval_expert"], "SummEval expert")
        iso = {"budget_target": None, "rows": []}
        iso_extra = {}
    # Rollout and within-prompt on the first matrix that has a CE pack.
    primary = "summeval_expert"
    roll = rollout_experiment(
        matrices[primary]["_packs"]["ce"],
        matrices[primary]["_prep"]["x_test"],
        by_name[primary].ratings,
        matrices[primary]["_prep"]["test"],
    )
    oracles_all = {}
    for o_name in ("dices990", "popquorn_offensiveness", "popquorn_politeness", "veribench_fsc_rv1", "veribench_fsc_pooled"):
        if o_name in by_name:
            oracles_all[o_name] = within_prompt(by_name[o_name])
    oracle_name = "dices990" if "dices990" in by_name else primary
    oracle = oracles_all[oracle_name] if oracle_name in oracles_all else within_prompt(by_name[oracle_name])
    certjudge_panels = load_certjudge_real_panels()
    # Drop non-serializable handles.
    for block in matrices.values():
        block.pop("_prep", None)
        block.pop("_packs", None)
    payload = {
        "smoke": SMOKE,
        "bootstrap_B": BOOT,
        "availability": availability(),
        "proofs_numeric": proofs,
        "datasets": {s.name: s.summary() for s in sets},
        "matrices": matrices,
        "crossover": curves,
        "iso_budget": iso,
        "iso_budget_extra": iso_extra,
        "rollout": {"dataset": primary, "rows": roll, "floor_on_monte_carlo": 1e-6},
        "within_prompt_oracle": {"dataset": oracle_name, **oracle},
        "within_prompt_oracles": oracles_all,
        "certjudge_real_panels": certjudge_panels,
        "seconds": time.time() - t0,
    }
    text = json.dumps(_jsonable(payload), indent=2)
    (HERE / "results.json").write_text(text)
    _write_results_txt(payload)
    from make_figures import make_all
    make_all(payload, HERE / "figures")
    from write_reports import write_all
    write_all(payload, HERE)
    print(f"[done] {time.time()-t0:.1f}s", flush=True)


def _write_results_txt(payload):
    lines = ["Sanmi distributional pitch — machine summary", ""]
    for name, block in payload["matrices"].items():
        lines.append(f"## {name}  train {block['n_train']} val {block['n_val']} test {block['n_test']}")
        for obj, row in block["rows"].items():
            p = row["point"]
            ci = row["ci"]["ce"]
            lines.append(
                f"  {obj:8} CE {p['ce']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]  "
                f"sqrtRPS {p['sqrt_rps']:.4f}  Brier {p['brier']:.4f}  "
                f"r {p['pearson_r']}  rho {p['spearman_rho']}  tau {p['kendall_tau']}"
            )
        lines.append(f"  uniform CE {block['uniform']['point']['ce']:.4f}  marginal CE {block['marginal']['point']['ce']:.4f}  shuffle CE {block['shuffle_ce']:.4f}")
        lines.append("")
    (HERE / "results.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
