"""Independent re-analysis of the experiment 40 allocation study.

Re-fits every allocation cell exactly as run_all.allocation() does (same helpers, seeds and
order), keeps the per-prompt loss vectors that run_all discards, and then computes:

  R. reproduction of every committed cell (test CE, val CE) and of the per-budget picks/intervals;
  A. D-study minus VALIDATION-chosen allocation (the deployable comparator), item- and group-clustered;
  B. bootstrap estimate of the selection bias of "D-study minus test-oracle" (oracle re-selected in
     each replicate), and a bias-corrected point estimate;
  C. two-sided bootstrap p-values (10,000 item-clustered draws) for every D-minus-oracle cell, Holm
     adjustment over all cells and over the 7 largest-budget (headline) cells;
  D. a constant "train marginal" predictor's test CE per dataset (what a head that learns nothing
     from the text scores), against the full-train head;
  E. L-BFGS-B convergence diagnostics (success flag, iterations at the 35 cap);
  F. exact ties among the options of each budget.

Usage: python3 05_allocation_stats.py <exp40_dir> <out_json>
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

EXP40 = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(EXP40))
import run_all as ra  # noqa: E402  (also puts experiment 36 on sys.path)

COMMITTED = json.loads(Path("/tmp/e40_cursor/experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/results.json").read_text())
BOOT = 1000
P_DRAWS = 10000


def boot_mean_draws(n, B, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(B, n))


def group_draws(groups_of_test, B, seed):
    """Cluster bootstrap over the split groups (article / item / task) of the test prompts."""
    uniq = sorted(set(groups_of_test))
    members = {g: np.flatnonzero(np.asarray(groups_of_test) == g) for g in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        out.append(np.concatenate([members[uniq[j]] for j in pick]))
    return out


def pct(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        run = max(run, val)
        adj[idx] = run
    return adj


def main():
    t0 = time.time()
    specs = [
        ("summeval_expert", lambda: ra.load_summeval("expert"), "vote_index"),
        ("summeval_crowd", lambda: ra.load_summeval("crowd"), "vote_index"),
        ("popquorn_offensiveness", lambda: ra.load_popquorn("offensiveness"), "crossed_person"),
        ("popquorn_politeness", lambda: ra.load_popquorn("politeness"), "crossed_person"),
        ("dices990", lambda: ra._load_dices("990"), "crossed_person"),
        ("chaosnli_snli_mnli", lambda: ra.load_chaos_3way(), "vote_index"),
        ("veribench_fsc_native_main", lambda: ra.load_veribench_fsc_native_main(), "crossed_person"),
    ]
    report = {"datasets": {}, "all_cells": []}
    for name, loader, facet in specs:
        data = loader()
        if name == "veribench_fsc_native_main":
            sp = ra.exp27_split(data)
            train, val, test = sp["train"], sp["val"], sp["test"]
            assert not sp["missing_groups"] and not sp["mixed_tasks"]
        else:
            train, val, test = ra.split_by_group(data.groups, seed=0)
        g_train = ra.run_gstudy(data, train, facet)
        k_star = g_train["recommendation"]["k"]
        feats = ra.hash_features(data.texts)
        ks = sorted(set(ra.K_GRID) | ({int(k_star)} if k_star else set()))
        orders = {}
        for k in ks:
            elig = np.asarray([int(i) for i in train if data.ratings[int(i)].size >= k], dtype=int)
            if elig.size < 4:
                continue
            orders[k] = np.random.default_rng(0).permutation(elig)
        total_train = int(sum(data.ratings[int(i)].size for i in train))
        budgets = [b for b in ra.BUDGETS if b <= total_train] or [total_train]
        cache, cells = {}, []
        for k, order in orders.items():
            for B in budgets:
                n_take = min(int(order.size), int(B) // int(k))
                if n_take < 4:
                    continue
                key = (int(k), int(n_take))
                if key not in cache:
                    fitted = ra.fit_cell(feats, data, order[:n_take], [val, test], "ce", int(k))
                    pack, (x_val, x_test) = fitted
                    ce_te, rps_te = ra.item_losses(pack, x_test, data.ratings, test)
                    ce_va, _ = ra.item_losses(pack, x_val, data.ratings, val)
                    cache[key] = {"k": int(k), "N": int(n_take), "ce_te": ce_te, "rps_te": rps_te, "ce_va": ce_va,
                                  "success": bool(pack.get("success")), "nit": int(pack.get("nit", -1))}
                cells.append((int(B), key))
        # Full-train reference and a constant train-marginal predictor.
        full = ra.fit_cell(feats, data, np.asarray(train), [val, test], "ce", None)
        pack, (_xv, x_test) = full
        full_ce, _ = ra.item_losses(pack, x_test, data.ratings, test)
        Kc = len(data.class_values)
        cnt = np.zeros(Kc)
        for i in train:  # item-equal marginal, as the soft targets are item-equal
            y = np.asarray(data.ratings[int(i)], int)
            cnt += np.bincount(y, minlength=Kc) / y.size
        marg = cnt / cnt.sum()
        marg_ce = np.array([np.mean([-math.log(max(marg[int(y)], 1e-12)) for y in data.ratings[int(i)]]) for i in test])
        test_groups = [data.groups[int(i)] for i in test]
        gdraws = group_draws(test_groups, BOOT, 7)
        n = len(test)
        idraws = boot_mean_draws(n, BOOT, 1)
        pdraws_chunks = [boot_mean_draws(n, 2000, 100 + c) for c in range(P_DRAWS // 2000)]
        committed = COMMITTED["datasets"][name]["allocation"]
        com_cells = {(c["k"], c["N"]): c for c in committed["cells"]}
        repro_max_abs = 0.0
        for key, c in cache.items():
            cc = com_cells.get(key)
            if cc is None:
                repro_max_abs = float("inf")
                continue
            repro_max_abs = max(repro_max_abs, abs(float(np.mean(c["ce_te"])) - cc["test"]["ce"]),
                                abs(float(np.mean(c["ce_va"])) - cc["val_ce"]))
        rows = []
        for row_c in committed["budgets"]:
            B = row_c["B"]
            keys = [key for (bb, key) in cells if bb == B]
            opts = [cache[key] for key in keys]
            te = np.array([np.mean(o["ce_te"]) for o in opts])
            va = np.array([np.mean(o["ce_va"]) for o in opts])
            i_or = int(np.argmin(te))
            i_va = int(np.argmin(va))
            n_ties_test = int(np.sum(np.abs(te - te[i_or]) < 1e-12)) - 1
            d_idx = [j for j, o in enumerate(opts) if o["k"] == k_star]
            r = {"B": B, "options": [(o["k"], o["N"], o["k"] * o["N"]) for o in opts],
                 "oracle_k": opts[i_or]["k"], "val_k": opts[i_va]["k"], "exact_ties_with_oracle": n_ties_test,
                 "matches_committed": opts[i_or]["k"] == row_c["oracle_ce"]["k"] and opts[i_va]["k"] == row_c["validation_chosen"]["k"]}
            if d_idx:
                dj = d_idx[0]
                D = opts[dj]["ce_te"]
                O = opts[i_or]["ce_te"]
                V = opts[i_va]["ce_te"]
                # R: reproduce the committed fixed-pair interval exactly.
                fixed = D[idraws].mean(1) - O[idraws].mean(1)
                r["dk"] = opts[dj]["k"]
                r["d_minus_oracle_point"] = float(D.mean() - O.mean())
                r["d_minus_oracle_ci_fixed_pair"] = pct(fixed)
                if row_c["dstudy_minus_oracle_ce"]:
                    cm = row_c["dstudy_minus_oracle_ce"]
                    r["repro_ci_abs_err"] = max(abs(r["d_minus_oracle_ci_fixed_pair"][0] - cm["lo"]),
                                                abs(r["d_minus_oracle_ci_fixed_pair"][1] - cm["hi"]),
                                                abs(r["d_minus_oracle_point"] - cm["point"]))
                # C: two-sided bootstrap p-value of the fixed pair (10,000 draws).
                le = ge = tot = 0
                for dr in pdraws_chunks:
                    s = D[dr].mean(1) - O[dr].mean(1)
                    le += int(np.sum(s <= 0)); ge += int(np.sum(s >= 0)); tot += s.size
                p_two = min(1.0, 2 * min((le + 1) / (tot + 1), (ge + 1) / (tot + 1)))
                same = opts[dj]["k"] == opts[i_or]["k"] and opts[dj]["N"] == opts[i_or]["N"]
                r["p_two_sided_fixed_pair"] = 1.0 if same else p_two
                # B: selection-aware bootstrap (oracle re-picked inside each replicate).
                M = np.stack([o["ce_te"] for o in opts])  # options x n
                sel = np.array([M[dj, dr].mean() - M[:, dr].mean(1).min() for dr in idraws])
                r["selection_bias_estimate"] = float(sel.mean() - r["d_minus_oracle_point"])
                r["d_minus_best_bias_corrected"] = float(2 * r["d_minus_oracle_point"] - sel.mean())
                # A: deployable comparison, D-study minus validation-chosen.
                dv = D[idraws].mean(1) - V[idraws].mean(1)
                dvg = np.array([D[g].mean() - V[g].mean() for g in gdraws])
                r["d_minus_val_point"] = float(D.mean() - V.mean())
                r["d_minus_val_ci_item"] = pct(dv)
                r["d_minus_val_ci_group"] = pct(dvg)
                fixg = np.array([D[g].mean() - O[g].mean() for g in gdraws])
                r["d_minus_oracle_ci_group"] = pct(fixg)
                report["all_cells"].append({"dataset": name, "B": B, "p": r["p_two_sided_fixed_pair"],
                                            "point": r["d_minus_oracle_point"], "ci": r["d_minus_oracle_ci_fixed_pair"],
                                            "largest_B": B == committed["budgets"][-1]["B"], "same_cell": same})
            rows.append(r)
        conv = [(c["success"], c["nit"]) for c in cache.values()]
        report["datasets"][name] = {
            "k_star": k_star, "n_test": n, "n_test_groups": len(set(test_groups)),
            "reproduction_max_abs_err_cell_ce": repro_max_abs,
            "full_train_test_ce": float(np.mean(full_ce)), "train_marginal_test_ce": float(marg_ce.mean()),
            "full_minus_marginal_ce": float(np.mean(full_ce) - marg_ce.mean()),
            "full_minus_marginal_ci_item": pct(full_ce[idraws].mean(1) - marg_ce[idraws].mean(1)),
            "n_fits": len(conv), "n_fits_success_false": sum(1 for s, _ in conv if not s),
            "n_fits_at_iter_cap_35": sum(1 for _, it in conv if it >= 35),
            "rows": rows,
        }
        print(f"[{time.time()-t0:.0f}s] {name}: repro max abs err {repro_max_abs:.2e}; full CE {np.mean(full_ce):.4f} vs marginal {marg_ce.mean():.4f}", flush=True)
    cells = [c for c in report["all_cells"] if not c["same_cell"]]
    p = np.array([c["p"] for c in cells])
    adj = holm(p)
    for c, a in zip(cells, adj):
        c["holm_all"] = float(a)
    big = [c for c in cells if c["largest_B"]]
    adj_big = holm(np.array([c["p"] for c in big])) if big else []
    for c, a in zip(big, adj_big):
        c["holm_largest_B"] = float(a)
    report["n_tests_all"] = len(cells)
    report["n_tests_largest_B"] = len(big)
    OUT.write_text(json.dumps(report, indent=1, default=float))
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
