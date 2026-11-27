"""Synthetic K-sweep check (labeled synthetic law; not human data).

Re-runs run_all.synthetic_sweep() line for line (same RNG stream), confirms the committed rows,
and adds one alternative competitor: the same ridge-mean Gaussian, but with its predictive SD
estimated from individual ratings around the fitted mean instead of from the residuals of the
k-rating MEAN (which shrink like 1/sqrt(k) and make the Gaussian overconfident as k grows).
Prints, per (K, k): committed-vs-rerun CRPS, the SD used, the corrected SD, and whether the
"CE head better (CI excludes 0)" verdict changes.
Usage: python3 09_synthetic_sigma_check.py <exp40_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

EXP40 = Path(sys.argv[1])
sys.path.insert(0, str(EXP40))
import run_all as ra  # noqa: E402

committed = json.loads(Path("/tmp/e40_cursor/experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/results.json").read_text())["synthetic"]
com_rows = {(r["K"], r["k_ratings"]): r for r in committed["rows"]}

rng = np.random.default_rng(0)
n_items = 240
u = (np.arange(n_items) + 0.5) / n_items
grid = np.arange(101, dtype=float)


def pmf_row(ui):
    d1 = np.exp(-0.5 * ((grid - 100.0 * ui) / 8.0) ** 2)
    d2 = np.exp(-0.5 * ((grid - 100.0 * (1.0 - ui)) / 12.0) ** 2)
    p = 0.7 * d1 + 0.3 * d2
    return p / p.sum()


pmf = np.stack([pmf_row(ui) for ui in u])
pool = np.stack([rng.choice(101, size=40, p=pmf[i]) for i in range(n_items)])
perm = rng.permutation(n_items)
n_tr, n_va = int(0.6 * n_items), int(0.2 * n_items)
train, val, test = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
feat = np.column_stack([np.ones(n_items), u, u ** 2])


def bin_ids(samples, k):
    if k >= 101:
        return samples.astype(int)
    edges = np.linspace(0, 100, k + 1)
    return np.clip(np.digitize(samples, edges[1:-1], right=False), 0, k - 1)


def crps(prob, y_idx):
    return np.array([ra.rps_one(prob[i], int(y_idx[i])) for i in range(prob.shape[0])])


print(f"{'K':>3} {'k':>3} | CE crps (rerun/committed) | MSE crps (rerun/committed) | sd_used sd_indiv | "
      f"verdict committed -> with individual-rating SD")
changed = 0
for k_bin in (2, 3, 5, 10, 20, 101):
    for k_rating in (1, 2, 4, 8, 16, 32):
        y = bin_ids(pool[:, :k_rating], k_bin)
        targets = np.zeros((train.size, k_bin))
        for row, i in enumerate(train):
            targets[row] = np.bincount(y[i], minlength=k_bin).astype(float)
            targets[row] /= targets[row].sum()
        x_tr = feat[train]
        d = x_tr.shape[1]

        def ce_loss(w):
            prob = ra._softmax_rows(x_tr @ w.reshape(d, k_bin))
            return float(-(targets * np.log(np.clip(prob, 1e-12, 1.0))).sum(axis=1).mean())

        sol = minimize(ce_loss, np.zeros(d * k_bin), method="L-BFGS-B", options={"maxiter": 80})
        w = sol.x.reshape(d, k_bin)
        prob_te = ra._softmax_rows(feat[test] @ w)
        fresh = np.stack([rng.choice(101, size=20, p=pmf[i]) for i in test])
        y_te = bin_ids(fresh, k_bin)
        ce_scores = np.mean([crps(prob_te, y_te[:, t]) for t in range(20)], axis=0)
        y_mean = pool[train, :k_rating].mean(axis=1)
        pen = np.eye(d); pen[0, 0] = 0.0
        beta = np.linalg.solve(x_tr.T @ x_tr + 1e-2 * pen, x_tr.T @ y_mean)
        mu_te = feat[test] @ beta
        resid = y_mean - x_tr @ beta
        sigma = float(np.sqrt(max(np.mean(resid ** 2), 1.0)))
        # Corrected: SD of individual training ratings around the fitted mean.
        mu_tr = x_tr @ beta
        sigma_ind = float(np.sqrt(max(np.mean((pool[train, :k_rating] - mu_tr[:, None]) ** 2), 1.0)))
        centers = (np.linspace(0, 100, k_bin + 1)[:-1] + np.linspace(0, 100, k_bin + 1)[1:]) / 2 if k_bin < 101 else grid

        def gauss_scores(sd):
            dens = np.exp(-0.5 * ((centers[None, :] - mu_te[:, None]) / sd) ** 2)
            dens /= dens.sum(axis=1, keepdims=True)
            return np.mean([crps(dens, y_te[:, t]) for t in range(20)], axis=0)

        mse_scores = gauss_scores(sigma)
        mse_fix = gauss_scores(sigma_ind)
        diff = ra.paired_diff(mse_scores, ce_scores, boot_seed=1)
        diff_fix = ra.paired_diff(mse_fix, ce_scores, boot_seed=1)
        cr = com_rows[(k_bin, k_rating)]
        v_old = diff["lo"] > 0
        v_new = diff_fix["lo"] > 0
        changed += v_old != v_new
        print(f"{k_bin:>3} {k_rating:>3} | {ce_scores.mean():.4f}/{cr['ce_crps']:.4f} | {mse_scores.mean():.4f}/{cr['mse_crps']:.4f} | "
              f"{sigma:6.2f} {sigma_ind:6.2f} | CE-better-CI {v_old} -> {v_new} "
              f"(MSE-CE {diff['point']:.4f} -> {diff_fix['point']:.4f} [{diff_fix['lo']:.4f},{diff_fix['hi']:.4f}])")
print("verdicts changed by the SD correction:", changed)
