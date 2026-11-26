"""Linear distributional heads on hashed text, and the held-out metric panel.

The search space is a softmax linear head on 24 principal components of a
256-dimensional signed token hash, plus an unpenalized intercept. Cross-entropy
in the logits is convex, so L-BFGS from the zero start is a convex program up
to the iteration cap. Brier, ranked probability score (RPS) and the Pearson
surrogate are not claimed to be convex. Scalar mean squared error (MSE) is a
ridge regression, which has a closed form, and is turned into a distribution
by the two-point grid distribution with that mean.

Hyperparameters below are pre-registered in PROTOCOL.md. Validation is not used
to choose them.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
from scipy.optimize import minimize

from metrics import (
    brier_one,
    ce_one,
    kendall_tau,
    one_hot,
    pearson_from_sums,
    percentile_interval,
    rps_one,
    softmax,
    spearman_rho,
    two_point_on_grid,
    unbiased_risk,
)

HASH_DIM = 256
N_COMP = 24
L2 = 1e-2
MAXITER = 35
PROB_FLOOR = 1e-6
TOKEN = re.compile(r"[a-z0-9]+")
OBJECTIVES = ("ce", "brier", "rps", "pearson", "mse")


def hash_features(texts: list[str], dim: int = HASH_DIM) -> np.ndarray:
    x = np.zeros((len(texts), dim), dtype=np.float64)
    for i, text in enumerate(texts):
        toks = TOKEN.findall(text.lower())
        grams = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        if not grams:
            grams = ["empty"]
        for gram in grams:
            digest = hashlib.md5(gram.encode()).hexdigest()
            idx = int(digest[:8], 16) % dim
            sign = 1.0 if int(digest[8], 16) % 2 == 0 else -1.0
            x[i, idx] += sign
        norm = np.linalg.norm(x[i])
        if norm > 0:
            x[i] /= norm
    return x


def _svd_project(x_train, x_other, n_comp=N_COMP):
    mu = x_train.mean(axis=0)
    centered = x_train - mu
    # Economy SVD. Components are fit on train prompts only.
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(s > 1e-8))
    k = max(1, min(n_comp, rank))
    basis = vt[:k].T

    def project(x):
        z = (x - mu) @ basis
        return np.concatenate([z, np.ones((z.shape[0], 1))], axis=1)

    return project(x_train), [project(x) for x in x_other], k


def _soft_targets(ratings: list[np.ndarray], indices: np.ndarray, k_classes: int, k_keep: int | None, rng):
    """Item-equal soft labels from a nested prefix of a permutation of the item's ratings."""
    targets = []
    ybar = []
    kept = []
    for i in indices:
        y = np.asarray(ratings[i], dtype=int)
        order = rng.permutation(y.size)
        take = y.size if k_keep is None else min(k_keep, y.size)
        chosen = y[order[:take]]
        counts = np.bincount(chosen, minlength=k_classes).astype(float)
        targets.append(counts / counts.sum())
        kept.append(chosen)
        ybar.append(float(chosen.mean()))
    return np.stack(targets), np.asarray(ybar), kept


def _penalty(w, d, k, l2=L2):
    weight = w.reshape(d, k)
    # The last row is the intercept.
    return 0.5 * l2 * float(np.sum(weight[:-1] ** 2))


def _fit_softmax(x, targets, ybar, values, objective: str):
    n, d = x.shape
    k = targets.shape[1]
    values = np.asarray(values, dtype=float)

    def loss(w):
        prob = softmax(x @ w.reshape(d, k))
        pen = _penalty(w, d, k)
        if objective == "ce":
            p = np.clip(prob, 1e-12, 1.0)
            return float(-(targets * np.log(p)).sum(axis=1).mean() + pen)
        if objective == "brier":
            # Expected one-hot Brier under the item's empirical distribution.
            brier = np.sum(prob * prob, axis=1) + 1.0 - 2.0 * np.sum(targets * prob, axis=1)
            return float(brier.mean() + pen)
        if objective == "rps":
            fq = np.cumsum(prob, axis=1)[:, :-1]
            ft = np.cumsum(targets, axis=1)[:, :-1]
            return float(np.sum((fq - ft) ** 2, axis=1).mean() + pen)
        if objective == "pearson":
            mu = prob @ values
            var_c = prob @ (values ** 2) - mu ** 2
            mu_c = mu - mu.mean()
            y_c = ybar - ybar.mean()
            cov = float(np.mean(mu_c * y_c))
            var_x = float(np.mean(var_c) + np.mean(mu_c ** 2))
            var_y = float(np.mean((ybar - ybar.mean()) ** 2))
            # Within-item human variance is not in ybar; use a floor so a constant mean is not rewarded.
            # The sample correlation uses Var(X) = E[Var(X|c)] + Var(E[X|c]), which is var_x above,
            # and Var(Y) estimated from the item means only would ignore within-item noise and would
            # not punish spread. Add the mean within-item second moment of the soft target.
            second = targets @ (values ** 2)
            var_y = float(np.mean(second - (targets @ values) ** 2) + np.var(targets @ values))
            if var_x <= 1e-12 or var_y <= 1e-12:
                return pen
            return float(-cov / np.sqrt(var_x * var_y) + pen)
        raise ValueError(objective)

    result = minimize(loss, np.zeros(d * k), method="L-BFGS-B", options={"maxiter": MAXITER, "ftol": 1e-9})
    weight = result.x.reshape(d, k)
    return {
        "weight": weight,
        "success": bool(result.success),
        "nit": int(result.nit),
        "fun": float(result.fun),
        "objective": objective,
        "convex_claim": objective == "ce",
    }


def _fit_ridge(x, y, l2=L2):
    d = x.shape[1]
    penalty = np.eye(d)
    penalty[-1, -1] = 0.0  # intercept
    xtx = x.T @ x + l2 * penalty
    xty = x.T @ y
    beta = np.linalg.solve(xtx, xty)
    return beta


def predict_distributions(pack, x, objective: str):
    values = pack["values"]
    if objective == "mse":
        mu = x @ pack["beta"]
        return two_point_on_grid(mu, values), mu
    prob = softmax(x @ pack["weight"])
    mu = prob @ values
    return prob, mu


def fit_objective(x_train, ratings, train_idx, values, objective, k_keep, seed):
    k_classes = len(values)
    rng = np.random.default_rng(seed)
    targets, ybar, kept = _soft_targets(ratings, train_idx, k_classes, k_keep, rng)
    # ybar above is the mean of class indices. Convert to the value scale.
    ybar_value = np.array([float(values[ys].mean()) for ys in kept])
    if objective == "mse":
        beta = _fit_ridge(x_train, ybar_value)
        return {"objective": "mse", "beta": beta, "values": np.asarray(values, dtype=float),
                "success": True, "nit": 0, "convex_claim": True, "closed_form": "ridge"}
    fitted = _fit_softmax(x_train, targets, ybar_value, values, objective)
    fitted["values"] = np.asarray(values, dtype=float)
    return fitted


def _item_records(prob, mu, ratings, indices, values):
    """Per-prompt losses and the sums needed for a pooled Pearson correlation."""
    ce, br, rp = [], [], []
    stats = []
    human_mean = []
    scalar = []
    for row, i in enumerate(indices):
        q = prob[row]
        y = np.asarray(ratings[i], dtype=int)
        c = b = r = 0.0
        ys = values[y]
        s = float(mu[row])
        for yj, yv in zip(y, ys):
            c += ce_one(q, int(yj))
            b += brier_one(q, int(yj))
            r += rps_one(q, int(yj))
        k = float(y.size)
        ce.append(c / k)
        br.append(b / k)
        rp.append(r / k)
        stats.append((k, float(ys.sum()), float(np.dot(ys, ys)), s * k, (s * s) * k, float(s * ys.sum())))
        human_mean.append(float(ys.mean()))
        scalar.append(s)
    return {
        "ce": np.asarray(ce),
        "brier": np.asarray(br),
        "rps": np.asarray(rp),
        "stats": stats,
        "human_mean": np.asarray(human_mean),
        "scalar": np.asarray(scalar),
    }


def _pooled_pearson(stats, idx=None):
    if idx is None:
        chosen = stats
    else:
        chosen = [stats[i] for i in idx]
    n = sy = sy2 = ss = ss2 = ssy = 0.0
    for k, a, b, c, d, e in chosen:
        n += k
        sy += a
        sy2 += b
        ss += c
        ss2 += d
        ssy += e
    return pearson_from_sums(n, sy, sy2, ss, ss2, ssy)


def evaluate_pack(pack, x, ratings, indices, bootstrap=1000, boot_seed=1):
    prob, mu = predict_distributions(pack, x, pack["objective"])
    rec = _item_records(prob, mu, ratings, indices, pack["values"])
    point = {
        "ce": float(np.mean(rec["ce"])),
        "brier": float(np.mean(rec["brier"])),
        "rps": float(np.mean(rec["rps"])),
        "rps_norm": float(np.mean(rec["rps"]) / max(len(pack["values"]) - 1, 1)),
        "sqrt_rps": float(np.sqrt(np.mean(rec["rps"]))),
        "pearson_r": _pooled_pearson(rec["stats"]),
        "spearman_rho": spearman_rho(rec["scalar"], rec["human_mean"]),
        "kendall_tau": kendall_tau(rec["scalar"], rec["human_mean"]),
    }
    rng = np.random.default_rng(boot_seed)
    n = len(indices)
    draws = rng.integers(0, n, size=(bootstrap, n))
    boot = {key: [] for key in ("ce", "brier", "rps", "sqrt_rps", "pearson_r", "spearman_rho", "kendall_tau")}
    for b in range(bootstrap):
        idx = draws[b]
        boot["ce"].append(float(np.mean(rec["ce"][idx])))
        boot["brier"].append(float(np.mean(rec["brier"][idx])))
        mean_rps = float(np.mean(rec["rps"][idx]))
        boot["rps"].append(mean_rps)
        boot["sqrt_rps"].append(float(np.sqrt(mean_rps)))
        boot["pearson_r"].append(_pooled_pearson(rec["stats"], idx))
        boot["spearman_rho"].append(spearman_rho(rec["scalar"][idx], rec["human_mean"][idx]))
        boot["kendall_tau"].append(kendall_tau(rec["scalar"][idx], rec["human_mean"][idx]))
    intervals = {}
    for key, samples in boot.items():
        lo, hi, n_ok = percentile_interval(np.asarray(samples))
        intervals[key] = {"lo": lo, "hi": hi, "n_finite": n_ok, "level": 0.95, "B": bootstrap,
                          "unit": "prompt/item cluster"}
    intervals["rps_norm"] = {
        "lo": intervals["rps"]["lo"] / max(len(pack["values"]) - 1, 1),
        "hi": intervals["rps"]["hi"] / max(len(pack["values"]) - 1, 1),
        "n_finite": intervals["rps"]["n_finite"],
        "level": 0.95,
        "B": bootstrap,
        "unit": "prompt/item cluster",
    }
    return {"point": point, "ci": intervals, "n_test_items": int(n),
            "mean_abs_mu": float(np.mean(np.abs(mu))),
            "mean_predictive_entropy": float(np.mean(-np.sum(np.clip(prob, 1e-12, 1) * np.log(np.clip(prob, 1e-12, 1)), axis=1)))}


def design_matrices(texts, train_idx, other_idx_list):
    features = hash_features(texts)
    return _svd_project(features[train_idx], [features[idx] for idx in other_idx_list])


def constant_marginal(train_ratings, indices, k_classes, values):
    counts = np.zeros(k_classes)
    for i in indices:
        counts += np.bincount(train_ratings[i], minlength=k_classes)
    prob = counts / counts.sum()
    mu = float(prob @ np.asarray(values))
    return prob, mu


def evaluate_constant(prob, mu, ratings, indices, values, bootstrap=1000, boot_seed=1):
    pack = {"objective": "constant", "values": np.asarray(values, dtype=float), "prob": prob, "mu": mu}
    # Reuse item records by faking a design-free prediction.
    probs = np.repeat(prob.reshape(1, -1), len(indices), axis=0)
    mus = np.full(len(indices), mu)
    rec = _item_records(probs, mus, ratings, indices, pack["values"])
    point = {
        "ce": float(np.mean(rec["ce"])),
        "brier": float(np.mean(rec["brier"])),
        "rps": float(np.mean(rec["rps"])),
        "rps_norm": float(np.mean(rec["rps"]) / max(len(values) - 1, 1)),
        "sqrt_rps": float(np.sqrt(max(np.mean(rec["rps"]), 0.0))),
        "pearson_r": _pooled_pearson(rec["stats"]),
        "spearman_rho": spearman_rho(rec["scalar"], rec["human_mean"]),
        "kendall_tau": kendall_tau(rec["scalar"], rec["human_mean"]),
    }
    rng = np.random.default_rng(boot_seed)
    n = len(indices)
    draws = rng.integers(0, n, size=(bootstrap, n))
    boot_ce = np.mean(rec["ce"][draws], axis=1)
    lo, hi, n_ok = percentile_interval(boot_ce)
    return {"point": point, "ci_ce": {"lo": lo, "hi": hi, "n_finite": n_ok}}


def rollout_from_probs(prob, m: int, rng) -> np.ndarray:
    """M multinomial draws from each row, averaged. A Monte Carlo readout of the same q."""
    k = prob.shape[1]
    # Sample class indices.
    cdf = np.cumsum(prob, axis=1)
    u = rng.random((m, prob.shape[0]))
    # For each draw, class = first cdf bin exceeding u.
    draws = (u[..., None] > cdf[None, :, :]).sum(axis=2)
    draws = np.clip(draws, 0, k - 1)
    out = np.zeros_like(prob)
    for j in range(k):
        out[:, j] = np.mean(draws == j, axis=0)
    # Keep a floor so a missed class in a small M does not make CE infinite in a way we cannot plot,
    # but do NOT floor here: the comparison is allowed to show infinite or huge CE at M=1.
    return out
