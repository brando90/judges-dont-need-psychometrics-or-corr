"""Shared by Experiments 47, 48 and 50: rebuild Experiment 41's 16 views and train its preregistered heads once per (view, split seed).

Imports Experiment 41's e40_common (which imports Experiment 36's loaders and heads unchanged). Heads: CE-, Pearson- and MSE-trained,
L2 0.01, at most 35 L-BFGS iterations (Experiment 41's preregistered run). Cached to $E47_HEAD_CACHE (default /workspace/calib_followups/cache41)
as .npz so the three experiments use byte-identical heads.
"""
import os, sys
from pathlib import Path
import numpy as np

E41S = Path(__file__).resolve().parents[2] / "41_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026" / "scripts"
sys.path.insert(0, str(E41S))
os.environ.setdefault("E40_DATA_CACHE", "/workspace/calib_followups/data_cache")
from e40_common import calibrate, d, fe, histograms, fit_logit_map, paired_ci, rebin, three_level  # noqa: E402
from metrics import softmax  # noqa: E402

CACHE = Path(os.environ.get("E47_HEAD_CACHE", "/workspace/calib_followups/cache41"))
SIGNAL = {"summeval_crowd": False, "chaosnli_alphanli": False}  # Exp 41 'carries signal' = no only for these bases (all K)
fe.MAXITER = 35


def build_views():
    pop_off = d.load_popquorn("offensiveness"); pop_pol = d.load_popquorn("politeness")
    se_exp = d.load_summeval("expert"); se_crd = d.load_summeval("crowd")
    d990 = d._load_dices("990"); chaos3 = d.load_chaos_3way(); chaosa = d.load_chaos_alpha()
    views = []
    for base in (pop_off, pop_pol, se_exp, se_crd):
        views.append((base.name + "|K2", base.name, 2, d.binarize(base, d.SUMMEVAL_BINARY_CUT, base.name + "_K2")))
        views.append((base.name + "|K3", base.name, 3, three_level(base, base.name + "_K3")))
        views.append((base.name + "|K5", base.name, 5, base))
    views.append(("dices990|K2", "dices990", 2, d.binarize(d990, 2, "dices990_K2_yes")))
    views.append(("dices990|K3", "dices990", 3, d990))
    views.append(("chaosnli_snli_mnli|K3", "chaosnli_snli_mnli", 3, chaos3))
    views.append(("chaosnli_alphanli|K2", "chaosnli_alphanli", 2, chaosa))
    return views


def heads(vname, data, seed):
    """Returns dict with split indices and, per head, raw test probs and val/test scalar mean scores; plus constant forecast."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{vname.replace('|', '_')}_seed{seed}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        return {k: z[k] for k in z.files}
    train, val, test = d.split_by_group(data.groups, seed=seed)
    d.assert_no_prompt_leakage(data.groups, data.rater_ids, train, val, test)
    x_train, (x_val, x_test), _ = fe.design_matrices(data.texts, train, [val, test])
    values = np.asarray(data.class_values, float); k = data.n_classes
    out = {"train": train, "val": val, "test": test}
    for obj in ("ce", "pearson", "mse"):
        pack = fe.fit_objective(x_train, data.ratings, train, data.class_values, obj, None, seed)
        pv, mv = fe.predict_distributions(pack, x_val, obj)
        pt, mt = fe.predict_distributions(pack, x_test, obj)
        out[f"{obj}_pt"], out[f"{obj}_mt"], out[f"{obj}_mv"], out[f"{obj}_pv"] = pt, mt, mv, pv
    cprob, _ = fe.constant_marginal(data.ratings, train, k, values)
    out["const"] = cprob
    np.savez_compressed(f, **out)
    return out


def item_nrps_ce(q, ratings, idx, k):
    """Per-item mean normalized RPS and CE over each item's ratings (the unbiased per-sample risk)."""
    H = histograms(ratings, idx, k)
    q = np.clip(q, 1e-12, 1); q = q / q.sum(1, keepdims=True)
    F, G = np.cumsum(q, 1)[:, :-1], np.cumsum(H, 1)[:, :-1]
    return (F * F - 2 * F * G + G).sum(1) / (k - 1), -(H * np.log(q)).sum(1)


def map_from(z_cal, targets, z_apply):
    """Exp 41's map (multinomial logit on [standardized score, 1], CE) fitted on (z_cal, targets); standardization uses z_cal's moments."""
    m, s = float(z_cal.mean()), float(z_cal.std()); s = s if s > 1e-12 else 1.0
    w = fit_logit_map((z_cal - m) / s, targets)
    return softmax(np.column_stack([(z_apply - m) / s, np.ones_like(z_apply)]) @ w)
