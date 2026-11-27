"""Does the k=1 learning curve still saturate at 512 when the L2 penalty is 1e-4 instead of 0.01? POPQUORN politeness (K=5), split seed 0,
same subsample orders and seeds as gd_study_and_allocation.fit_eval_subset. Read-only: works on a COPY of the code."""
import sys, json
from pathlib import Path
sys.path.insert(0, "/tmp/opus_review_exp40/copy/experiments/40_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/scripts")
import numpy as np
from e40_common import d, fe
data = d.load_popquorn("politeness")
values = np.asarray(data.class_values, float); kk = data.n_classes
train, val, test = d.split_by_group(data.groups, seed=0)
def run(n, l2):
    fe._penalty = (lambda w, dd, k, l2=l2: 0.5 * l2 * float(np.sum(w.reshape(dd, k)[:-1] ** 2)))
    vecs = []
    for s in range(3):
        order = np.random.default_rng(100 + s).permutation(train)
        idx = np.sort(order[:n])
        x_tr, (x_te,), _ = fe.design_matrices(data.texts, idx, [test])
        pack = fe.fit_objective(x_tr, data.ratings, idx, values, "ce", 1, 7 + s)
        prob, mu = fe.predict_distributions(pack, x_te, "ce")
        rec = fe._item_records(prob, mu, data.ratings, test, values)
        vecs.append(rec["rps"] / (kk - 1))
    return float(np.mean(vecs, axis=0).mean())
cprob, cmu = fe.constant_marginal(data.ratings, train, kk, values)
crec = fe._item_records(np.repeat(cprob[None], len(test), 0), np.full(len(test), cmu), data.ratings, test, values)
print("constant", round(float(crec["rps"].mean() / (kk - 1)), 4))
for l2 in (1e-2, 1e-4):
    for n in (128, 512, 2230):
        fe.MAXITER = 35 if l2 == 1e-2 else 300
        print(f"L2 {l2:g} maxiter {fe.MAXITER} N {n}: test nRPS {run(n, l2):.4f}", flush=True)
