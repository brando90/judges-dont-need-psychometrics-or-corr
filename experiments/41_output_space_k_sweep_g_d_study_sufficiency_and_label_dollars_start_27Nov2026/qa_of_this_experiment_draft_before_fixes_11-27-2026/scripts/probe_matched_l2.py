"""Is the matched gap (Pearson-trained minus CE-trained, after the validation-fitted 1-D logit map) still near zero when L2 = 1e-4?
POPQUORN politeness K=5 and K=3, split seed 0, maxiter 300. Read-only: COPY of the code."""
import sys
sys.path.insert(0, "/tmp/opus_review_exp40/copy/experiments/40_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/scripts")
import numpy as np
from e40_common import d, fe, calibrate, three_level
base = d.load_popquorn("politeness")
for name, data in (("pol_K5", base), ("pol_K3", three_level(base, "x")), ("pol_K2", d.binarize(base, 4, "x"))):
    k = data.n_classes; values = np.asarray(data.class_values, float)
    train, val, test = d.split_by_group(data.groups, seed=0)
    xtr, (xva, xte), _ = fe.design_matrices(data.texts, train, [val, test])
    for l2, it in ((1e-2, 35), (1e-4, 300)):
        fe._penalty = (lambda w, dd, kk, l2=l2: 0.5 * l2 * float(np.sum(w.reshape(dd, kk)[:-1] ** 2)))
        fe.MAXITER = it
        res = {}
        for obj in ("ce", "pearson"):
            pack = fe.fit_objective(xtr, data.ratings, train, data.class_values, obj, None, 0)
            pv, mv = fe.predict_distributions(pack, xva, obj); pt, mt = fe.predict_distributions(pack, xte, obj)
            raw = fe._item_records(pt, mt, data.ratings, test, values)["rps"] / (k - 1)
            qc = calibrate(mv, mt, data.ratings, val, k)
            cal = fe._item_records(qc, qc @ values, data.ratings, test, values)["rps"] / (k - 1)
            res[obj] = (raw.mean(), cal.mean(), pack["nit"], pack["success"])
        print(f"{name} L2 {l2:g}: CE raw {res['ce'][0]:.4f} cal {res['ce'][1]:.4f} (nit {res['ce'][2]}) | Pearson raw {res['pearson'][0]:.4f} cal {res['pearson'][1]:.4f} (nit {res['pearson'][2]}, success {res['pearson'][3]}) | raw gap {res['pearson'][0]-res['ce'][0]:+.4f} matched gap {res['pearson'][1]-res['ce'][1]:+.4f}", flush=True)
