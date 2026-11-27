"""Probe: is the Pearson-trained head at a maximum of its objective when L-BFGS-B stops after 2-24 iterations?
Re-implements nothing: calls fe._fit_softmax's loss through a copy, with default and tight tolerances."""
import sys, json
from pathlib import Path
COPY = Path("/tmp/opus_review_exp40/copy/experiments/40_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/scripts")
sys.path.insert(0, str(COPY))
import numpy as np
import scipy.optimize as so
from e40_common import d, fe, calibrate, three_level
from metrics import softmax

def view(name):
    if name == "pol_K2":
        return d.binarize(d.load_popquorn("politeness"), 4, "x"), 2
    if name == "off_K2":
        return d.binarize(d.load_popquorn("offensiveness"), 4, "x"), 2
    if name == "se_exp_K2":
        return d.binarize(d.load_summeval("expert"), 4, "x"), 2
    if name == "pol_K5":
        return d.load_popquorn("politeness"), 5
    if name == "dices_K2":
        return d.binarize(d._load_dices("990"), 2, "x"), 2
    raise ValueError(name)

def nrps(prob, data, idx, k, values):
    rec = fe._item_records(prob, prob @ values, data.ratings, idx, values)
    return float(rec["rps"].mean() / (k - 1))

orig_min = so.minimize
captured = {}
def spy(fun, x0, **kw):
    r = orig_min(fun, x0, **kw)
    captured["default"] = (r, fun)
    return r

for name in sys.argv[1:]:
    data, k = view(name)
    values = np.asarray(data.class_values, float)
    train, val, test = d.split_by_group(data.groups, seed=0)
    xtr, (xva, xte), rank = fe.design_matrices(data.texts, train, [val, test])
    fe.minimize = spy
    pack = fe.fit_objective(xtr, data.ratings, train, data.class_values, "pearson", None, 0)
    fe.minimize = orig_min
    r, loss = captured["default"]
    rt = orig_min(loss, np.zeros_like(r.x), method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-15, "gtol": 1e-12})
    rows = {}
    for tag, rr in (("default", r), ("tight", rt)):
        W = rr.x.reshape(xtr.shape[1], k)
        pt, pv = softmax(xte @ W), softmax(xva @ W)
        qc = calibrate(pv @ values, pt @ values, data.ratings, val, k)
        rows[tag] = {"nit": int(rr.nit), "message": str(rr.message), "objective(-corr+pen)": float(rr.fun),
                     "test_nrps_raw": nrps(pt, data, test, k, values), "test_nrps_matched": nrps(qc, data, test, k, values)}
    print(name, json.dumps(rows, indent=1), flush=True)
