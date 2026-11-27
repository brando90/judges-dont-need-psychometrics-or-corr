"""QA: both exp36 heads followed by the same post-hoc calibration (temperature + class offsets on validation CE)."""
import os, sys
os.environ["SANMI36_DATA"] = "/tmp/qa55/data"
sys.path.insert(0, "/tmp/qa55/code/exp36"); sys.path.insert(0, "/tmp/qa55/code")
import numpy as np
from scipy.optimize import minimize
import fit_eval
from fairness import load, item_losses, targets_of, paired
from run_all import prepare
def calibrate(Zva, Tva, Zte, K):
    def f(th):
        P = fit_eval.softmax(Zva * np.exp(-th[0]) + th[1:])
        return float(-(Tva * np.log(np.clip(P, 1e-300, 1))).sum(1).mean())
    r = minimize(f, np.zeros(K + 1), method="L-BFGS-B")
    return fit_eval.softmax(Zte * np.exp(-r.x[0]) + r.x[1:])
for name in ("summeval_expert", "chaosnli_snli_mnli", "dices990", "summeval_expert_binary", "dices990_binary_yes"):
    d = load(name); prep = prepare(d); v = d.class_values; K = d.n_classes
    Tva = targets_of(d.ratings, prep["val"], K)
    P = {}
    for o in ("ce", "pearson"):
        W = fit_eval.fit_objective(prep["x_train"], d.ratings, prep["train"], v, o, None, 0)["weight"]
        P[o] = calibrate(prep["x_val"] @ W, Tva, prep["x_test"] @ W, K)
    a = item_losses(P["ce"], d.ratings, prep["test"], v); b = item_losses(P["pearson"], d.ratings, prep["test"], v)
    pr = paired(a[1], b[1], a[2], b[2]); pc = paired(a[0], b[0], a[2], b[2])
    print(f"{name:24s} calibrated CE-head normRPS {a[1].mean():.4f} vs calibrated Pearson-head {b[1].mean():.4f}: diff(P-CE) {pr['loss_b_minus_a']:+.4f} [{pr['ci_loss'][0]:+.4f},{pr['ci_loss'][1]:+.4f}] | CE {a[0].mean():.4f} vs {b[0].mean():.4f}: diff {pc['loss_b_minus_a']:+.4f} [{pc['ci_loss'][0]:+.4f},{pc['ci_loss'][1]:+.4f}] | r_CE - r_P {pr['r_a_minus_r_b']:+.4f} [{pr['ci_r'][0]:+.4f},{pr['ci_r'][1]:+.4f}]")
