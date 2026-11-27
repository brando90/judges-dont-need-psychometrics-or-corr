"""QA: how much does the exp36 CE head beat the constant train-marginal forecast? (paired, item-clustered)"""
import os, sys
os.environ["SANMI36_DATA"] = "/tmp/qa55/data"
sys.path.insert(0, "/tmp/qa55/code/exp36"); sys.path.insert(0, "/tmp/qa55/code")
import numpy as np
import fit_eval
from fairness import load, item_losses
from run_all import prepare
rng0 = 1
for name in ("summeval_expert", "chaosnli_snli_mnli", "dices990", "summeval_expert_binary", "dices990_binary_yes"):
    d = load(name); prep = prepare(d); v = d.class_values; K = d.n_classes
    pack = fit_eval.fit_objective(prep["x_train"], d.ratings, prep["train"], v, "ce", None, 0)
    P = fit_eval.softmax(prep["x_test"] @ pack["weight"])
    prob, _ = fit_eval.constant_marginal(d.ratings, prep["train"], K, v)
    M = np.repeat(prob[None, :], len(prep["test"]), 0)
    ce_h, rp_h, _ = item_losses(P, d.ratings, prep["test"], v)
    ce_m, rp_m, _ = item_losses(M, d.ratings, prep["test"], v)
    rng = np.random.default_rng(1); n = len(ce_h)
    bs_ce, bs_rp = [], []
    for _ in range(1000):
        i = rng.integers(0, n, n); bs_ce.append(ce_m[i].mean() - ce_h[i].mean()); bs_rp.append(rp_m[i].mean() - rp_h[i].mean())
    q = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
    print(f"{name:24s} test CE head {ce_h.mean():.4f} marginal {ce_m.mean():.4f} | marginal minus head: CE {ce_m.mean()-ce_h.mean():+.4f} [{q(bs_ce)[0]:+.4f},{q(bs_ce)[1]:+.4f}]  normRPS {rp_m.mean()-rp_h.mean():+.4f} [{q(bs_rp)[0]:+.4f},{q(bs_rp)[1]:+.4f}]")
