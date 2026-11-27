"""Population-level check of Experiment 42's headline gap under a matched calibration design (not part of Cursor's experiment).

Arms, all evaluated by exact expected normalized RPS on Experiment 42's polarized deployment law (lower is better):
  oracle                       the polarized law itself
  lookup, pooled (train mix)   six histograms from the pooled persona mixture (what pooled add-one cross-entropy converges to)
  mean-matched binomial        Experiment 42's "Pearson binomial"
  Pearson scalar + CE map      one scalar per bin (the pooled mean score, z-standardized) -> softmax(a + c z) fitted by population cross-entropy
                               (linear logits, as in Experiment 41's calibration map), fitted on (a) the pooled mixture or (b) the polarized law
  same with a quadratic term   logits a + c z + d z^2
Usage: python3 -B matched_calibration_population_check.py   (reads Experiment 42's benchmark.py from its folder next to this one's experiment)"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

here = Path(__file__).resolve()
E42 = next((p / "42_multimodal_simulated_population_ce_rps_vs_corr_gtheory_start_27Nov2026" for p in here.parents
            if (p / "42_multimodal_simulated_population_ce_rps_vs_corr_gtheory_start_27Nov2026").is_dir()), None)
if E42 is None:
    E42 = Path(sys.argv[1])
sys.path.insert(0, str(E42))
import benchmark as B

pop = B.build_population()
laws = pop["laws"]
pol = laws[2]
train_mix = B.mix(B.TRAIN_WEIGHTS, laws)
score = B.row_means(train_mix)                      # the Pearson-scored scalar per bin: the pooled mean score
z = (score - score.mean()) / score.std()
K = pol.shape[1]


def features(degree):
    return np.stack([z ** d for d in range(1, degree + 1)], axis=1)           # (bins, degree)


def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fit_map(target, degree):
    F = features(degree)
    nb = F.shape[0]

    def unpack(theta):
        a = np.concatenate([[0.0], theta[:K - 1]])
        C = np.vstack([np.zeros((1, degree)), theta[K - 1:].reshape(K - 1, degree)])
        return a, C

    def loss_grad(theta):
        a, C = unpack(theta)
        q = softmax(a[None, :] + F @ C.T)
        loss = -(target * np.log(q)).sum() / nb
        g = (q - target) / nb                                                   # (bins, K) gradient on logits
        ga = g.sum(axis=0)[1:]
        gC = (g.T @ F)[1:]
        return loss, np.concatenate([ga, gC.ravel()])

    res = minimize(loss_grad, np.zeros((K - 1) * (1 + degree)), jac=True, method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12})
    a, C = unpack(res.x)
    return softmax(a[None, :] + F @ C.T), res.success


def rps_on_pol(q):
    return float(B.expected_rps_rows(q, pol).mean())


rows = [("oracle (polarized law)", pol), ("lookup, pooled persona mixture", train_mix), ("mean-matched binomial (Exp 42 'Pearson binomial')", B.binom_rows(score))]
print("Normalized expected RPS on the polarized deployment law (lower is better). Population version: no sampling noise.")
for name, q in rows:
    print("  %-62s %.6f" % (name, rps_on_pol(q)))
print("Pearson of the scalar with the bin means:", round(float(np.corrcoef(score, B.BIN_MEANS)[0, 1]), 6))
for deg, label in ((1, "linear logits (Experiment 41's map form)"), (2, "quadratic logits")):
    for tname, target in (("pooled mixture", train_mix), ("polarized law", pol)):
        q, ok = fit_map(target, deg)
        r = rps_on_pol(q)
        print("  Pearson scalar + CE map, %-40s fitted on %-15s %.6f  (converged %s)" % (label, tname, r, ok))
base = rps_on_pol(pol)
q1, _ = fit_map(train_mix, 1)
q1p, _ = fit_map(pol, 1)
print("\nGaps in expected RPS on the polarized law (each minus the oracle %.6f):" % base)
print("  mean-matched binomial:                    %+.6f" % (rps_on_pol(B.binom_rows(score)) - base))
print("  lookup pooled mixture:                    %+.6f" % (rps_on_pol(train_mix) - base))
print("  scalar + linear-logit CE map, pooled fit: %+.6f" % (rps_on_pol(q1) - base))
print("  scalar + linear-logit CE map, polarized:  %+.6f" % (rps_on_pol(q1p) - base))
print("The headline gap of Experiment 42 (binomial minus persona CE at 256 ratings per item) is 0.059517 and uses finite-sample persona CE (0.222275).")
