#!/usr/bin/env python3
"""Numerical check of the four facts in ../lean/UncertaintyTerm.lean. Exact fractions plus a seeded Monte Carlo.
Deterministic: python3 check_uncertainty_term.py (identical output twice). Exit code 1 on any failure."""
from fractions import Fraction as F
import numpy as np

fails = 0
def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name); fails += (not ok)

rng = np.random.default_rng(20261127)
bad = 0
for K in (2, 3, 5, 6):
    for _ in range(20):
        w = rng.integers(1, 20, size=K); p = [F(int(x), int(w.sum())) for x in w]
        S = sum(x * x for x in p)
        agree = sum(p[i] * p[j] * (1 if i == j else 0) for i in range(K) for j in range(K))
        brier = sum(p[y] * F(1, 2) * sum((p[j] - (1 if j == y else 0)) ** 2 for j in range(K)) for y in range(K))
        cum = [sum(p[:j + 1]) for j in range(K - 1)]
        rps = sum(p[y] * sum((cum[j] - (1 if y <= j else 0)) ** 2 for j in range(K - 1)) for y in range(K))
        bad += not (agree == S and brier == F(1, 2) * (1 - S) and rps == sum(c * (1 - c) for c in cum))
check("1-3: pair agreement = sum p^2; Brier term = (1 - sum p^2)/2; unnormalized RPS term = sum_j P_j(1-P_j) (80 exact rational laws, K = 2,3,5,6)", bad == 0)
# Monte Carlo: two ratings per item estimate the mean Brier uncertainty term without bias.
p_items = rng.dirichlet(np.ones(4), size=500)
truth = float((0.5 * (1 - (p_items ** 2).sum(1))).mean())
cdf = np.cumsum(p_items, axis=1)
est = []
for _ in range(400):
    u1, u2 = rng.random(500), rng.random(500)
    y1 = (u1[:, None] > cdf).sum(1); y2 = (u2[:, None] > cdf).sum(1)
    est.append(float((0.5 * (y1 != y2)).mean()))
se = float(np.std(est) / np.sqrt(len(est)))
check(f"pair estimator unbiased for the mean Brier uncertainty term: mean {np.mean(est):.4f} vs truth {truth:.4f} (|diff| < 3 MC s.e. = {3*se:.4f})", abs(np.mean(est) - truth) < 3 * se)
unc = lambda a: F(1, 2) * (1 - (a * a + (1 - a) ** 2))
check("4: populations A and B share the single-rating law (1/2, 1/2); uncertainty terms 1/4 vs 0",
      F(1, 2) == F(1, 2) * 1 + F(1, 2) * 0 and unc(F(1, 2)) == F(1, 4) and F(1, 2) * unc(F(1)) + F(1, 2) * unc(F(0)) == 0)
check("claimed-false: one rating per item identifies the term (fails, as expected)", not (unc(F(1, 2)) == F(1, 2) * unc(F(1)) + F(1, 2) * unc(F(0))))
print("ALL PASS" if fails == 0 else f"{fails} FAILURES"); raise SystemExit(1 if fails else 0)
