#!/usr/bin/env python3
"""Deterministic numerical checks paired with AllocationAndClosedForms.lean (Experiment 41).

Each check prints PASS or FAIL and names the Lean theorem it mirrors. Checks marked NUMERICS ONLY have no
Lean counterpart (the variance formula of a clustered mean, the link to the real-data G-study, the
cross-check against Experiment 39's own code). Fixed seed; exact fractions wherever the quantity is
rational; a named tolerance otherwise. Exit code 1 if any check fails. Run: python3 -I <this file>.
"""
import importlib
import json
import math
import sys
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "gd_study_and_allocation.json"
EXP39_SRC = HERE.parent.parent / "39_proper_score_vs_d_study_budget_allocation_start_26Nov2026" / "src"
SEED = 20261127
REL_TOL = 1e-9
SIM_TOL = 0.02  # relative tolerance for the simulated variance (about 4 standard errors at 100000 replications)
FAILED = []


def check(tag, ok, detail=""):
    print(("PASS" if ok else "FAIL"), tag, detail, flush=True)
    if not ok:
        FAILED.append(tag)


# ------------------------------------------------------------------ Part 1 (exact fractions)
C1 = dict(u=Fr(1), a=Fr(3, 5), b=Fr(1, 10), ua=Fr(3, 5), ub=Fr(1, 10), ab=Fr(1, 10), e=Fr(1, 2))
C2 = dict(u=Fr(1), a=Fr(1, 10), b=Fr(3, 5), ua=Fr(1, 10), ub=Fr(3, 5), ab=Fr(1, 10), e=Fr(1, 2))


def mix(t):
    return {k: (1 - t) * C1[k] + t * C2[k] for k in C1}


def v_abs(v, na, nb):
    """Absolute-error variance of the mean over na x nb levels, written independently of Experiment 39's code."""
    return (v["ua"] + v["a"]) / na + (v["ub"] + v["b"]) / nb + (v["e"] + v["ab"]) / (na * nb)


GRID = sorted({Fr(i, 60) for i in range(61)} | {Fr(4, 15), Fr(11, 15), Fr(1, 2)})

ok = all(v_abs(mix(t), 8, 2) == Fr(23, 80) + Fr(3, 8) * t for t in GRID)
ok &= all(v_abs(mix(t), 4, 4) == Fr(31, 80) for t in GRID)
ok &= all(v_abs(mix(t), 2, 8) == Fr(53, 80) - Fr(3, 8) * t for t in GRID)
check("[N1] closed forms 23/80+3t/8, 31/80, 53/80-3t/8 on %d rational t (Lean: vDelta_8x2, vDelta_4x4, vDelta_2x8)" % len(GRID), ok)

lo = [t for t in GRID if v_abs(mix(t), 8, 2) == v_abs(mix(t), 4, 4)]
hi = [t for t in GRID if v_abs(mix(t), 2, 8) == v_abs(mix(t), 4, 4)]
check("[N2] crossovers are exactly t = 4/15 and t = 11/15 (Lean: crossover_low, crossover_high)", lo == [Fr(4, 15)] and hi == [Fr(11, 15)], f"found {lo} and {hi}")

mid = [t for t in GRID if Fr(4, 15) < t < Fr(11, 15)]
low = [t for t in GRID if t < Fr(4, 15)]
high = [t for t in GRID if t > Fr(11, 15)]
ok = all(v_abs(mix(t), 4, 4) < v_abs(mix(t), 8, 2) and v_abs(mix(t), 4, 4) < v_abs(mix(t), 2, 8) for t in mid)
check("[N3] 4x4 is the unique optimum strictly between the crossovers (Lean: unique_optimum_middle)", ok and len(mid) > 0, f"{len(mid)} grid points")
ok = all(v_abs(mix(t), 8, 2) < v_abs(mix(t), 4, 4) and v_abs(mix(t), 8, 2) < v_abs(mix(t), 2, 8) for t in low)
ok &= all(v_abs(mix(t), 2, 8) < v_abs(mix(t), 4, 4) and v_abs(mix(t), 2, 8) < v_abs(mix(t), 8, 2) for t in high)
check("[N4] 8x2 is the unique optimum below 4/15 and 2x8 above 11/15 (Lean: optimum_low, optimum_high)", ok and len(low) > 0 and len(high) > 0)

h = Fr(1, 2)
vals = (v_abs(mix(h), 8, 2), v_abs(mix(h), 2, 8), v_abs(mix(h), 4, 4))
check("[N5] at t = 1/2 the three designs are NOT tied: 19/40, 19/40, 31/80 (Lean: not_a_three_way_tie_at_half)",
      vals == (Fr(19, 40), Fr(19, 40), Fr(31, 80)) and vals[2] < vals[0], str(vals))

# cross-check against Experiment 39's own implementation (NUMERICS ONLY)
try:
    sys.path.insert(0, str(EXP39_SRC))
    L = importlib.import_module("gtheory_lib")
    ok = all(L.d_study(mix(t), na, nb)["v_Delta"] == v_abs(mix(t), na, nb) for t in GRID for (na, nb) in [(8, 2), (4, 4), (2, 8)])
    check("[N6] NUMERICS ONLY: Experiment 39's d_study returns the same v_Delta as the independent formula", ok)
except Exception as exc:  # the other experiment's folder or its dependencies may be absent
    print("SKIP [N6] cross-check against Experiment 39's gtheory_lib:", type(exc).__name__, str(exc)[:120], flush=True)

# ------------------------------------------------------------------ Part 2 (exact fractions, random draws)
rng = np.random.default_rng(SEED)


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


ok_dec = ok_eq = ok_more = True
for _ in range(2000):
    B = int(2 ** rng.integers(1, 9) * 3 ** rng.integers(0, 4))
    ds = divisors(B)
    if len(ds) < 2:
        continue
    i, j = sorted(rng.choice(len(ds), size=2, replace=False))
    n1, n2 = ds[i], ds[j]  # n1 < n2 items, so k1 = B/n1 > k2 = B/n2 raters
    k1, k2 = Fr(B, n1), Fr(B, n2)
    u = Fr(int(rng.integers(1, 40)), int(rng.integers(1, 40)))
    e = Fr(int(rng.integers(1, 40)), int(rng.integers(1, 40)))
    ok_dec &= (e / k1 < e / k2)          # more raters per item: smaller per-item residual variance
    ok_eq &= (u / n1 + e / (n1 * k1) == u / n1 + e / B)
    ok_more &= (u / n2 + e / B < u / n1 + e / B)   # more items: smaller grand-mean variance
check("[N7] per-item residual e/k falls with k (Lean: per_item_decreasing)", ok_dec)
check("[N8] at fixed B = n k the grand-mean variance is u/n + e/B (Lean: grand_mean_var_eq)", ok_eq)
check("[N9] at fixed B, more items strictly lowers the grand-mean variance when u > 0 (Lean: grand_mean_more_items)", ok_more)

# ------------------------------------------------------------------ Part 3 (floats, named tolerance)
def f(a, b, c, d, k):
    return (a + b / k) * (c + d * k)


ok_id = ok_ge = ok_uniq = ok_int = True
n_int = 0
for _ in range(2000):
    a, b, c, d = np.exp(rng.normal(0, 1.2, 4))
    k0 = math.sqrt(b * c / (a * d))
    if not (0.05 < k0 < 300):
        continue
    for k in np.exp(rng.normal(math.log(k0), 1.0, 6)):
        lhs = f(a, b, c, d, k) - f(a, b, c, d, k0)
        rhs = a * d * (k - k0) ** 2 / k
        ok_id &= abs(lhs - rhs) <= REL_TOL * max(1.0, abs(f(a, b, c, d, k)))
        ok_ge &= lhs >= -REL_TOL * f(a, b, c, d, k)
    for k in (k0 * 0.97, k0 * 1.03):
        ok_uniq &= f(a, b, c, d, k) > f(a, b, c, d, k0)
    if k0 >= 1:  # integer minimizer over k = 1..400 is the floor or the ceiling of k0
        vals_int = [f(a, b, c, d, k) for k in range(1, 401)]
        best = int(np.argmin(vals_int)) + 1
        ok_int &= best in (max(1, math.floor(k0)), max(1, math.ceil(k0)))
        n_int += 1
check("[N10] f(k) - f(k0) = a d (k-k0)^2 / k when b c = a d k0^2 (Lean: cost_identity)", ok_id)
check("[N11] k0 minimizes f over k > 0 (Lean: cost_optimal)", ok_ge)
check("[N12] k0 is the unique minimizer: nearby k are strictly worse (Lean: cost_optimal_unique)", ok_uniq)
check("[N13] the best integer k is the floor or ceiling of k0 (consequence; NUMERICS ONLY)", ok_int, f"{n_int} draws")

ok_free = True
for _ in range(500):
    a, b, d = np.exp(rng.normal(0, 1.2, 3))
    vals_int = [f(a, b, 0.0, d, k) for k in range(1, 401)]
    ok_free &= int(np.argmin(vals_int)) == 0
check("[N14] with free items (c = 0) one rating per item is optimal (Lean: free_items_one_rating_optimal)", ok_free)

# ------------------------------------------------------------------ Part 4 (exact fractions)
ok_rel = True
for _ in range(3000):
    s = Fr(int(rng.integers(1, 51)), int(rng.integers(1, 51)))
    w = Fr(int(rng.integers(1, 51)), int(rng.integers(1, 51)))
    q = int(rng.integers(2, 40))
    r = Fr(int(rng.integers(1, q)), q)
    k = Fr(int(rng.integers(1, 61)), int(rng.integers(1, 5)))
    ok_rel &= ((r <= s / (s + w / k)) == (r / (1 - r) * (w / s) <= k))
    kb = r / (1 - r) * (w / s)  # the boundary: reliability equals r exactly
    ok_rel &= (s / (s + w / kb) == r)
check("[N15] reliability s/(s+w/k) >= r exactly when k >= r/(1-r) * w/s, with equality at the boundary (Lean: reliability_iff)", ok_rel)

# ------------------------------------------------------------------ link to the real-data G-study (NUMERICS ONLY)
payload = json.loads(RESULTS.read_text())
ok_k80 = ok_kopt = True
rows = []
for r in payload["datasets"]:
    rho = r["g_study"]["icc1"]
    k80 = r["g_study"]["k_star_080"]
    pred = math.ceil(4.0 * (1 - rho) / rho - 1e-12)  # r/(1-r) = 4 at reliability 0.80, w/s = (1-rho)/rho
    comp = r["test_loss_components_ce_head"]
    kopt_pred = max(1.0, math.sqrt(10.0 * comp["sigma_w2"] / comp["sigma_c2"]))
    ok_k80 &= (pred == k80)
    ok_kopt &= abs(kopt_pred - comp["k_opt_if_item_costs_10x_rating"]) <= 1e-6
    rows.append(f"{r['name']}: k*={k80} (formula {pred}), k_opt(10x)={comp['k_opt_if_item_costs_10x_rating']:.3f} (formula {kopt_pred:.3f})")
check("[N16] NUMERICS ONLY: the stored k* for reliability 0.80 equals ceil(4(1-rho)/rho) on all %d real datasets" % len(payload["datasets"]), ok_k80)
check("[N17] NUMERICS ONLY: the stored cost-optimal k (items cost 10x a rating) equals max(1, sqrt(10 sw2/sc2)) on all real datasets", ok_kopt)
for line in rows:
    print("      ", line)

# ------------------------------------------------------------------ variance of a clustered mean by simulation (NUMERICS ONLY)
def sim_var(N, k, sc2, sw2, reps=100000, chunks=10):
    out = []
    per = reps // chunks
    for _ in range(chunks):
        items = rng.normal(0.0, math.sqrt(sc2), size=(per, N, 1))
        noise = rng.normal(0.0, math.sqrt(sw2), size=(per, N, k))
        out.append((items + noise).mean(axis=(1, 2)))
    return float(np.var(np.concatenate(out), ddof=1))


ok_sim = True
for (N, k, sc2, sw2) in [(40, 1, 0.3, 0.7), (40, 4, 0.3, 0.7), (10, 16, 0.3, 0.7), (160, 1, 0.1, 0.9), (20, 8, 0.1, 0.9)]:
    got, want = sim_var(N, k, sc2, sw2), (sc2 + sw2 / k) / N
    good = abs(got - want) <= SIM_TOL * want
    ok_sim &= good
    print(f"       N={N} k={k} sigma_c^2={sc2} sigma_w^2={sw2}: simulated {got:.5f}, formula {want:.5f}, rel diff {abs(got - want) / want:.4f}")
check("[N18] NUMERICS ONLY: Var(mean over N items x k ratings) = (sigma_c^2 + sigma_w^2/k)/N within %.0f%% in simulation" % (100 * SIM_TOL), ok_sim)
v_many, v_deep = sim_var(160, 1, 0.1, 0.9), sim_var(40, 4, 0.1, 0.9)  # same budget 160
check("[N19] NUMERICS ONLY: at the same budget B = 160, 160 items x 1 rating has smaller variance than 40 items x 4 ratings", v_many < v_deep, f"{v_many:.5f} < {v_deep:.5f}")

print()
if FAILED:
    print("FAILED:", ", ".join(FAILED))
    sys.exit(1)
print("ALL PASS")
