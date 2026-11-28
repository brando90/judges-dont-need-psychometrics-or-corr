#!/usr/bin/env python3
"""Seeded exact-fraction checks of the Point 1 claims (target versus ruler), run BEFORE the Lean file.

Every claim prints PASS or FAIL; the process exits nonzero if any FAIL. Claims stated as false are tested
as failing cases: the check PASSes only if a counterexample is found. Seed 20261127, `fractions.Fraction`
only, no floating point, so two runs print identical output.

Claims (C) and what they back:
  C1  Expected Brier score under p of forecast q equals sum_k (q_k - p_k)^2 + sum_k p_k (1 - p_k).
  C2  Forecasts with the same strict mode agree with exactly the same labels.
  C3  (claimed false) Same strict mode implies the same Brier excess: a counterexample exists.
  C4  (claimed false, wording of the pasted draft) Same conditional mean implies the same agreement:
      a counterexample with equal means and different modes exists for K >= 3.
  C5  For K = 2 equal means imply equal distributions; for K = 3 a distinct equal-mean pair exists.
  C6  Pool dependence: the truthful forecast for pool A has Brier excess sum_k (pA_k - pB_k)^2 > 0 under pool B.
  C7  Moving a forecast toward the human distribution by a fraction t multiplies the excess by (1 - t)^2.
  C8  Consensus item (point mass): sharpening toward the mode strictly lowers the excess whenever q_m < 1.
  C9  (claimed false) Sharpening toward the mode always lowers the excess: a contested item contradicts it.
"""
import random
import sys
from fractions import Fraction as F

SEED = 20261127
rng = random.Random(SEED)
FAILS = []


def report(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (" -- " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def rand_dist(k, denom=20):
    """A random distribution over k categories with rational entries (denominator `denom`)."""
    cuts = sorted(rng.randint(0, denom) for _ in range(k - 1))
    parts = [b - a for a, b in zip([0] + cuts, cuts + [denom])]
    return [F(x, denom) for x in parts]


def brier(q, y):
    return sum((q[k] - (1 if k == y else 0)) ** 2 for k in range(len(q)))


def exp_brier(q, p):
    return sum(p[y] * brier(q, y) for y in range(len(p)))


def excess(q, p):
    return sum((a - b) ** 2 for a, b in zip(q, p))


def strict_mode(q):
    m = max(range(len(q)), key=lambda k: q[k])
    return m if all(q[k] < q[m] for k in range(len(q)) if k != m) else None


def mean(q):
    return sum(k * q[k] for k in range(len(q)))


def main():
    # C1
    ok = True
    for _ in range(500):
        k = rng.randint(2, 6)
        p, q = rand_dist(k), rand_dist(k)
        ok &= exp_brier(q, p) == excess(q, p) + sum(x * (1 - x) for x in p)
    report("C1 expected Brier = excess + irreducible term (500 exact cases)", ok)

    # C2 and C3
    n_same = 0
    ok2 = True
    counter_c3 = None
    while n_same < 300:
        k = rng.randint(3, 6)
        q, q2 = rand_dist(k), rand_dist(k)
        m, m2 = strict_mode(q), strict_mode(q2)
        if m is None or m != m2:
            continue
        n_same += 1
        for y in range(k):
            ok2 &= (strict_mode(q) == y) == (strict_mode(q2) == y)
        p = rand_dist(k)
        if counter_c3 is None and excess(q, p) != excess(q2, p):
            counter_c3 = (q, q2, p)
    report("C2 same strict mode => same agreement with every label (300 exact pairs)", ok2)
    report("C3 (claimed false) same strict mode => same Brier excess: counterexample found", counter_c3 is not None,
           "q=%s q'=%s p=%s" % tuple([[str(x) for x in v] for v in counter_c3]) if counter_c3 else "")

    # C4
    found = None
    for _ in range(20000):
        k = rng.randint(3, 5)
        q, q2 = rand_dist(k), rand_dist(k)
        if mean(q) == mean(q2) and strict_mode(q) is not None and strict_mode(q2) is not None \
                and strict_mode(q) != strict_mode(q2):
            found = (q, q2)
            break
    report("C4 (claimed false) same conditional mean => same agreement: counterexample with different modes found",
           found is not None, "means %s, modes %s vs %s" % (mean(found[0]), strict_mode(found[0]), strict_mode(found[1]))
           if found else "")

    # C5
    ok5 = True
    for _ in range(2000):
        a, b = rand_dist(2), rand_dist(2)
        if mean(a) == mean(b):
            ok5 &= a == b
    pair3 = None
    for _ in range(20000):
        a, b = rand_dist(3), rand_dist(3)
        if a != b and mean(a) == mean(b):
            pair3 = (a, b)
            break
    report("C5a K=2: equal means imply equal distributions (2000 exact pairs)", ok5)
    report("C5b K=3: a distinct pair with equal means exists", pair3 is not None,
           "%s vs %s, mean %s" % ([str(x) for x in pair3[0]], [str(x) for x in pair3[1]], mean(pair3[0])) if pair3 else "")

    # C6
    ok6 = True
    for _ in range(500):
        k = rng.randint(3, 6)
        pa, pb = rand_dist(k), rand_dist(k)
        if pa == pb:
            continue
        ok6 &= excess(pa, pb) > 0 and exp_brier(pb, pb) < exp_brier(pa, pb)
        ok6 &= excess(pa, pb) == exp_brier(pa, pb) - exp_brier(pb, pb)
    report("C6 truthful forecast for pool A has positive Brier excess under pool B (500 exact pairs)", ok6)

    # C7
    ok7 = True
    for _ in range(500):
        k = rng.randint(2, 6)
        p, q = rand_dist(k), rand_dist(k)
        t = F(rng.randint(0, 10), 10)
        q2 = [qi + t * (pi - qi) for qi, pi in zip(q, p)]
        ok7 &= excess(q2, p) == (1 - t) ** 2 * excess(q, p)
    report("C7 moving toward p by fraction t scales the excess by (1-t)^2 (500 exact cases)", ok7)

    # C8
    ok8 = True
    for _ in range(500):
        k = rng.randint(2, 6)
        m = rng.randrange(k)
        p = [F(1) if i == m else F(0) for i in range(k)]
        q = rand_dist(k)
        if q[m] == 1:
            continue
        t = F(rng.randint(1, 10), 10)
        q2 = [qi + t * (pi - qi) for qi, pi in zip(q, p)]
        ok8 &= excess(q2, p) < excess(q, p)
    report("C8 consensus item: sharpening toward the point-mass mode strictly lowers the excess (500 exact cases)", ok8)

    # C9
    counter = None
    for _ in range(5000):
        k = 3
        p = rand_dist(k)
        if max(p) > F(2, 5) or strict_mode(p) is None:
            continue
        m = strict_mode(p)
        t = F(1, 2)
        q = [pi + t * ((1 if i == m else 0) - pi) for i, pi in enumerate(p)]
        if excess(q, p) > excess(p, p):
            counter = (p, q)
            break
    report("C9 (claimed false) sharpening toward the mode always lowers the excess: contested-item counterexample found",
           counter is not None,
           "p=%s sharpened q=%s" % ([str(x) for x in counter[0]], [str(x) for x in counter[1]]) if counter else "")

    print("SEED %d; claims failed: %d" % (SEED, len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
