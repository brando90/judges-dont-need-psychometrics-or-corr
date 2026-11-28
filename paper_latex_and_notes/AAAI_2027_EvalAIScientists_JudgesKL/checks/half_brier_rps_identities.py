#!/usr/bin/env python3
"""Exact-fraction and randomized checks of Brier/RPS identities (27 Nov 2026).

Claims
------
(A) Half-Brier on a point mass: (1/2) ||δ_a - e_y||_2^2 = 1[a != y].
(B) Normalized RPS on a point mass: RPS(δ_a, y) / (K-1) wait -- if RPS is
    already divided by (K-1), RPS(δ_a, y) = |a-y|/(K-1).
    Unnormalized sum_{j=0}^{K-2} (F_j(δ_a) - 1[y<=j])^2 equals |a-y|.
(C) One-rating unbiasedness: for any score S(q, y),
    E_{Y~p}[S(q, Y)] = sum_y p(y) S(q, y), and the average of independent
    item-level scores is unbiased for the mean item-level risk (linearity
    of expectation). Checked by comparing the exact finite-sum expectation
    to the definition, and by a Monte Carlo mean matching that exact value.
(D) Permutation-average RPS vs Brier: the claim
    (1/K!) sum_{π in S_K} RPS_π(q, y) = (K+1)/(6(K-1)) ||q - e_y||_2^2
    is tested under both the paper's normalized RPS (divide by K-1) and the
    unnormalized sum, with Brier := ||q - e_y||_2^2.

All probability and score arithmetic uses fractions.Fraction.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from math import factorial
import random
import sys


def F(*args) -> Fraction:
    if len(args) == 1 and isinstance(args[0], Fraction):
        return args[0]
    return Fraction(*args)


def one_hot(K: int, a: int) -> list[Fraction]:
    return [F(1) if k == a else F(0) for k in range(K)]


def brier(q: list[Fraction], y: int) -> Fraction:
    """Unnormalized Brier ||q - e_y||_2^2 (paper L_Brier)."""
    s = F(0)
    for k, qk in enumerate(q):
        s += (qk - (F(1) if k == y else F(0))) ** 2
    return s


def half_brier(q: list[Fraction], y: int) -> Fraction:
    return brier(q, y) / 2


def rps_unnorm(q: list[Fraction], y: int) -> Fraction:
    cdf = F(0)
    s = F(0)
    K = len(q)
    for j in range(K - 1):
        cdf += q[j]
        target = F(1) if y <= j else F(0)
        s += (cdf - target) ** 2
    return s


def rps_norm(q: list[Fraction], y: int) -> Fraction:
    return rps_unnorm(q, y) / (len(q) - 1)


def apply_perm(q: list[Fraction], y: int, pi: tuple[int, ...]) -> tuple[list[Fraction], int]:
    """pi[i] is the original category placed at position i on the new scale."""
    qn = [q[pi[i]] for i in range(len(q))]
    yn = pi.index(y)
    return qn, yn


def avg_rps(q: list[Fraction], y: int, normalized: bool) -> Fraction:
    total = F(0)
    n = 0
    for pi in permutations(range(len(q))):
        qn, yn = apply_perm(q, y, pi)
        total += rps_norm(qn, yn) if normalized else rps_unnorm(qn, yn)
        n += 1
    return total / n


def random_rational_simplex(K: int, rng: random.Random) -> list[Fraction]:
    """Positive rational weights, then normalize (exact)."""
    w = [F(rng.randint(1, 12)) for _ in range(K)]
    z = sum(w)
    return [wi / z for wi in w]


def expect_score(p: list[Fraction], q: list[Fraction], score) -> Fraction:
    return sum((p[y] * score(q, y) for y in range(len(p))), F(0))


def check_A(failures: list[str]) -> int:
    n = 0
    for K in range(2, 9):
        for a in range(K):
            for y in range(K):
                n += 1
                got = half_brier(one_hot(K, a), y)
                want = F(0) if a == y else F(1)
                if got != want:
                    failures.append(f"A fail K={K} a={a} y={y}: {got} != {want}")
    return n


def check_B(failures: list[str]) -> int:
    n = 0
    for K in range(2, 9):
        for a in range(K):
            for y in range(K):
                n += 1
                un = rps_unnorm(one_hot(K, a), y)
                nm = rps_norm(one_hot(K, a), y)
                want_un = F(abs(a - y))
                want_nm = Fraction(abs(a - y), K - 1)
                if un != want_un:
                    failures.append(f"B unnorm fail K={K} a={a} y={y}: {un} != {want_un}")
                if nm != want_nm:
                    failures.append(f"B norm fail K={K} a={a} y={y}: {nm} != {want_nm}")
    return n


def check_C(failures: list[str], rng: random.Random) -> int:
    """Unbiasedness is the finite-sum definition; check it holds identically
    for Brier and RPS, and that a Monte Carlo average matches the exact
    expectation for a one-item, one-rating estimator."""
    n = 0
    for K in (2, 3, 5, 6):
        p = random_rational_simplex(K, rng)
        q = random_rational_simplex(K, rng)
        for score, name in ((brier, "brier"), (rps_norm, "rps_norm")):
            n += 1
            exact = expect_score(p, q, score)
            # Recompute from the definition; must match exactly.
            again = sum((p[y] * score(q, y) for y in range(K)), F(0))
            if exact != again:
                failures.append(f"C definition fail {name} K={K}")
            # Monte Carlo: N i.i.d. ratings from p; mean should be close.
            # Convert p to a cumulative list for sampling with exact weights.
            weights = [int(py.numerator * (F(p[0].denominator).numerator))
                       if False else py for py in p]
            # Use integer weights by a common denominator.
            d = 1
            for py in p:
                d = d * py.denominator // _gcd(d, py.denominator)
            int_w = [int(py * d) for py in p]
            assert sum(int_w) == d
            N = 20000
            acc = F(0)
            for _ in range(N):
                u = rng.randrange(d)
                c = 0
                y = K - 1
                for k, w in enumerate(int_w):
                    c += w
                    if u < c:
                        y = k
                        break
                acc += score(q, y)
            mc = acc / N
            # Tolerance: not a proof; flag only if far (Hoeffding-scale).
            err = abs(float(mc - exact))
            bound = 1.0  # scores are in a small range; 20k samples << 1 if broken
            if err > 0.05:
                failures.append(
                    f"C MC fail {name} K={K}: |mc-exact|={err:.4f} exact={exact} mc={mc}"
                )
    return n


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def check_D(failures: list[str], rng: random.Random) -> dict:
    """Test the claimed identity under both RPS normalizations.

    Claimed: avg_π RPS_π(q,y) = (K+1)/(6(K-1)) * Brier(q,y)
    Also record the unnormalized analogue (K+1)/6 * Brier, and any
    constant that actually fits.
    """
    report = {
        "norm_claim_holds": True,
        "unnorm_times_brier_is_(K+1)/6": True,
        "examples": [],
        "n_checked": 0,
        "first_norm_counterexample": None,
        "first_unnorm_counterexample": None,
    }
    cases: list[tuple[int, list[Fraction], int, str]] = []
    for K in range(2, 7):  # 6! = 720, feasible
        # all one-hot pairs
        for a in range(K):
            for y in range(K):
                cases.append((K, one_hot(K, a), y, f"onehot K={K} a={a} y={y}"))
        # a few exact structured forecasts
        if K >= 3:
            q = [F(0)] * K
            q[0] = F(1, 2)
            q[K - 1] = F(1, 2)
            cases.append((K, q, 0, f"half-ends K={K} y=0"))
            cases.append((K, q, K // 2, f"half-ends K={K} y=mid"))
        # randomized rational forecasts
        for i in range(4):
            q = random_rational_simplex(K, rng)
            y = rng.randrange(K)
            cases.append((K, q, y, f"rand K={K} i={i} y={y}"))

    for K, q, y, tag in cases:
        report["n_checked"] += 1
        br = brier(q, y)
        avg_n = avg_rps(q, y, normalized=True)
        avg_u = avg_rps(q, y, normalized=False)
        claim_n = (F(K + 1) / (F(6) * F(K - 1))) * br
        claim_u = (F(K + 1) / F(6)) * br  # if unnorm RPS = (K-1) * norm RPS
        # Also, unnorm average should be (K-1) * norm average identically
        if avg_u != avg_n * (K - 1):
            failures.append(f"D internal avg_u != (K-1) avg_n at {tag}")
        ok_n = avg_n == claim_n
        ok_u = avg_u == claim_u
        if not ok_n:
            report["norm_claim_holds"] = False
            if report["first_norm_counterexample"] is None:
                report["first_norm_counterexample"] = {
                    "tag": tag,
                    "avg_norm_RPS": str(avg_n),
                    "claim": str(claim_n),
                    "brier": str(br),
                    "ratio_avg_over_brier": (str(avg_n / br) if br != 0 else "undef"),
                    "claimed_factor": str(F(K + 1) / (F(6) * F(K - 1))),
                }
            failures.append(
                f"D NORM claim fail {tag}: avg={avg_n} claim={claim_n} brier={br}"
            )
        if not ok_u:
            report["unnorm_times_brier_is_(K+1)/6"] = False
            if report["first_unnorm_counterexample"] is None:
                report["first_unnorm_counterexample"] = {
                    "tag": tag,
                    "avg_unnorm_RPS": str(avg_u),
                    "claim": str(claim_u),
                    "brier": str(br),
                    "ratio_avg_over_brier": (str(avg_u / br) if br != 0 else "undef"),
                    "claimed_factor": str(F(K + 1) / F(6)),
                }
        # Record a few exact ratios for the writeup
        if tag.startswith("onehot") or tag.startswith("half-ends") or tag.startswith("rand K=3 i=0"):
            report["examples"].append({
                "tag": tag,
                "brier": str(br),
                "avg_norm_RPS": str(avg_n),
                "avg_unnorm_RPS": str(avg_u),
                "norm_factor_claimed": str(F(K + 1) / (F(6) * F(K - 1))),
                "norm_ratio_if_brier_pos": (str(avg_n / br) if br != 0 else "n/a"),
                "norm_match": ok_n,
                "unnorm_match": ok_u,
            })
    return report


def main() -> int:
    rng = random.Random(20261127)
    failures: list[str] = []
    nA = check_A(failures)
    nB = check_B(failures)
    # strip C's Monte Carlo failures vs exact-definition: run C
    nC = check_C(failures, rng)
    reportD = check_D(failures, rng)

    print("=== (A) half-Brier on point masses ===")
    print(f"checked {nA} pairs (K=2..8, all a,y). "
          f"{'PASS' if not any(f.startswith('A ') for f in failures) else 'FAIL'}")
    print("=== (B) RPS on point masses ===")
    print(f"checked {nB} pairs (K=2..8). unnorm = |a-y|, norm = |a-y|/(K-1). "
          f"{'PASS' if not any(f.startswith('B ') for f in failures) else 'FAIL'}")
    print("=== (C) one-rating unbiasedness (definition + MC sanity) ===")
    print(f"checked {nC} (Brier and normalized RPS on K=2,3,5,6). "
          f"{'PASS' if not any(f.startswith('C ') for f in failures) else 'FAIL'}")
    print("=== (D) average RPS over K! orderings vs c * Brier ===")
    print(f"checked {reportD['n_checked']} forecasts, K=2..6, exact fractions.")
    print("normalized-RPS claim  avg = (K+1)/(6(K-1)) * Brier :",
          "HOLDS" if reportD["norm_claim_holds"] else "REFUTED")
    print("unnormalized analogue avg = (K+1)/6 * Brier         :",
          "HOLDS" if reportD["unnorm_times_brier_is_(K+1)/6"] else "REFUTED")
    if reportD["first_norm_counterexample"]:
        print("first normalized counterexample:", reportD["first_norm_counterexample"])
    if reportD["first_unnorm_counterexample"]:
        print("first unnormalized counterexample:", reportD["first_unnorm_counterexample"])
    print("example ratios:")
    for ex in reportD["examples"][:16]:
        print(" ", ex)

    # If the exact claim failed, search for a simple closed form that does fit
    # all one-hot and random cases: is avg/brier constant in q,y for fixed K?
    print("=== (D) per-K ratios avg_norm_RPS / Brier (when Brier>0) ===")
    rng2 = random.Random(1)
    for K in range(2, 7):
        ratios = set()
        for a in range(K):
            for y in range(K):
                if a == y:
                    continue
                br = brier(one_hot(K, a), y)
                ratios.add(avg_rps(one_hot(K, a), y, True) / br)
        for i in range(3):
            q = random_rational_simplex(K, rng2)
            y = i % K
            br = brier(q, y)
            if br != 0:
                ratios.add(avg_rps(q, y, True) / br)
        print(f"  K={K}: unique ratios = {sorted(ratios, key=lambda r: (r.denominator, r.numerator))}")
        print(f"       claimed (K+1)/(6(K-1)) = {F(K+1)/(F(6)*F(K-1))}")

    if failures:
        # Print D failures compactly (can be many if claim is false)
        d_fails = [f for f in failures if f.startswith("D ")]
        other = [f for f in failures if not f.startswith("D ")]
        print(f"\nFAILURES: {len(other)} non-D, {len(d_fails)} D-claim mismatches")
        for f in other + d_fails[:8]:
            print(" -", f)
        # D mismatches are informational if we report the verdict; only
        # non-D failures are hard errors for A/B/C.
        if other:
            return 1
        return 0
    print("\nAll A/B/C checks passed; D verdict printed above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
