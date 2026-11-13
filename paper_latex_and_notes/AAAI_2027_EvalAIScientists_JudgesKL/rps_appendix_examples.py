"""Print every number in the ranked probability score (RPS) appendix (8-appendix-rps.tex).

Run: python3 paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/rps_appendix_examples.py
Pure arithmetic on the 0-5 rating scale; no data, models or randomness except the identity check.
"""
import numpy as np

K = 6  # ratings 0..5
GRID = np.arange(K)


def rps(q, r):
    """Normalized RPS of forecast q (length 6) against one observed rating r."""
    cdf = np.cumsum(q)[:5]
    observed = (r <= np.arange(5)).astype(float)
    return float(np.mean((cdf - observed) ** 2))


def expected_rps(q, p):
    """Expected RPS when the rating is drawn from the human distribution p."""
    return sum(p[r] * rps(q, r) for r in range(K))


def point_mass(k):
    v = np.zeros(K)
    v[k] = 1.0
    return v


def main():
    print("Single human rating r = 4:")
    for name, q in [("all mass on 4", point_mass(4)), ("all mass on 5", point_mass(5)),
                    ("all mass on 0", point_mass(0)),
                    ("half on 3, half on 5", np.array([0, 0, 0, 0.5, 0, 0.5]))]:
        brier = float(((q - point_mass(4)) ** 2).sum())
        print(f"  {name:<22} RPS={rps(q, 4):.3f}  Brier={brier:.1f}  q(4)={q[4]:.1f}")

    q = np.array([0, 0, 0.1, 0.3, 0.4, 0.2])
    cdf = np.cumsum(q)[:5]
    gaps = cdf - (4 <= np.arange(5)).astype(float)
    print("Staircase example, r = 4, q =", q.tolist())
    print("  Q_j =", np.round(cdf, 2).tolist(), " squared gaps =", np.round(gaps ** 2, 2).tolist(),
          f" RPS = {rps(q, 4):.3f}")

    p = np.array([0, 0, 0, 0.5, 0, 0.5])
    print("Raters split 50/50 between 3 and 5:")
    print(f"  forecast = human distribution:  expected RPS = {expected_rps(p, p):.2f}")
    print(f"  forecast = all mass on 4 (mean): expected RPS = {expected_rps(point_mass(4), p):.2f}")
    print(f"  forecast = uniform on 0..5:      expected RPS = {expected_rps(np.ones(K) / K, p):.3f}")
    print(f"  both first two forecasts have mean rating {float(p @ GRID):.1f} and {float(point_mass(4) @ GRID):.1f}")

    # The mean is the sum of the survival function: survival gaps can cancel in the mean, not in RPS.
    surv = lambda f: 1 - np.cumsum(f)[:5]
    s_true, s_four = surv(p), surv(point_mass(4))
    print("  survival S_j, j=0..4: human distribution", s_true.tolist(), " all mass on 4", s_four.tolist())
    print(f"  sums (= mean ratings): {s_true.sum():.1f} and {s_four.sum():.1f};"
          f" gaps {(s_four - s_true).tolist()} sum to {(s_four - s_true).sum():.1f};"
          f" squared gaps / 5 = {np.mean((s_four - s_true) ** 2):.2f} = excess expected RPS")
    worst = 0.0
    rng_mean = np.random.default_rng(1)
    for _ in range(1000):
        f = rng_mean.dirichlet(np.ones(K))
        worst = max(worst, abs(float(f @ GRID) - surv(f).sum()))
    print(f"  E[R] = sum_j S_j on 1,000 random forecasts: max absolute error {worst:.1e}")

    print("Range: RPS(all mass on 0, r=5) =", rps(point_mass(0), 5))

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        q, p = rng.dirichlet(np.ones(K)), rng.dirichlet(np.ones(K))
        r = int(rng.integers(K))
        abs_form = (q * np.abs(GRID - r)).sum() - 0.5 * (np.outer(q, q) * np.abs(GRID[:, None] - GRID[None, :])).sum()
        Q, P = np.cumsum(q)[:5], np.cumsum(p)[:5]
        decomposition = np.mean((Q - P) ** 2 + P * (1 - P))
        worst = max(worst, abs(5 * rps(q, r) - abs_form), abs(expected_rps(q, p) - decomposition))
    print(f"Identity checks on 2,000 random forecasts: max absolute error {worst:.1e}")

    # Excess expected RPS = mean squared CDF gap = energy distance / 10 on the integer scale.
    worst = 0.0
    for _ in range(2000):
        q, p = rng.dirichlet(np.ones(K)), rng.dirichlet(np.ones(K))
        excess = expected_rps(q, p) - expected_rps(p, p)
        d = np.abs(GRID[:, None] - GRID[None, :])
        energy = 2 * (np.outer(q, p) * d).sum() - (np.outer(q, q) * d).sum() - (np.outer(p, p) * d).sum()
        worst = max(worst, abs(excess - np.mean((np.cumsum(q)[:5] - np.cumsum(p)[:5]) ** 2)), abs(10 * excess - energy))
    print(f"Excess RPS = mean squared CDF gap = energy distance / 10: max absolute error {worst:.1e}")

    # Sparse ratings: plug-in distances to a few-rating histogram reward overconfidence.
    p = np.array([0, 0, 0, 0.6, 0, 0.4])

    def expected_tvd_to_histogram(q, n):
        total = 0.0
        for ratings in np.ndindex(*([K] * n)):
            prob = np.prod([p[r] for r in ratings])
            if prob:
                hist = np.bincount(ratings, minlength=K) / n
                total += prob * 0.5 * np.abs(q - hist).sum()
        return total

    def expected_w1_one_rating(q):
        return sum(p[r] * np.abs(np.cumsum(q) - np.cumsum(point_mass(r))).sum() for r in range(K))

    print("Raters choose 3 w.p. 0.6 and 5 w.p. 0.4:")
    print("  truth's expected TVD to its own 1/2/3-rating histogram:",
          [round(expected_tvd_to_histogram(p, n), 3) for n in (1, 2, 3)])
    for name, q in [("truth", p), ("all mass on 3", point_mass(3)), ("all mass on 4", point_mass(4))]:
        print(f"  {name:<14} one-rating TVD={expected_tvd_to_histogram(q, 1):.2f}"
              f"  one-rating W1={expected_w1_one_rating(q):.2f}  expected RPS={expected_rps(q, p):.3f}")

    # The squared-absolute variant of RPS is not proper: a concentrated forecast beats the truth.
    def sa_rps(q, r):
        return float(np.abs(np.cumsum(q)[:5] - (r <= np.arange(5))).sum() ** 2 / 5)

    p = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    q = np.array([0, 0, 0.9, 0.1, 0, 0])
    sa = lambda f: sum(p[r] * sa_rps(f, r) for r in range(K))
    print("Raters spread (0.2, 0.2, 0.2, 0.2, 0.1, 0.1); forecast (0, 0, 0.9, 0.1, 0, 0):")
    print(f"  squared-absolute RPS: truth {sa(p):.3f}, concentrated forecast {sa(q):.3f}")
    print(f"  RPS:                  truth {expected_rps(p, p):.3f}, concentrated forecast {expected_rps(q, p):.3f}")
    print("Point mass d ratings away from r = 4, RPS:", [round(rps(point_mass(4 - d), 4), 1) for d in range(5)])


if __name__ == "__main__":
    main()
