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


if __name__ == "__main__":
    main()
