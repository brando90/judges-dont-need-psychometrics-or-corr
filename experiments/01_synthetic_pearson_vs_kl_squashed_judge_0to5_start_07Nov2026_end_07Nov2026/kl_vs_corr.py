"""Toy counterexample: high correlation, yet the judge's score distribution
does not match the human one (large KL).

Scores are on a 0-5 integer scale. The "biased judge" is a monotone, nearly
linear transform of the human score (compressed toward the top of the scale),
so Pearson and Spearman are high, but its score histogram is very different.

Run: python kl_vs_corr.py   (needs numpy, scipy)
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr

K = 6  # scores 0..5
rng = np.random.default_rng(0)


def hist(scores, k=K, alpha=1.0):
    """Score histogram with additive (Laplace) smoothing so KL is finite."""
    counts = np.bincount(scores, minlength=k).astype(float) + alpha
    return counts / counts.sum()


def kl(p, q):
    """KL(p || q) in nats."""
    return float(np.sum(p * np.log(p / q)))


def report(name, human, judge):
    r = pearsonr(human, judge)[0]
    rho = spearmanr(human, judge)[0]
    d = kl(hist(human), hist(judge))
    mse = np.mean(((human - judge) / (K - 1)) ** 2)
    print(f"{name:<22} pearson={r:5.3f} spearman={rho:5.3f} "
          f"KL(P_h||P_j)={d:6.3f} nMSE={mse:5.3f}")


n = 2000
human = rng.integers(0, K, size=n)  # humans use the full 0-5 range

# Judge A: noisy copy of the human score (matches the distribution).
judge_a = np.clip(human + rng.integers(-1, 2, size=n), 0, K - 1)

# Judge B: perfectly ordered but squashed to {4,5}: "everything is good".
judge_b = np.where(human >= 3, 5, 4)

report("A: noisy faithful", human, judge_a)
report("B: inflated/squashed", human, judge_b)
print("Human hist:  ", np.round(hist(human), 3))
print("Judge B hist:", np.round(hist(judge_b), 3))
