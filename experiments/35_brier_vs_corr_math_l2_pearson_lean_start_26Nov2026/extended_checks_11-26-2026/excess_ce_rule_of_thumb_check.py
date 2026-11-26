"""Check the rule of thumb for the excess cross-entropy of a per-item histogram estimate.

Run from anywhere (paths come from __file__):

    python3 -I excess_ce_rule_of_thumb_check.py

Question. A judge that outputs a distribution over K = 6 ratings for one item is fitted from n
independent ratings of that item. How much worse than the true distribution is the fit, in
expected cross-entropy, when n is small? The asymptotic rule of thumb is (K - 1) / (2 n) nats.

Data. Synthetic only; no human ratings and no model calls. Six random true distributions over the
ratings 0 to 5 (Dirichlet with all six concentrations equal to 3). For each true distribution and
each n, 40,000 independent draws of n ratings. Seed 20261126.

Estimator. Empirical frequencies with add-one smoothing, (count + 1) / (n + K), so the divergence
is finite. This is a Laplace-smoothed histogram, not a trained judge.

Metric. Excess cross-entropy of the estimate over the true distribution, which is the
Kullback-Leibler divergence KL(q || p_hat) = sum_r q_r log(q_r / p_hat_r), in nats. Lower is
better. It is zero only when the estimate equals the truth. Reference value: the rule of thumb
(K - 1) / (2 n).

Reported number. The mean over the 40,000 draws, averaged over the six truths, next to the rule of
thumb, and their ratio (measured over predicted). A ratio of 1 means the rule is exact; a ratio
below 1 means the rule overstates the excess. No interval is computed; with 40,000 draws per cell
the Monte Carlo error of each mean is far below the third decimal of the ratio (printed per cell
as the standard error of the six-truth average).
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "excess_ce_rule_of_thumb_output.txt"

K = 6
REPS = 40000
NS = [20, 50, 200, 1000, 5000]
N_TRUTHS = 6
SEED = 20261126

rng = np.random.default_rng(SEED)
cell = {n: [] for n in NS}       # per-truth mean excess KL
cell_se = {n: [] for n in NS}    # per-truth standard error of that mean
truths = []
for _ in range(N_TRUTHS):
    q = rng.dirichlet(np.full(K, 3.0))
    truths.append(q)
    for n in NS:
        counts = rng.multinomial(n, q, size=REPS)
        p_hat = (counts + 1.0) / (n + K)
        kl = np.sum(q * np.log(q / p_hat), axis=1)
        cell[n].append(kl.mean())
        cell_se[n].append(kl.std(ddof=1) / np.sqrt(REPS))

lines = []
lines.append("Excess cross-entropy (KL divergence) of a Laplace-smoothed per-item histogram, synthetic data")
lines.append(f"K = {K} ratings (0 to 5); {N_TRUTHS} random true distributions; {REPS} draws per (truth, n); seed {SEED}")
lines.append("Metric: KL(q || p_hat) in nats; lower is better; reference = (K - 1) / (2 n)")
lines.append("")
lines.append(f"{'n':>6s} {'measured mean':>14s} {'rule (K-1)/(2n)':>16s} {'ratio':>7s} {'SE of mean':>11s}")
for n in NS:
    m = float(np.mean(cell[n]))
    rt = (K - 1) / (2 * n)
    se = float(np.sqrt(np.sum(np.square(cell_se[n]))) / N_TRUTHS)
    lines.append(f"{n:6d} {m:14.5f} {rt:16.5f} {m / rt:7.3f} {se:11.6f}")
lines.append("")
lines.append("Per-truth ratios (measured / predicted), rows = truths, columns = n")
lines.append("      " + " ".join(f"{n:>7d}" for n in NS))
for t in range(N_TRUTHS):
    lines.append(f"truth{t + 1} " + " ".join(f"{cell[n][t] / ((K - 1) / (2 * n)):7.3f}" for n in NS))

text = "\n".join(lines) + "\n"
OUT.write_text(text, encoding="utf-8")
print(text, end="")
