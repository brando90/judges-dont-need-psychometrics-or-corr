"""Numerical check of the cross-entropy (CE) versus correlation information gap (experiment 05).

Free, CPU only, numpy/scipy. Run: python gap_check.py   (prints results and writes results/gap_check.json)
No model calls, no human ratings. Scores are integers 0..K-1 with K = 6 (the 0-5 rating scale).
"""
import json, pathlib
import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

K = 6
Y = np.arange(K)
OUT = pathlib.Path(__file__).parent / 'results'


def H(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def KL(p, q):
    m = p > 0
    return float((p[m] * (np.log(p[m]) - np.log(q[m]))).sum())


def maxent(m):
    """Maximum-entropy distribution on {0..K-1} with mean m: q(y) ~ exp(lam*y) (tilted uniform)."""
    if abs(m - (K - 1) / 2) < 1e-12:
        return np.ones(K) / K
    f = lambda lam: (Y * np.exp(lam * Y)).sum() / np.exp(lam * Y).sum() - m
    lam = brentq(f, -60, 60)
    w = np.exp(lam * Y - (lam * Y).max())
    return w / w.sum()


def rand_simplex(rng):
    return rng.dirichlet(np.ones(K) * rng.choice([0.3, 1.0, 3.0]))


def main():
    rng = np.random.default_rng(0)
    res = {}
    # Identity 1: CE(P, q) = H(P) + KL(P || q)
    e1 = 0
    for _ in range(2000):
        p, q = rand_simplex(rng), rand_simplex(rng)
        ce = float(-(p * np.log(q)).sum())
        e1 = max(e1, abs(ce - (H(p) + KL(p, q))))
    res['identity_ce_equals_entropy_plus_kl_max_abs_err'] = e1
    # Identity 2: KL(P || q_m) = H(q_m) - H(P) when q_m is max-entropy with the mean of P
    e2, gaps, ratios = 0, [], []
    for _ in range(2000):
        p = rand_simplex(rng)
        m = float((Y * p).sum())
        qm = maxent(m)
        g = KL(p, qm)
        e2 = max(e2, abs(g - (H(qm) - H(p))))
        gaps.append(g)
    res['identity_gap_equals_entropy_deficit_max_abs_err'] = e2
    res['gap_over_2000_random_P_min_mean_max'] = [min(gaps), float(np.mean(gaps)), max(gaps)]
    # Property 3: q_m is minimax for log loss over {P : mean(P) = m}: E_P[-log q_m] = H(q_m) for every such P
    m = 2.2; qm = maxent(m); worst = []
    for _ in range(2000):
        p = rand_simplex(rng)
        # project to mean m by mixing with a tilted point: rejection sampling on the mean
        if abs((Y * p).sum() - m) < 0.02:
            worst.append(float(-(p * np.log(qm)).sum()))
    res['minimax_check_mean_2.2'] = {'n_P': len(worst), 'H_qm': H(qm), 'CE_of_qm_on_P_min_max': [min(worst), max(worst)]}
    # Property 4: CE-optimal => correlation-optimal; converse false (exact, population level, many items)
    n = 11
    th = np.arange(1, n + 1) / 12
    P = np.array([0.9 * binom.pmf(Y, K - 1, t) + 0.1 / K for t in th])  # experiment 04 construction A1
    mP = P @ Y
    Qm = np.array([maxent(m) for m in mP])
    Q_noisy = P.copy()  # CE-optimal judge
    rho = lambda a: float(np.corrcoef(mP, a)[0, 1])
    rows = []
    for j in range(n):
        rows.append(dict(item=j + 1, mean=float(mP[j]), H_human=H(P[j]), H_maxent=H(Qm[j]), gap_nats=KL(P[j], Qm[j]),
                         tvd_to_maxent=0.5 * float(np.abs(P[j] - Qm[j]).sum())))
    res['A1_items'] = rows
    res['A1_mean_gap_nats'] = float(np.mean([r['gap_nats'] for r in rows]))
    res['A1_pearson_of_judge_mean_vs_human_mean'] = {'true_distribution_judge': rho(Q_noisy @ Y), 'maxent_judge': rho(Qm @ Y)}
    res['A1_CE_nats_mean'] = {'true_distribution_judge': float(np.mean([-(P[j] * np.log(P[j])).sum() for j in range(n)])),
                              'maxent_judge_same_means': float(np.mean([-(P[j] * np.log(Qm[j])).sum() for j in range(n)]))}
    # Pearson blind to everything past the mean: a mean-matched judge with a wrong shape (mass pushed to the ends)
    bad = []
    for j in range(n):
        p = P[j]; m = mP[j]
        lo, hi = 0, K - 1
        w = (hi - m) / (hi - lo)       # two-point distribution on {0, 5} with the same mean
        q = np.full(K, 1e-3); q[lo] += w; q[hi] += 1 - w; q /= q.sum()
        # re-center mean to m exactly via tilt on the two points
        bad.append(q)
    bad = np.array(bad)
    res['A1_two_point_judge'] = {'mean_abs_mean_error': float(np.abs(bad @ Y - mP).mean()),
                                 'pearson_vs_human_mean': rho(bad @ Y),
                                 'CE_nats_mean': float(np.mean([-(P[j] * np.log(bad[j])).sum() for j in range(n)])),
                                 'KL_nats_mean': float(np.mean([KL(P[j], bad[j]) for j in range(n)]))}
    # Experiment 00 synthetic: judge B squashed (order preserving) vs judge A noisy-faithful
    r0 = np.random.default_rng(0); h = r0.integers(0, 6, 2000)
    A = np.clip(h + r0.integers(-1, 2, 2000), 0, 5); B = np.where(h >= 3, 5, 4)
    res['exp00_pearson'] = {'A': float(np.corrcoef(h, A)[0, 1]), 'B': float(np.corrcoef(h, B)[0, 1])}
    # Finite family check: softmax over affine-in-score logits (exponential family, sufficient statistic y):
    # CE minimization is moment matching, so CE-opt and Pearson-opt coincide: gap needs richer judges.
    p = P[2]
    best = min(((-(p * np.log(np.exp(l * Y - 0) / np.exp(l * Y).sum())).sum()), l) for l in np.linspace(-3, 3, 60001))
    res['affine_softmax_family'] = {'best_slope': best[1], 'mean_of_best': float((np.exp(best[1] * Y) / np.exp(best[1] * Y).sum()) @ Y), 'mean_of_P': float(p @ Y)}
    (OUT / 'gap_check.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'A1_items'}, indent=1))
    for r in rows:
        print({k: round(v, 4) for k, v in r.items()})


if __name__ == '__main__':
    main()
