"""Numerical and exact checks for BRIER_PROPER_6437_STYLE_11-26-2026.md.

Brier score is strictly proper (Claim 1'), local only for two outcomes, and the
ranked probability score (RPS) inherits strict propriety. Synthetic and exact
arithmetic only; no human ratings. Uses numpy and fractions only. Deterministic
(every generator is numpy.random.default_rng with a fixed seed).

Notation: x = rating in {0, ..., K-1}, p = true rating distribution of an item,
q = reported distribution. Brier(x, q) = sum_a (q(a) - 1{x = a})^2.

Run from anywhere:  python3 -I check_brier_proper_6437_style.py
"""
from fractions import Fraction as Fr

import numpy as np

RESULTS = []


def report(name, ok, detail):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    for line in detail:
        print("       " + line)
    print()


# ---------------------------------------------------------------- helpers
def exp_brier_direct(p, q):
    """E_{x~p}[ sum_a (q(a) - 1{x=a})^2 ] computed literally from the definition.
    p, q: arrays of shape (n, K). Returns shape (n,)."""
    K = p.shape[1]
    eye = np.eye(K)
    # cost[n, x] = sum_a (q[n, a] - eye[x, a])^2
    cost = ((q[:, None, :] - eye[None, :, :]) ** 2).sum(axis=2)
    return (p * cost).sum(axis=1)


def exp_brier_identity(p, q):
    return ((q - p) ** 2).sum(axis=1) + 1.0 - (p ** 2).sum(axis=1)


def vec(v, d=4):
    return "(" + ", ".join(f"{float(a):.{d}f}" for a in v) + ")"


def brier_frac(x, q):
    return sum((qa - (1 if a == x else 0)) ** 2 for a, qa in enumerate(q))


def rps_direct(p, q):
    """Unnormalized expected RPS, E_{x~p} sum_{k=0}^{K-2} (F_q(k) - 1{x<=k})^2."""
    K = p.shape[1]
    Fq = np.cumsum(q, axis=1)[:, : K - 1]
    ind = (np.arange(K)[:, None] <= np.arange(K - 1)[None, :]).astype(float)  # ind[x, k]
    cost = ((Fq[:, None, :] - ind[None, :, :]) ** 2).sum(axis=2)  # (n, x)
    return (p * cost).sum(axis=1)


def simplex_grid3(n):
    pts = [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]
    return np.array(pts)


# ---------------------------------------------------------------- (i)
def check_i():
    rng = np.random.default_rng(1)
    lines, ok = [], True
    for K in (2, 3, 6, 10):
        p = rng.dirichlet(np.ones(K), size=10_000)
        q = rng.dirichlet(np.ones(K), size=10_000)
        gap = np.max(np.abs(exp_brier_direct(p, q) - exp_brier_identity(p, q)))
        ok &= gap < 1e-12
        lines.append(f"K={K:2d}: 10,000 random (p, q), max |E[Brier|y] - (||q-p||^2 + 1 - ||p||^2)| = {gap:.2e}")
    report("(i) E[Brier|y] = ||q-p||^2 + 1 - ||p||^2", ok, lines)


# ---------------------------------------------------------------- (ii)
def check_ii():
    rng = np.random.default_rng(2)
    lines, ok = [], True
    for K in (2, 3, 6, 10):
        n = 10_000
        p = rng.dirichlet(np.ones(K), size=n)
        q = rng.dirichlet(np.ones(K), size=n)
        A = rng.uniform(0.01, 10.0, size=n)
        B = rng.normal(0.0, 5.0, size=(n, K))
        direct = A * exp_brier_direct(p, q) + (p * B).sum(axis=1)
        at_p = A * exp_brier_direct(p, p) + (p * B).sum(axis=1)
        ident = A * ((q - p) ** 2).sum(axis=1) + A * (1 - (p ** 2).sum(axis=1)) + (p * B).sum(axis=1)
        gap = np.max(np.abs(direct - ident))
        excess_gap = np.max(np.abs((direct - at_p) - A * ((q - p) ** 2).sum(axis=1)))
        ok &= gap < 1e-9 and excess_gap < 1e-9 and np.all(direct >= at_p - 1e-12)
        lines.append(f"K={K:2d}: A~U(0.01,10), B~N(0,25): max identity gap {gap:.2e}, "
                     f"max |excess - A||q-p||^2| {excess_gap:.2e}, min excess {np.min(direct - at_p):.3e}")
    report("(ii) A*Brier + B(x): E = A||q-p||^2 + A(1-||p||^2) + E[B(x)|y]", ok, lines)


# ---------------------------------------------------------------- (iii)
def check_iii():
    rng = np.random.default_rng(3)
    lines, ok = [], True
    for K in (3, 6, 10):
        p = rng.dirichlet(np.ones(K) * 2, size=1)
        base = exp_brier_direct(p, p)[0]
        # tangent perturbations that keep q in the simplex
        d = rng.normal(size=(20_000, K))
        d -= d.mean(axis=1, keepdims=True)
        t = rng.uniform(0, 1, size=(20_000, 1))
        tmax = np.min(np.where(d < 0, p / -np.where(d < 0, d, -1), np.inf), axis=1, keepdims=True)
        q = p + t * tmax * d
        q = np.clip(q, 0, None)
        q /= q.sum(axis=1, keepdims=True)
        vals = exp_brier_direct(np.repeat(p, len(q), 0), q)
        incr = vals - base
        ok &= np.all(incr > 0)
        lines.append(f"K={K:2d}: E[Brier] at p = {base:.6f}; 20,000 perturbations q != p: "
                     f"min increase {incr.min():.3e} (all > 0: {bool(np.all(incr > 0))})")
    # fine grid for K = 3
    p = np.array([[0.2, 0.5, 0.3]])
    G = simplex_grid3(200)
    vals = exp_brier_direct(np.repeat(p, len(G), 0), G)
    i = int(np.argmin(vals))
    second = np.sort(vals)[1]
    ok_grid = np.allclose(G[i], p[0], atol=1e-12)
    ok &= ok_grid
    lines.append(f"K= 3 grid step 1/200 ({len(G)} points), p = (0.2, 0.5, 0.3): argmin = "
                 f"({G[i][0]:.3f}, {G[i][1]:.3f}, {G[i][2]:.3f}), min = {vals[i]:.6f}, "
                 f"next-smallest = {second:.6f} (gap {second - vals[i]:.2e} = 2/200^2 = {2/200**2:.2e})")
    report("(iii) p minimises expected Brier over the simplex", ok, lines)


# ---------------------------------------------------------------- (iv)
def check_iv():
    lines, ok = [], True
    K = 6
    p = np.array([0.05, 0.10, 0.20, 0.30, 0.25, 0.10])
    # (a) empirical minimiser is the frequency vector
    rng = np.random.default_rng(4)
    x = rng.choice(K, size=500, p=p)
    f = np.bincount(x, minlength=K) / len(x)
    eye = np.eye(K)

    def emp_brier(q):
        return ((q[:, None, :] - eye[x][None, :, :]) ** 2).sum(axis=2).mean(axis=1)

    Q = rng.dirichlet(np.ones(K), size=5_000)
    ident_gap = np.max(np.abs(emp_brier(Q) - (((Q - f) ** 2).sum(1) + 1 - (f ** 2).sum())))
    at_f = emp_brier(f[None])[0]
    beaten = np.sum(emp_brier(Q) < at_f)
    ok &= ident_gap < 1e-12 and beaten == 0
    lines.append(f"N=500 sample from p={vec(p, 2)}: frequencies f = {vec(f, 3)}")
    lines.append(f"  empirical Brier identity (||q-f||^2 + 1 - ||f||^2) max gap {ident_gap:.2e}; "
                 f"empirical Brier at f = {at_f:.6f}; random q beating f: {beaten} of 5,000")
    # (b) E||f - p||^2 = (1 - ||p||^2)/N
    one_minus = 1 - (p ** 2).sum()
    lines.append(f"E||f-p||^2 vs (1-||p||^2)/N, 1-||p||^2 = {one_minus:.4f}; 5 seeds x 2,000 replications each")
    for N in (10, 100, 1_000, 10_000, 100_000, 1_000_000):
        per_seed = []
        for seed in range(5):
            r = np.random.default_rng(100 + seed)
            counts = r.multinomial(N, p, size=2_000)
            per_seed.append(((counts / N - p) ** 2).sum(1).mean())
        mean = float(np.mean(per_seed))
        pred = one_minus / N
        ratio = mean / pred
        ok &= abs(ratio - 1) < 0.05
        lines.append(f"  N={N:>9,d}: mean {mean:.4e}  predicted {pred:.4e}  ratio {ratio:.4f}  "
                     f"(seed ratios {', '.join(f'{v / pred:.3f}' for v in per_seed)})")
    report("(iv) empirical Brier minimiser = frequencies; E||f-p||^2 = (1-||p||^2)/N", ok, lines)


# ---------------------------------------------------------------- (v)
def check_v():
    rng = np.random.default_rng(5)
    K = 6
    p = rng.dirichlet(np.ones(K), size=10_000)
    q = rng.dirichlet(np.ones(K), size=10_000)
    Fp = np.cumsum(p, 1)[:, : K - 1]
    Fq = np.cumsum(q, 1)[:, : K - 1]
    ident = ((Fq - Fp) ** 2).sum(1) + (Fp * (1 - Fp)).sum(1)
    gap = np.max(np.abs(rps_direct(p, q) - ident))
    excess = rps_direct(p, q) - rps_direct(p, p)
    ok = gap < 1e-12 and np.all(excess > 0)
    # normalized RPS (divide by K-1), used as the primary metric in this repository
    gapn = np.max(np.abs(rps_direct(p, q) / (K - 1) - ident / (K - 1)))
    report("(v) E[RPS|y] = sum_k (F_q(k)-F_p(k))^2 + sum_k F_p(k)(1-F_p(k)), K=6", ok,
           [f"10,000 random (p, q): max identity gap {gap:.2e} (normalized by K-1: {gapn:.2e}); "
            f"min excess E[RPS(q)] - E[RPS(p)] = {excess.min():.3e}"])


# ---------------------------------------------------------------- (vi)
def check_vi():
    lines, ok = [], True
    rng = np.random.default_rng(6)
    q = rng.dirichlet(np.ones(2), size=10_000)
    x = rng.integers(0, 2, size=10_000)
    cost = ((q - np.eye(2)[x]) ** 2).sum(1)
    phi = 2 * (1 - q[np.arange(len(q)), x]) ** 2
    gap = np.max(np.abs(cost - phi))
    ok &= gap < 1e-14
    lines.append(f"K=2: 10,000 random (x, q): max |Brier(x,q) - 2(1-q(x))^2| = {gap:.2e}  (local)")
    # exact counterexample, L = 3, outcome = first label (index 0)
    qa = (Fr(1, 2), Fr(1, 2), Fr(0))
    qb = (Fr(1, 2), Fr(1, 4), Fr(1, 4))
    ca, cb = brier_frac(0, qa), brier_frac(0, qb)
    ok &= qa[0] == qb[0] and ca == Fr(1, 2) and cb == Fr(3, 8) and ca != cb
    lines.append(f"K=3, x = first outcome: q = (1/2, 1/2, 0) -> Brier = {ca}; "
                 f"q' = (1/2, 1/4, 1/4) -> Brier = {cb}; q(x) = q'(x) = {qa[0]}  (not local)")
    # a cloud at K = 6: fixed x and fixed q(x) = 0.4, different costs
    x0, qx = 2, 0.4
    rest = rng.dirichlet(np.ones(5), size=1_000) * (1 - qx)
    qs = np.insert(rest, x0, qx, axis=1)
    costs = ((qs - np.eye(6)[x0]) ** 2).sum(1)
    ok &= costs.max() - costs.min() > 0.1
    lines.append(f"K=6, x = {x0}, q(x) = {qx} fixed, 1,000 random q: Brier ranges "
                 f"{costs.min():.4f} to {costs.max():.4f}  (= (1-q(x))^2 + sum_(a!=x) q(a)^2)")
    # RPS is not local either: K = 3, x = first label, same q(x) = 1/2
    def rps_frac(x, q):
        F, out = Fr(0), Fr(0)
        for k in range(len(q) - 1):
            F += q[k]
            out += (F - (1 if x <= k else 0)) ** 2
        return out
    ra, rc = rps_frac(0, qa), rps_frac(0, (Fr(1, 2), Fr(0), Fr(1, 2)))
    ok &= ra == Fr(1, 4) and rc == Fr(1, 2)
    lines.append(f"RPS (unnormalized), K=3, x = first outcome: q = (1/2, 1/2, 0) -> {ra}; "
                 f"q'' = (1/2, 0, 1/2) -> {rc}  (not local)")
    report("(vi) locality: Brier local at K=2, not local at K>=3", ok, lines)


# ---------------------------------------------------------------- (vii)
def water_fill(p):
    """argmin_q sum_i p_i (1 - q_i)^2 over the simplex: q_i = max(0, 1 - c/p_i)."""
    lo, hi = 0.0, float(p.max())
    for _ in range(200):
        c = 0.5 * (lo + hi)
        s = np.maximum(0, 1 - c / p).sum()
        lo, hi = (c, hi) if s > 1 else (lo, c)
    c = 0.5 * (lo + hi)
    return np.maximum(0, 1 - c / p), c


def check_vii():
    lines, ok = [], True
    G = simplex_grid3(400)
    for p in (np.array([0.2, 0.5, 0.3]), np.array([0.6, 0.3, 0.1]), np.array([0.4, 0.35, 0.25])):
        v1 = ((1 - G) ** 2 * p).sum(1)
        g1 = G[int(np.argmin(v1))]
        wf, c = water_fill(p)
        v2 = 1 - (G * p).sum(1)
        g2 = G[int(np.argmin(v2))]
        not_p1 = np.max(np.abs(g1 - p)) > 0.01
        not_p2 = np.max(np.abs(g2 - p)) > 0.01
        agree = np.max(np.abs(g1 - wf)) <= 1 / 400 + 1e-12
        ok &= not_p1 and not_p2 and agree
        lines.append(f"p = {vec(p, 2)}")
        lines.append(f"  C = (1-q(x))^2: water-filling q = ({', '.join(f'{v:.4f}' for v in wf)}), c = {c:.4f}; "
                     f"grid (step 1/400) argmin = ({', '.join(f'{v:.4f}' for v in g1)})")
        lines.append(f"     expected cost at water-filling q = {((1 - wf) ** 2 * p).sum():.6f} "
                     f"< at p = {((1 - p) ** 2 * p).sum():.6f}")
        lines.append(f"  C = 1 - q(x): grid argmin = ({', '.join(f'{v:.4f}' for v in g2)}) "
                     f"(the vertex at argmax p); expected cost {v2.min():.4f} < at p {1 - (p * p).sum():.4f}")
    report("(vii) negative controls: local costs (1-q(x))^2 and 1-q(x) are not proper at K=3", ok, lines)


# ---------------------------------------------------------------- worked example
def worked_example():
    p = np.array([0.7, 0.3])
    lines = ["true p = (0.7, 0.3); expected Brier = sum (q-p)^2 + 1 - ||p||^2; log-loss in nats"]
    ok = True
    exact_brier = {}
    for q in ((0.7, 0.3), (0.5, 0.5), (0.9, 0.1)):
        qa = np.array(q)
        b = exp_brier_direct(p[None], qa[None])[0]
        ll = -(p * np.log(qa)).sum()
        pf = (Fr(7, 10), Fr(3, 10))
        qf = tuple(Fr(v).limit_denominator(100) for v in q)
        bf = sum(pf[x] * brier_frac(x, qf) for x in range(2))
        exact_brier[q] = bf
        lines.append(f"q = {q}: E[Brier] = {b:.4f} (exact {bf}), excess {abs(b - 0.42):.4f}; "
                     f"E[log-loss] = {ll:.4f} nats, excess = KL(p||q) = {abs(ll + (p * np.log(p)).sum()):.4f}")
    ok &= exact_brier[(0.7, 0.3)] == Fr(21, 50) and exact_brier[(0.5, 0.5)] == Fr(1, 2) \
        and exact_brier[(0.9, 0.1)] == Fr(1, 2)
    report("worked example (K=2, p=(0.7,0.3))", ok, lines)


if __name__ == "__main__":
    check_i()
    check_ii()
    check_iii()
    check_iv()
    check_v()
    check_vi()
    check_vii()
    worked_example()
    n_pass = sum(ok for _, ok in RESULTS)
    print(f"SUMMARY: {n_pass} PASS, {len(RESULTS) - n_pass} FAIL of {len(RESULTS)} checks")
