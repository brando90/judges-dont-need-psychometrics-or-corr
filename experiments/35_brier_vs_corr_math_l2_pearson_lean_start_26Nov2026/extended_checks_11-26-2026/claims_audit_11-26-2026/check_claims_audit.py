#!/usr/bin/env python3
"""Line-by-line numerical check of every equation on the handwritten pages (experiment 35,
11-26-2026), plus the claims added around them.  Deterministic (fixed seed); exact fractions
wherever the quantity is rational, floats (tolerance 1e-12) where a square root appears.

Run:  python3 -I check_claims_audit.py            (about ten seconds)

Every check prints one line, PASS or FAIL, with the page line it tests.  The matching Lean
statements are in AuditClaims.lean (names in brackets).  Synthetic distributions only; no
human ratings.
"""
import math
import random
from fractions import Fraction as F

import numpy as np
from scipy import integrate

SEED = 26112026
rng = random.Random(SEED)
results = []


def check(cid, text, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {cid} {text}" + (f": {detail}" if detail else ""), flush=True)


# ---------------------------------------------------------------------------------------------
# helpers: a finite joint law is a list of (probability, x, y)
# ---------------------------------------------------------------------------------------------
def random_joint(r, nmax=7, vmax=5):
    n = r.randint(3, nmax)
    pts = set()
    while len(pts) < n:
        pts.add((r.randint(0, vmax), r.randint(0, vmax)))
    ws = [r.randint(1, 9) for _ in pts]
    tot = sum(ws)
    return [(F(w, tot), x, y) for w, (x, y) in zip(ws, sorted(pts))]


def Ex(J, f):
    return sum(p * f(x, y) for p, x, y in J)


def moments(J):
    mx = Ex(J, lambda x, y: x)
    my = Ex(J, lambda x, y: y)
    vx = Ex(J, lambda x, y: (x - mx) ** 2)
    vy = Ex(J, lambda x, y: (y - my) ** 2)
    cv = Ex(J, lambda x, y: (x - mx) * (y - my))
    return mx, my, vx, vy, cv


def good_joint(r):
    while True:
        J = random_joint(r)
        mx, my, vx, vy, cv = moments(J)
        if vx > 0 and vy > 0:
            return J


# ---------------------------------------------------------------------------------------------
# C1  page 1 identities, exactly   [cov_eq_raw, var_eq_raw, E_sq, E_mul, sqdiff_expand, boxed_raw]
# ---------------------------------------------------------------------------------------------
N = 300
trials = [good_joint(rng) for _ in range(N)]
ok_a = ok_b = ok_c = ok_d = ok_e = ok_f = True
for J in trials:
    mx, my, vx, vy, cv = moments(J)
    exy = Ex(J, lambda x, y: x * y)
    ex2 = Ex(J, lambda x, y: x * x)
    ey2 = Ex(J, lambda x, y: y * y)
    sq = Ex(J, lambda x, y: (x - y) ** 2)
    ok_a &= cv == exy - mx * my                       # Cov = E[XY] - E[X]E[Y]
    ok_b &= vx == ex2 - mx ** 2 and vx == ex2 - mx * mx   # Var = E[X^2] - mu^2 = E[X^2] - E[X]^2
    ok_c &= ex2 == vx + mx ** 2 and ey2 == vy + my ** 2 and exy == cv + mx * my
    ok_d &= sq == ex2 + ey2 - 2 * exy                 # E[(X-Y)^2] = E[X^2]+E[Y^2]-2E[XY]
    ok_e &= sq == vx + vy - 2 * cv + mx ** 2 + my ** 2 - 2 * mx * my   # boxed line, page 1 and 3
    ok_f &= sq == (mx - my) ** 2 + (vx + vy - 2 * cv)
check("C1a", f"page 1 right margin Cov = E[XY]-E[X]E[Y], exact on {N} random joint laws", ok_a)
check("C1b", "page 1/2 Var[X] = E[X^2]-mu^2 = E[X]^2 form, exact", ok_b)
check("C1c", "page 1 small notes E[X^2]=sigma^2+mu^2 and E[XY]=cov+mu_X mu_Y, exact", ok_c)
check("C1d", "page 1 E[(X-Y)^2] = E[X^2]+E[Y^2]-2E[XY], exact", ok_d)
check("C1e", "page 1 boxed (and page 3) E[(X-Y)^2] = s_X^2+s_Y^2-2cov+mu_X^2+mu_Y^2-2mu_X mu_Y, exact", ok_e)
check("C1f", "location gap plus centered squared difference, exact", ok_f)

# ---------------------------------------------------------------------------------------------
# C2  standardized lines   [E_Z, E_Z_sq, E_ZZ, std_sqdiff, three_term, var_Z]
# ---------------------------------------------------------------------------------------------
tol = 1e-12
m0 = m1 = m2 = m3 = m4 = m5 = 0.0
for J in trials:
    mx, my, vx, vy, cv = [float(t) for t in moments(J)]
    sx, sy = math.sqrt(vx), math.sqrt(vy)
    rho = cv / (sx * sy)
    zx = lambda x: (x - mx) / sx
    zy = lambda y: (y - my) / sy
    Ef = lambda f: float(sum(p * f(float(x), float(y)) for p, x, y in J))
    m0 = max(m0, abs(Ef(lambda x, y: zx(x))))                         # E[Z_X] = 0
    m1 = max(m1, abs(Ef(lambda x, y: zx(x) ** 2) - 1))                # E[Z_X^2] = 1
    m2 = max(m2, abs(Ef(lambda x, y: (zx(x) - zy(y)) ** 2) - 2 * (1 - rho)))   # page 1 boxed
    m3 = max(m3, abs(Ef(lambda x, y: (zx(x) + zy(y)) ** 2) - 2 * (1 + rho)))   # (a+b)^2 line
    sqd = Ef(lambda x, y: (x - y) ** 2)
    m4 = max(m4, abs(sqd - ((mx - my) ** 2 + (sx - sy) ** 2 + 2 * sx * sy * (1 - rho))))  # three-term
    mu_z = Ef(lambda x, y: zx(x))
    m5 = max(m5, abs(Ef(lambda x, y: (zx(x) - mu_z) ** 2) - 1))       # Var[Z_X] = 1
check("C2a", "page 2 E[X~] = (1/sigma)(E[X]-mu) = 0 for the z-score", m0 < tol, f"max |E[Z_X]| = {m0:.1e}")
check("C2b", "page 2 Var[X~] = E[(X-mu)^2]/sigma^2 = 1", m1 < tol and m5 < tol, f"max gaps {m1:.1e}, {m5:.1e}")
check("C2c", "page 1 boxed E[(Z_X-Z_Y)^2] = 2(1-corr)", m2 < tol, f"max gap {m2:.1e} on {N} laws")
check("C2d", "page 2 line (a+b)^2 = ... with unit second moments: E[(Z_X+Z_Y)^2] = 2(1+corr)", m3 < tol, f"{m3:.1e}")
check("C2e", "three-term identity (mu gap)^2+(sigma gap)^2+2 s_X s_Y (1-rho)", m4 < tol, f"max gap {m4:.1e}")

# ---------------------------------------------------------------------------------------------
# C3  the slip on page 2 and the scribbled line   [slip_missing_two, slip_agrees_iff,
#     scribbled_line_false, centered_sqdiff]
# ---------------------------------------------------------------------------------------------
dev = 0.0
diffs_nonzero = 0
cent_diff = 0
for J in trials:
    mx, my, vx, vy, cv = [float(t) for t in moments(J)]
    sx, sy = math.sqrt(vx), math.sqrt(vy)
    rho = cv / (sx * sy)
    zx = lambda x: (x - mx) / sx
    zy = lambda y: (y - my) / sy
    slip = float(sum(p * (zx(float(x)) ** 2 - zx(float(x)) * zy(float(y)) + zy(float(y)) ** 2) for p, x, y in J))
    dev = max(dev, abs(slip - (2 - rho)))
    if abs(slip - 2 * (1 - rho)) > 1e-9:
        diffs_nonzero += 1
    # new meaning of the tilde (centered): E[(Xc - Yc)^2] = vx + vy - 2cov, generally not 2(1-rho)
    cen = float(sum(p * ((float(x) - mx) - (float(y) - my)) ** 2 for p, x, y in J))
    if abs(cen - 2 * (1 - rho)) > 1e-9:
        cent_diff += 1
check("C3a", "page 2 first expansion as written (cross term -X~Y~, no factor 2) equals 2 - rho, not 2(1-rho)",
      dev < tol and diffs_nonzero > 0.9 * N,
      f"max |value-(2-rho)| = {dev:.1e}; it differs from 2(1-rho) on {diffs_nonzero} of {N} laws (exactly when rho != 0)")
# scribbled line: Var[X-mu]/sqrt(Var[Z_X]) = Var[X], not 1
bad = 0
for J in trials:
    mx, my, vx, vy, cv = [float(t) for t in moments(J)]
    if abs(vx - 1) > 1e-9:
        bad += 1
        assert abs(vx / math.sqrt(1.0) - vx) < 1e-15
check("C3b", "page 2 scribbled line Var[X~] = Var[X-mu]/sqrt(Var[X~]) is false: right side is Var[X], not 1",
      bad > 0.9 * N, f"false on {bad} of {N} laws (true only when Var[X] = 1); correct divisor is sigma^2")
check("C3c", "after the PS (tilde = centered) page 1's boxed line would be wrong if the tilde were kept: "
      "E[(Xc-Yc)^2] = var_X+var_Y-2cov != 2(1-rho)", cent_diff > 0.9 * N,
      f"differs on {cent_diff} of {N} laws; the standardized variable needs its own symbol Z")

# ---------------------------------------------------------------------------------------------
# C4  page 3 degrees of freedom   [finrank_ker_momMap, four_free_directions, ...]
# ---------------------------------------------------------------------------------------------
def rank_exact(M):
    M = [[F(v) for v in row] for row in M]
    r = 0
    rows, cols = len(M), len(M[0])
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        for i in range(rows):
            if i != r and M[i][c] != 0:
                f = M[i][c] / M[r][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        r += 1
        if r == rows:
            break
    return r


table = {}
okdim = True
for K in range(2, 11):
    for j in range(0, K):
        Mm = [[i ** k for i in range(K)] for k in range(j + 1)]
        dim = K - rank_exact(Mm)
        table[(K, j)] = dim
        okdim &= dim == K - 1 - j
k6 = [table[(6, j)] for j in range(1, 6)]
check("C4a", "page 3 '6 degrees of freedom, 1 equation': free directions after normalization and j moments "
      "= K-1-j (exact rank, K=2..10, every j)", okdim, f"K=6, j=1..5 gives {k6}; K=2, j=1 gives {table[(2,1)]}")
check("C4b", "page 3: six ratings, normalization plus the mean leave four free directions, not six",
      table[(6, 1)] == 4 and k6 == [4, 3, 2, 1, 0])
# two different strictly positive pmfs with mean 5/2 and one with the extreme coin
cands = [[F(1, 6)] * 6, [F(1, 12), F(1, 6), F(1, 4), F(1, 4), F(1, 6), F(1, 12)], [F(1, 2), 0, 0, 0, 0, F(1, 2)],
         [0, 0, F(1, 2), F(1, 2), 0, 0]]
means = [sum(i * p for i, p in enumerate(c)) for c in cands]
check("C4c", "page 3: infinitely many laws share a mean: four explicit pmfs, all with mean 5/2, pairwise different",
      all(m == F(5, 2) for m in means) and len({tuple(c) for c in cands}) == 4 and all(sum(c) == 1 for c in cands))

# ---------------------------------------------------------------------------------------------
# C5  fixed pairing: conditional mean decomposition   [fixed_pairing_decomposition, condmean_unique_minimizer]
# ---------------------------------------------------------------------------------------------
okdec = True
okmin = True
for _ in range(200):
    nz = rng.randint(2, 5)
    nx = rng.randint(2, 5)
    w = [[F(rng.randint(1, 6)) for _ in range(nx)] for _ in range(nz)]
    tot = sum(sum(row) for row in w)
    w = [[v / tot for v in row] for row in w]
    v = [F(rng.randint(0, 5)) for _ in range(nx)]
    mz = [sum(row) for row in w]
    cm = [sum(w[z][x] * v[x] for x in range(nx)) / mz[z] for z in range(nz)]
    cvr = [sum(w[z][x] * (v[x] - cm[z]) ** 2 for x in range(nx)) / mz[z] for z in range(nz)]
    f = [F(rng.randint(-10, 60), 10) for _ in range(nz)]
    risk = lambda ff: sum(w[z][x] * (v[x] - ff[z]) ** 2 for z in range(nz) for x in range(nx))
    okdec &= risk(f) == sum(mz[z] * cvr[z] for z in range(nz)) + sum(mz[z] * (cm[z] - f[z]) ** 2 for z in range(nz))
    okmin &= risk(cm) <= risk(f) and (risk(f) == risk(cm)) == (f == cm)
check("C5a", "page 3 squared difference with a fixed pairing: E[(X-f(Z))^2] = E[Var(X|Z)] + E[(E[X|Z]-f(Z))^2], exact on 200 random tables", okdec)
check("C5b", "the conditional mean is the unique minimizer", okmin)
# two coins: same conditional mean, different irreducible error
pcoin = [F(1, 2), 0, 0, 0, 0, F(1, 2)]
qcoin = [0, 0, F(1, 2), F(1, 2), 0, 0]
mp = sum(i * p for i, p in enumerate(pcoin))
mq = sum(i * p for i, p in enumerate(qcoin))
vp = sum(p * (i - mp) ** 2 for i, p in enumerate(pcoin))
vq = sum(p * (i - mq) ** 2 for i, p in enumerate(qcoin))
check("C5c", "the two coins have the same conditional mean 5/2 (same minimizer) and irreducible errors 25/4 and 1/4",
      mp == mq == F(5, 2) and vp == F(25, 4) and vq == F(1, 4))

# ---------------------------------------------------------------------------------------------
# C6  coupling: lower bound and zero   [sqdiff_lower_bound, zero_sqdiff_forces_equal_laws, sqdiff_self]
# ---------------------------------------------------------------------------------------------
worst = 1.0
eq_cases = 0
for J in trials:
    mx, my, vx, vy, cv = [float(t) for t in moments(J)]
    sqd = float(sum(p * (x - y) ** 2 for p, x, y in J))
    gap = sqd - ((mx - my) ** 2 + (math.sqrt(vx) - math.sqrt(vy)) ** 2)
    worst = min(worst, gap)
check("C6a", "for every coupling E[(X-Y)^2] >= (mu_X-mu_Y)^2 + (sigma_X-sigma_Y)^2", worst > -1e-12, f"smallest gap {worst:.2e} on {N} laws")
# equality when Y is a positive affine function of X
J = [(F(1, 4), 0, 1), (F(1, 2), 1, 3), (F(1, 4), 2, 5)]     # Y = 2X + 1
mx, my, vx, vy, cv = [float(t) for t in moments(J)]
sqd = float(sum(p * (x - y) ** 2 for p, x, y in J))
check("C6b", "equality holds exactly when rho = 1 (Y = 2X+1 example)",
      abs(sqd - ((mx - my) ** 2 + (math.sqrt(vx) - math.sqrt(vy)) ** 2)) < 1e-12)
# zero squared difference forces equal laws: any coupling with E[(X-Y)^2] = 0 has X = Y on the support
okz = True
for _ in range(100):
    n = rng.randint(2, 6)
    vals = [rng.randint(0, 5) for _ in range(n)]
    ws = [F(rng.randint(1, 5)) for _ in range(n)]
    tot = sum(ws)
    J = [(w / tot, v, v) for w, v in zip(ws, vals)]
    lx, ly = {}, {}
    for p, x, y in J:
        lx[x] = lx.get(x, 0) + p
        ly[y] = ly.get(y, 0) + p
    okz &= sum(p * (x - y) ** 2 for p, x, y in J) == 0 and lx == ly
check("C6c", "page 3 'if X = Y the squared difference is zero too' and conversely zero forces equal laws", okz)
# pair A: minimum over couplings is 4, attained by the comonotone coupling; a in [0,1/2]
costs = [F(9) - 10 * F(k, 2000) for k in range(0, 1001)]            # a = k/2000 in [0, 1/2]
check("C6d", "two coins (0 or 5 against 2 or 3): minimum squared difference over all couplings is 4 (cost 9-10a, a<=1/2)",
      min(costs) == 4 and costs[-1] == 4)

# ---------------------------------------------------------------------------------------------
# C7  independent draws   [indep_formula, indep_point, indep_truthful, truthful_not_optimal]
# ---------------------------------------------------------------------------------------------
def rpmf(r, K=6):
    ws = [F(r.randint(0, 6)) for _ in range(K)]
    if sum(ws) == 0:
        ws[0] = F(1)
    t = sum(ws)
    return [w / t for w in ws]


okindep = okpt = okimp = True
n_impr = 0
for _ in range(300):
    p, q = rpmf(rng), rpmf(rng)
    mp_ = sum(i * x for i, x in enumerate(p))
    mq_ = sum(i * x for i, x in enumerate(q))
    vp_ = sum(x * (i - mp_) ** 2 for i, x in enumerate(p))
    vq_ = sum(x * (i - mq_) ** 2 for i, x in enumerate(q))
    ind = sum(p[i] * q[j] * (i - j) ** 2 for i in range(6) for j in range(6))
    okindep &= ind == vp_ + vq_ + (mp_ - mq_) ** 2
    # point masses
    best = min(vp_ + (y - mp_) ** 2 for y in range(6))
    okpt &= all(sum(p[i] * (i - y) ** 2 for i in range(6)) == vp_ + (y - mp_) ** 2 for y in range(6))
    truthful = 2 * vp_
    far = any(p[i] > 0 and (i - mp_) ** 2 > min((y - mp_) ** 2 for y in range(6)) for i in range(6))
    if far:
        okimp &= best < truthful
        n_impr += 1
    else:
        okimp &= best == truthful
check("C7a", "independent draws: E[(X-Y)^2] = sigma_p^2 + sigma_q^2 + (mu_p-mu_q)^2, exact on 300 random pairs", okindep)
check("C7b", "against a point mass y: E[(X-y)^2] = sigma_p^2 + (y-mu_p)^2", okpt)
check("C7c", "the truthful report q = p is beaten by a point mass unless p lives on its nearest rating(s)", okimp,
      f"strict improvement on {n_impr} of 300 random laws")
p0 = [F(1, 2), 0, 0, 0, 0, F(1, 2)]
vals = [sum(p0[i] * (i - y) ** 2 for i in range(6)) for y in range(6)]
check("C7d", "p = half on 0, half on 5: truthful report scores 25/2; point masses at 0..5 score 25/2, 17/2, 13/2, 13/2, 17/2, 25/2",
      vals == [F(25, 2), F(17, 2), F(13, 2), F(13, 2), F(17, 2), F(25, 2)] and 2 * vp == F(25, 2))

# ---------------------------------------------------------------------------------------------
# C8  log-loss proper, one-to-one scores are not   [gibbs, linear_score_improper, square_score_improper]
# ---------------------------------------------------------------------------------------------
def rposf(r, K):
    ws = [r.random() + 0.05 for _ in range(K)]
    s = sum(ws)
    return [w / s for w in ws]


okg = True
eqcount = 0
for _ in range(2000):
    K = rng.randint(2, 6)
    p, q = rposf(rng, K), rposf(rng, K)
    ce_q = -sum(a * math.log(b) for a, b in zip(p, q))
    ce_p = -sum(a * math.log(a) for a in p)
    okg &= ce_q >= ce_p - 1e-12
    if max(abs(a - b) for a, b in zip(p, q)) < 1e-9:
        eqcount += 1
check("C8a", "Gibbs: expected log-loss of any report q is at least the truth's, on 2000 random pairs", okg)

grid = np.linspace(0, 1, 100001)
p = np.array([0.6, 0.4])
scores = {
    "identity q(x)": lambda q: q,
    "square q(x)^2": lambda q: q ** 2,
    "sqrt q(x)": lambda q: np.sqrt(q),
    "log q(x)": lambda q: np.log(np.clip(q, 1e-300, None)),
    "Brier-type -(1-q(x))^2": lambda q: -(1 - q) ** 2,
}
verdict = {}
for name, phi in scores.items():
    exp = p[0] * phi(grid) + p[1] * phi(1 - grid)         # expected score of report (q, 1-q) when the truth is p
    verdict[name] = grid[np.argmax(exp)]
proper = {k: abs(v - 0.6) < 1e-4 for k, v in verdict.items()}
check("C8b", "K=2, truth p=(0.6,0.4), scores that are one-to-one in q(x): the best report for q(x) and for q(x)^2 is the corner q=1, "
      "for sqrt q(x) it is 9/13 = 0.692 (exaggerates the mode); only log q(x) and the local Brier form -(1-q(x))^2 are best at the truth 0.6",
      (not proper["identity q(x)"]) and abs(verdict["identity q(x)"] - 1.0) < 1e-9
      and (not proper["square q(x)^2"]) and abs(verdict["square q(x)^2"] - 1.0) < 1e-9
      and (not proper["sqrt q(x)"]) and abs(verdict["sqrt q(x)"] - 9 / 13) < 1e-4
      and proper["log q(x)"] and proper["Brier-type -(1-q(x))^2"],
      "best report q: " + ", ".join(f"{k}: {v:.4f}" for k, v in verdict.items()))

# ---------------------------------------------------------------------------------------------
# C9  Bayes rule of the 6.437 margin   [bayes_rule_optimal]
# ---------------------------------------------------------------------------------------------
okb = True
for _ in range(200):
    nY, nH, nA = rng.randint(2, 5), rng.randint(2, 4), rng.randint(2, 4)
    py = rposf(rng, nY)
    post = [rposf(rng, nH) for _ in range(nY)]
    C = [[rng.random() * 5 for _ in range(nH)] for _ in range(nA)]
    inner = lambda y, a: sum(post[y][h] * C[a][h] for h in range(nH))
    rule = [min(range(nA), key=lambda a: inner(y, a)) for y in range(nY)]
    risk = lambda rl: sum(py[y] * inner(y, rl[y]) for y in range(nY))
    for _ in range(50):
        other = [rng.randrange(nA) for _ in range(nY)]
        okb &= risk(rule) <= risk(other) + 1e-12
check("C9", "6.437 top-right margin: the per-observation posterior-cost argmin minimizes expected cost over all rules (200 random problems, 50 rival rules each)", okb)

# ---------------------------------------------------------------------------------------------
# C10 page 1 question and page 2 density Brier   [no Lean: calculus]
# ---------------------------------------------------------------------------------------------
okn = True
for _ in range(200):
    y, f = rng.uniform(-3, 3), rng.uniform(-3, 3)
    s = rng.uniform(0.3, 2.0)
    nll = -math.log(math.exp(-(y - f) ** 2 / (2 * s * s)) / (s * math.sqrt(2 * math.pi)))
    okn &= abs(nll - ((y - f) ** 2 / (2 * s * s) + math.log(s) + 0.5 * math.log(2 * math.pi))) < 1e-12
ys = [rng.uniform(-2, 4) for _ in range(50)]
fbest = min(np.linspace(-5, 8, 130001), key=lambda f: sum((v - f) ** 2 for v in ys))
check("C10a", "page 1 question: squared error is the Gaussian negative log-likelihood up to constants; its minimizer is the sample mean",
      okn and abs(fbest - sum(ys) / len(ys)) < 1e-3)
N0 = lambda x, m, s: np.exp(-(x - m) ** 2 / (2 * s * s)) / (s * math.sqrt(2 * math.pi))
pdf_p = lambda x: N0(x, 0.0, 1.0)
pdf_q = lambda x: N0(x, 1.0, 1.5)
l2, _ = integrate.quad(lambda x: (pdf_p(x) - pdf_q(x)) ** 2, -40, 40)
closed = 1 / (2 * math.sqrt(math.pi) * 1.0) + 1 / (2 * math.sqrt(math.pi) * 1.5) - 2 * N0(0.0, 1.0, math.sqrt(1 + 2.25))
qq, _ = integrate.quad(lambda x: pdf_q(x) ** 2, -40, 40)
pp, _ = integrate.quad(lambda x: pdf_p(x) ** 2, -40, 40)
gap, _ = integrate.quad(lambda y: pdf_p(y) * ((qq - 2 * pdf_q(y)) - (pp - 2 * pdf_p(y))), -40, 40)
check("C10b", "page 2 'or integral (p*(x)-q^(x))^2 dx': expected quadratic-score gap equals the integrated squared difference of the densities",
      abs(gap - l2) < 1e-8 and abs(l2 - closed) < 1e-8, f"{gap:.10f} vs {l2:.10f} (closed form {closed:.10f})")

# ---------------------------------------------------------------------------------------------
# C11 page 2 note: "centering doesn't really chg pmf/pdf of a 1-1 invertible op RV"
# ---------------------------------------------------------------------------------------------
pm = {0: F(1, 4), 1: F(1, 2), 2: F(1, 4)}
T = lambda a: 3 * a - 7                                          # injective
pm_T = {T(a): p for a, p in pm.items()}
check("C11a", "pmf: masses are preserved under an injective map (support relabelled): {0,1,2} -> {-7,-4,-1} with masses 1/4,1/2,1/4",
      sorted(pm_T.values()) == sorted(pm.values()) and set(pm_T) != set(pm))
u = (np.arange(200000) + 0.5) / 200000                           # Uniform[0,1] by midpoint grid
mu_u, sd_u = 0.5, math.sqrt(1 / 12)
z = (u - mu_u) / sd_u
# density = count / (number of ALL samples x bin width).  (numpy's density=True would divide by the
# samples inside the plotted range only, which cuts off part of the support (-1.732, 1.732) and
# reports 1/3 instead of sigma; the first version of this check made exactly that mistake.)
counts, edges = np.histogram(z, bins=40, range=(-1.5, 1.5))
dens = counts / (len(z) * (edges[1] - edges[0]))
pdf_u_at_mid = 1.0                                               # density of Uniform[0,1] before standardizing
check("C11b", "pdf: the standardized Uniform[0,1] has density sigma = 0.2887 (support (-1.732, 1.732)), not the original height 1: rescaling changes pdf heights, translation only moves them",
      abs(dens[20] - sd_u) < 5e-3 and abs(dens[20] - pdf_u_at_mid) > 0.5 and abs(dens[5] - sd_u) < 5e-3,
      f"density near 0 is {dens[20]:.4f}, near -1.1 is {dens[5]:.4f}; sigma = {sd_u:.4f}")
# the same fact by the change-of-variables formula, exactly: pdf_Z(z) = sigma * pdf_U(mu + sigma z)
zs = np.linspace(-1.7, 1.7, 7)
pdf_Z = sd_u * np.where(np.abs(mu_u + sd_u * zs - 0.5) <= 0.5, 1.0, 0.0)
check("C11c", "change of variables pdf_Z(z) = sigma * pdf_U(mu + sigma z) = sigma on the support, and the area is 1",
      np.allclose(pdf_Z[np.abs(zs) < 1.7], sd_u) and abs(sd_u * (1 / sd_u) - 1) < 1e-15)

print()
print(f"{sum(results)} of {len(results)} checks PASS")
if not all(results):
    raise SystemExit(1)
