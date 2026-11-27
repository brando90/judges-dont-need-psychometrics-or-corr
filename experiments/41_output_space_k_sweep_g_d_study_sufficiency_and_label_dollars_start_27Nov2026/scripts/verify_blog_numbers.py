#!/usr/bin/env python3
"""Fact-check numbers and identities of the live blog post 'The Bitter Judges Lesson' (Experiment 41 QA, 11-27-2026).

Part 1 (B1-B13) re-derives the post's scaling-law numbers from the saved fits of
experiments/31_scaling_laws_for_judges_ce_vs_pearson_start_25Nov2026/results/fits_11-25-2026.csv and the sentence quoted from
experiments/28_public_rating_sets_chaosnli_dices_frontier_judges_ce_vs_pearson_start_23Nov2026/README.md.
Part 2 (B14-B22) checks the appendix identities in exact rational arithmetic (fractions) or, where a square root is needed,
in floating point at a named tolerance with a fixed seed. Each PASS line says whether it confirms a claim of the post or a
caveat to it. Prints PASS or FAIL per check; exits 1 on any FAIL. Deterministic, no network.

Run: python3 -I verify_blog_numbers.py [repo_root]
"""
import csv
import math
import random
import statistics
import sys
from fractions import Fraction as F
from itertools import product
from pathlib import Path

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]
E31 = root / "experiments/31_scaling_laws_for_judges_ce_vs_pearson_start_25Nov2026/results/fits_11-25-2026.csv"
E28 = root / "experiments/28_public_rating_sets_chaosnli_dices_frontier_judges_ce_vs_pearson_start_23Nov2026/README.md"
SEED = 20261127
FLOAT_TOL = 1e-12
FAILED = []


def check(tag, ok, detail=""):
    print(("PASS" if ok else "FAIL"), tag, detail, flush=True)
    if not ok:
        FAILED.append(tag)


# ---------------------------------------------------------------- part 1: scaling-law numbers
rows = list(csv.DictReader(E31.open()))
panels = {}
for r in rows:
    if r["metric"] != "rps":
        continue
    pid = r["panel_id"]
    if (r["exp"] == "28" and r["level"] == "set" and pid.endswith("_d1")) or (r["exp"] == "29" and r["level"] == "panel" and pid.endswith("_d1")):
        panels.setdefault(pid, {})[r["objective"]] = r
check("[B1] eleven one-rating-per-item panels (5 Exp 28 sets + 6 Exp 29 LoRA panels)", len(panels) == 11, f"found {len(panels)}")


def n_fit(obj, strict=False):
    n = 0
    for d in panels.values():
        v = d[obj]["verdict"]
        if not v.startswith("fits"):
            continue
        if strict and "indicative" in v:
            continue
        n += 1
    return n


check("[B2] post: '9 of 11' (CE-trained curve 'fits')", n_fit("CE") == 9, f"count {n_fit('CE')}")
check("[B3] post: '1 of 11' (Pearson-trained curve 'fits')", n_fit("Pearson") == 1, f"count {n_fit('Pearson')}")
ind = sorted(pid for pid, d in panels.items() if "indicative" in d["CE"]["verdict"])
check("[B4] caveat: three of the nine CE 'fits' are four-point, one-degree-of-freedom fits labeled indicative", len(ind) == 3, f"{ind}")
check("[B5] caveat: counting only fits with more than four budgets gives 6 of 11 for the CE-trained curve", n_fit("CE", strict=True) == 6, f"count {n_fit('CE', strict=True)}")
fl = sorted(pid for pid, d in panels.items() if d["CE"]["verdict"].startswith("fits") and d["CE"]["floor_identified"] == "False")
check("[B6] caveat: three of the Exp 28 CE fits have no identified floor", len(fl) == 3 and all(p.startswith("28_") for p in fl), f"{fl}")
pf = sorted(pid for pid, d in panels.items() if d["Pearson"]["verdict"].startswith("fits"))
check("[B7] caveat: the one Pearson 'fit' has exponent b <= 0.2 (a near-flat curve)", len(pf) == 1 and "b <= 0.2" in panels[pf[0]]["Pearson"]["verdict"], f"{pf}")
n_flat_or_poor = sum(1 for d in panels.values() if not d["Pearson"]["verdict"].startswith("fits") or "b <= 0.2" in d["Pearson"]["verdict"])
check("[B7b] caveat: in 11 of 11 panels the Pearson-trained curve is flat, non-monotone, poorly fit or a b <= 0.2 fit", n_flat_or_poor == 11, f"count {n_flat_or_poor}")

vals = {}
for r in rows:
    if r["exp"] == "28" and r["level"] == "cell" and r["objective"] == "CE" and r["panel_id"].endswith("_d1"):
        vals.setdefault(r["metric"], {"all": 0, "ident": []})
        vals[r["metric"]]["all"] += 1
        if r["floor_identified"] == "True" and r["b"] not in ("", "nan"):
            vals[r["metric"]]["ident"].append(float(r["b"]))
med = {m: statistics.median(v["ident"]) for m, v in vals.items()}
cnt = {m: (len(v["ident"]), v["all"]) for m, v in vals.items()}
check("[B8] post: median exponent 0.66 on RPS (Exp 28 judge-by-set cells, 1 rating per item, floor identified)", abs(med["rps"] - 0.66) < 0.005, f"{med['rps']:.4f} over {cnt['rps'][0]} of {cnt['rps'][1]} cells")
check("[B9] post: median exponent 0.82 on CE (same cells)", abs(med["ce"] - 0.82) < 0.005, f"{med['ce']:.4f} over {cnt['ce'][0]} of {cnt['ce'][1]} cells")
check("[B9b] caveat: the memo states the denominators the post omits: 8 of 25 cells (RPS) and 20 of 25 (CE)", cnt["rps"] == (8, 25) and cnt["ce"] == (20, 25), f"{cnt}")
check("[B10] Exp 31's own text: median exponent 0.59 on TVD (not quoted in the post)", abs(med["tvd"] - 0.59) < 0.005, f"{med['tvd']:.4f} over {cnt['tvd'][0]} of {cnt['tvd'][1]} cells")
nmax = sorted({int(float(r["N_max"])) for d in panels.values() for r in d.values()})
check("[B11] post: largest budgets from a few hundred to about two thousand ratings", min(nmax) >= 256 and max(nmax) == 2048, f"N_max values {nmax}")
text = E28.read_text()
check("[B12] Exp 28 README: 'after 4 to 8 ratings (one per item)' (the post's 4-to-8 claim)", "after 4 to 8 ratings (one per item)" in text)
check("[B13] caveat in the same Exp 28 sentence: Pearson-then-calibration beats CE-fit on DICES from 64 ratings", "Pearson-then-calibration is better than CE-fit on DICES from 64 ratings" in text)
n_exp28 = sum(1 for pid in panels if pid.startswith("28_"))
n_exp29 = sum(1 for pid in panels if pid.startswith("29_"))
qwen = sum(1 for pid in panels if pid.startswith("29_") and "qwen" in pid)
llama = sum(1 for pid in panels if pid.startswith("29_") and "llama" in pid)
check("[B13b] caveat: the 11 panels are 5 frozen-frontier-judge panels (mean of five judges) plus 6 LoRA panels (4 Qwen3-4B, 2 Llama-3.1-8B), and HelpSteer2 is among them",
      (n_exp28, n_exp29, qwen, llama) == (5, 6, 4, 2) and "29_hs2_qwen_d1" in panels, f"exp28={n_exp28} exp29={n_exp29} qwen={qwen} llama={llama}")

# ---------------------------------------------------------------- part 2: appendix identities
# B14: binary label, judge draw X ~ Bern(q) independent of Y ~ Bern(p).
grid = [F(i, 8) for i in range(9)]
ok14 = True
for q, p in product(grid, grid):
    sampled = q * (1 - p) + (1 - q) * p                    # E[(X-Y)^2] for a sampled 0/1 draw = P(X != Y)
    brier1 = p * (1 - q) ** 2 + (1 - p) * q ** 2            # one-class Brier of the forecast q, averaged over Y ~ Bern(p)
    ok14 &= (sampled == brier1 + q * (1 - q))
check("[B14] caveat on A.1: for a SAMPLED judge draw, E[(X-Y)^2] = Brier(q) + q(1-q), not the Brier score (exact, 81 rational (q,p) pairs)", ok14)
check("[B14b] caveat on A.1: q = p = 1/2 gives sampled-draw squared error 1/2 versus Brier 1/4", (F(1, 2) * F(1, 2) + F(1, 2) * F(1, 2), F(1, 2) * F(1, 4) + F(1, 2) * F(1, 4)) == (F(1, 2), F(1, 4)))
ok15 = True
for q, y in product(grid, (0, 1)):
    two_class = (q - y) ** 2 + ((1 - q) - (1 - y)) ** 2
    ok15 &= (two_class == 2 * (q - y) ** 2)
check("[B15] caveat on A.1: with the sum-over-classes Brier the appendix defines, binary Brier is twice the scalar squared error (exact)", ok15)

# B16: plug-in distance bias, exact multinomial enumeration (K = 3).
def plugin_bias(k):
    p = (F(1, 2), F(1, 3), F(1, 6))
    q = (F(1, 5), F(1, 5), F(3, 5))
    exp_dist = F(0)
    for n0 in range(k + 1):
        for n1 in range(k + 1 - n0):
            n2 = k - n0 - n1
            prob = F(math.factorial(k), math.factorial(n0) * math.factorial(n1) * math.factorial(n2))
            prob *= p[0] ** n0 * p[1] ** n1 * p[2] ** n2
            ph = (F(n0, k), F(n1, k), F(n2, k))
            exp_dist += prob * sum((a - b) ** 2 for a, b in zip(ph, q))
    true_dist = sum((a - b) ** 2 for a, b in zip(p, q))
    return exp_dist - true_dist, F(1, k) * (1 - sum(a * a for a in p))


ok16 = all(lhs == rhs for lhs, rhs in (plugin_bias(k) for k in (1, 2, 3, 4, 5)))
check("[B16] post A.4: E over resampling of the plug-in distance exceeds the true distance by (1/k)(1 - ||p*||^2) (exact, k = 1..5); the post omits 'in expectation'", ok16)

# B17: A.2 on a concrete finite instance (three equally likely prompts, ratings on 0..5).
prompts = [{1: F(1)}, {0: F(1, 2), 4: F(1, 2)}, {2: F(1, 2), 5: F(1, 2)}]
w = F(1, len(prompts))
mu = [sum(y * pr for y, pr in d.items()) for d in prompts]
m2 = [sum(y * y * pr for y, pr in d.items()) for d in prompts]
EY = w * sum(mu)
VarY = w * sum(m2) - EY ** 2
VarMu = w * sum(m * m for m in mu) - EY ** 2
rho = VarMu / VarY
# sampled-from-p* judge X independent of Y given the prompt: same law as Y, so Var X = Var Y and Cov(X, Y) = Var(mu)
cov_X_Y = w * sum(m * m for m in mu) - EY * EY
corr2_true = cov_X_Y ** 2 / (VarY * VarY)
# collapsed affine copy X_d = a*mu + b with a = 2, b = 1: Cov(X_d, Y) = a Var(mu), Var(X_d) = a^2 Var(mu)
a = F(2)
corr2_collapsed = (a * VarMu) ** 2 / (a * a * VarMu * VarY)
check("[B17] post A.2 on an exact instance: Corr(X_p*, Y) = rho = Var(mu)/Var(Y) in (0,1)", corr2_true == rho ** 2 and 0 < rho < 1, f"rho = {rho}")
check("[B17b] post A.2 on the same instance: Corr(X_collapsed, Y)^2 = rho, so sqrt(rho) > rho", corr2_collapsed == rho and corr2_collapsed > corr2_true, f"rho = {rho}, rho^2 = {rho ** 2}")
check("[B17c] post A.2 arithmetic: sqrt(0.5) = 0.7071 (rounded)", abs(math.sqrt(0.5) - 0.7071) < 5e-5)

# B18: identities of A.1 on a random finite joint (floats, fixed seed).
rng = random.Random(SEED)
for trial in range(5):
    n = 12
    pr = [rng.random() for _ in range(n)]
    s = sum(pr)
    pr = [x / s for x in pr]
    X = [rng.uniform(-2, 3) for _ in range(n)]
    Y = [rng.uniform(-1, 4) for _ in range(n)]
    ex = sum(p * x for p, x in zip(pr, X)); ey = sum(p * y for p, y in zip(pr, Y))
    vx = sum(p * (x - ex) ** 2 for p, x in zip(pr, X)); vy = sum(p * (y - ey) ** 2 for p, y in zip(pr, Y))
    cxy = sum(p * (x - ex) * (y - ey) for p, x, y in zip(pr, X, Y))
    corr = cxy / math.sqrt(vx * vy)
    mse = sum(p * (x - y) ** 2 for p, x, y in zip(pr, X, Y))
    rhs = (ex - ey) ** 2 + (math.sqrt(vx) - math.sqrt(vy)) ** 2 + 2 * math.sqrt(vx * vy) * (1 - corr)
    zl2 = sum(p * ((x - ex) / math.sqrt(vx) - (y - ey) / math.sqrt(vy)) ** 2 for p, x, y in zip(pr, X, Y))
    cov_aff = sum(p * ((3 * x + 7) - (3 * ex + 7)) * (y - ey) for p, x, y in zip(pr, X, Y)) / math.sqrt(9 * vx * vy)
    ok = abs(mse - rhs) < 1e-10 and abs(zl2 - 2 * (1 - corr)) < 1e-10 and abs(cov_aff - corr) < 1e-10
    check(f"[B18.{trial}] post A.1 identities on a random joint: MSE decomposition, standardized L2 = 2(1 - Corr), positive-affine invariance (tolerance 1e-10)", ok)

print()
if FAILED:
    print("FAILED:", ", ".join(FAILED))
    sys.exit(1)
print("ALL PASS")
