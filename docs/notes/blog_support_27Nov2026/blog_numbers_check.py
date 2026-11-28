#!/usr/bin/env python3
"""Numbers check for the evidence-led rewrite of the blog post 'The Bitter Judges Lesson' (11-27-2026).

Every number the post quotes is re-derived here from a committed result file of this repository (CSV, JSON,
a saved log) or by exact rational arithmetic (fractions). Prints PASS or FAIL per claim, then the registry of
numeric tokens that the post may use; exits 1 if any claim fails. Deterministic: fixed seed, no network.
The scaling-law counts of experiment 31 are re-derived by experiment 41's verify_blog_numbers.py, which this
script runs and whose PASS lines it requires.

Run: python3 -I blog_numbers_check.py [repo_root]
Companion: scan_post_numbers.py POST.md checks that every number token in the post is in the registry printed here.
"""
import csv
import glob
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter
from fractions import Fraction as Fr
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments"
SEED = 20261127
REGISTRY = {}
FAILED = []
NCHECK = [0]


def one(pattern):
    hits = sorted(glob.glob(str(EXP / pattern)))
    assert len(hits) == 1, (pattern, hits)
    return Path(hits[0])


def check(cid, ok, text, tokens=()):
    NCHECK[0] += 1
    print(("PASS" if ok else "FAIL"), f"[{cid}]", text, flush=True)
    if not ok:
        FAILED.append(cid)
    for t in tokens:
        REGISTRY.setdefault(str(t), []).append(cid)


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rd(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ======================================================================= T: toy example, exact fractions
K = 6  # a 0-5 scale
panel = [Fr(1, 2) if k in (0, 4) else Fr(0) for k in range(K)]  # ten raters, five say 0 and five say 4
certain2 = [Fr(1) if k == 2 else Fr(0) for k in range(K)]


def brier(p, y):
    return sum((p[k] - (1 if k == y else 0)) ** 2 for k in range(K))


def rps(p, y):
    s, F, G = Fr(0), Fr(0), Fr(0)
    for k in range(K - 1):
        F += p[k]
        G = Fr(1) if y <= k else Fr(0)  # cumulative indicator of the observed rating
        s += (F - G) ** 2
    return s


def expect(score, p):
    return sum(panel[y] * score(p, y) for y in range(K))


mean = lambda p: sum(k * p[k] for k in range(K))
check("T1", mean(panel) == 2 and mean(certain2) == 2,
      f"both forecasts have mean {mean(panel)}, so any score that sees only the mean ties them", ["2"])
check("T2", (expect(brier, panel), expect(brier, certain2)) == (Fr(1, 2), Fr(2)),
      f"expected Brier against the panel: truthful {expect(brier, panel)}, certain-2 {expect(brier, certain2)}", ["0.5", "2.0"])
check("T3", (expect(rps, panel), expect(rps, certain2)) == (Fr(1), Fr(2)),
      f"expected RPS (cumulative Brier): truthful {expect(rps, panel)}, certain-2 {expect(rps, certain2)}", ["1.0", "2.0"])
check("T4", (expect(rps, panel) / (K - 1), expect(rps, certain2) / (K - 1)) == (Fr(1, 5), Fr(2, 5)),
      f"normalized RPS (RPS / (K-1) = RPS / {K - 1}): {expect(rps, panel) / (K - 1)} vs {expect(rps, certain2) / (K - 1)}", ["0.2", "0.4"])
mse = lambda p: sum(p[x] * panel[y] * (x - y) ** 2 for x in range(K) for y in range(K))
check("T5", (mse(panel), mse(certain2)) == (Fr(8), Fr(4)),
      f"squared error of one sampled judgment against one human rating: truthful {mse(panel)}, certain-2 {mse(certain2)} (the collapsed judge looks twice as good)", ["8", "4"])


def law_stats(items):
    """items: list of (item_prob, {y: prob}). Returns exact moments of Y, mu(X)."""
    EY = sum(w * sum(y * q for y, q in d.items()) for w, d in items)
    EY2 = sum(w * sum(y * y * q for y, q in d.items()) for w, d in items)
    VarY = EY2 - EY ** 2
    mus = [sum(y * q for y, q in d.items()) for w, d in items]
    Varmu = sum(w * (m - EY) ** 2 for (w, d), m in zip(items, mus))
    return EY, VarY, mus, Varmu


def table_truthful(items):
    """Joint law of (J, Y) for a judge that samples J from the item's rating distribution, independently of Y given the item."""
    t = {}
    for w, d in items:
        for j, qj in d.items():
            for y, qy in d.items():
                t[(j, y)] = t.get((j, y), Fr(0)) + w * qj * qy
    return t


def table_collapsed(items):
    """Joint law of (J', Y) for a judge that always reports the item's mean rating."""
    t = {}
    for w, d in items:
        m = sum(y * q for y, q in d.items())
        for y, qy in d.items():
            t[(m, y)] = t.get((m, y), Fr(0)) + w * qy
    return t


def table_arbitrary(items, scores):
    """Joint law of (J'', Y) for a judge that reports the fixed number scores[i] on item i."""
    t = {}
    for (w, d), s in zip(items, scores):
        for y, qy in d.items():
            t[(s, y)] = t.get((s, y), Fr(0)) + w * qy
    return t


def table_moments(t):
    """Exact moments of a finite joint law {(j, y): probability}: E J, E Y, Var J, Var Y, Cov(J, Y)."""
    EJ = sum(p * j for (j, y), p in t.items())
    EY = sum(p * y for (j, y), p in t.items())
    VarJ = sum(p * j * j for (j, y), p in t.items()) - EJ ** 2
    VarY = sum(p * y * y for (j, y), p in t.items()) - EY ** 2
    Cov = sum(p * j * y for (j, y), p in t.items()) - EJ * EY
    return EJ, EY, VarJ, VarY, Cov


def corr2_of(t):
    _, _, VarJ, VarY, Cov = table_moments(t)
    return Cov ** 2 / (VarJ * VarY)


def refit_error(t):
    """Mean squared error of the best straight-line fit of Y on J. The line comes from the normal equations, the error is
    summed over the joint table (not taken from a closed form), and the line is confirmed to be a minimum by exact nudges."""
    EJ, EY, VarJ, _, Cov = table_moments(t)
    b = Cov / VarJ
    a = EY - b * EJ
    err = lambda a_, b_: sum(p * (y - a_ - b_ * j) ** 2 for (j, y), p in t.items())
    nudge = Fr(1, 7)
    is_min = all(err(a + da, b + db) > err(a, b) for da, db in ((nudge, 0), (-nudge, 0), (0, nudge), (0, -nudge)))
    return err(a, b), is_min


def affine_map(t, a, b):
    out = {}
    for (j, y), p in t.items():
        out[(a * j + b, y)] = out.get((a * j + b, y), Fr(0)) + p
    return out


# the two-item example of the post: item 1 rated 0 or 2 equally often, item 2 rated 2 or 4 equally often
items = [(Fr(1, 2), {0: Fr(1, 2), 2: Fr(1, 2)}), (Fr(1, 2), {2: Fr(1, 2), 4: Fr(1, 2)})]
EY, VarY, mus, Varmu = law_stats(items)
rho = Varmu / VarY
_, _, vjt, vyt, cvt = table_moments(table_truthful(items))
check("T6", rho == Fr(1, 2) and vjt == vyt and cvt / vyt == rho,
      f"two-item law: rho = Var(mu)/Var(Y) = {rho}; the truthful sampling judge's correlation, computed from the joint table of judge and human, equals rho (= {cvt / vyt})", ["0.5"])
collapsed_corr2 = corr2_of(table_collapsed(items))
check("T7", collapsed_corr2 == rho and math.isclose(math.sqrt(collapsed_corr2), 0.5 ** 0.5, rel_tol=1e-15),
      f"collapsed judge (outputs mu(X)): squared correlation {collapsed_corr2} = rho from the joint table, so its correlation is sqrt(rho) = {math.sqrt(collapsed_corr2):.4f}", ["0.71"])

rng = random.Random(SEED)
rng2 = random.Random(SEED + 1)  # a separate stream for the arbitrary judge, so the random laws stay the same
n_laws, n_ok_exact, n_ok_gt, n_cases, n_ok_aff, n_ok_r2, n_ok_min = 2000, 0, 0, 0, 0, 0, 0
for _ in range(n_laws):
    ni = rng.randint(2, 5)
    ws = [rng.randint(1, 9) for _ in range(ni)]
    tw = sum(ws)
    its = []
    for w in ws:
        ys = rng.sample(range(0, 6), rng.randint(2, 4))
        qs = [rng.randint(1, 9) for _ in ys]
        tq = sum(qs)
        its.append((Fr(w, tw), {y: Fr(q, tq) for y, q in zip(ys, qs)}))
    EY, VarY, mus, Varmu = law_stats(its)
    if VarY == 0 or Varmu == 0 or Varmu == VarY:
        n_laws -= 1
        continue
    rho = Varmu / VarY
    tt, tc = table_truthful(its), table_collapsed(its)
    _, _, vjt, vyt, cvt = table_moments(tt)
    _, _, vjc, vyc, cvc = table_moments(tc)
    corr2_t, corr2_c = cvt ** 2 / (vjt * vyt), cvc ** 2 / (vjc * vyc)
    # truthful judge: Var(J) = Var(Y) and Cov(J, Y) = Var(mu), so its correlation is rho;
    # collapsed judge: Var(J') = Cov(J', Y) = Var(mu), so its squared correlation is rho. Every moment is read off the joint table.
    n_ok_exact += (vjt == vyt == VarY) and (cvt == Varmu) and (corr2_t == rho ** 2) and (vjc == cvc == Varmu) and (corr2_c == rho)
    # both covariances are positive, so each correlation is the positive root of its square and they compare as the squares do
    n_ok_gt += (cvt > 0) and (cvc > 0) and (corr2_c > corr2_t)
    a, b = Fr(rng.randint(1, 9), rng.randint(1, 9)), Fr(rng.randint(-9, 9), rng.randint(1, 9))
    scores = [rng2.randint(0, 5) for _ in its]
    for t in (tt, tc, table_arbitrary(its, scores)):
        _, _, VarJ_, VarY_, _ = table_moments(t)
        if VarJ_ == 0:
            continue
        n_cases += 1
        c2 = corr2_of(t)
        n_ok_aff += (corr2_of(affine_map(t, a, b)) == c2)
        mse_fit, is_min = refit_error(t)
        n_ok_r2 += (1 - mse_fit / VarY_ == c2)
        n_ok_min += is_min
check("T8", n_ok_exact == n_laws, f"{n_laws} random finite joint laws (seed {SEED}), each joint table of judge and human built explicitly: truthful judge's correlation = rho and collapsed judge's squared correlation = rho, exactly")
check("T9", n_ok_gt == n_laws, f"{n_laws} random laws: the collapsed judge's correlation sqrt(rho) exceeds the truthful judge's correlation rho whenever 0 < rho < 1")
check("T10", n_cases >= n_laws and n_ok_aff == n_cases and n_ok_r2 == n_cases and n_ok_min == n_cases,
      f"{n_cases} judge-by-law cases (truthful, collapsed and an arbitrary per-item judge on the {n_laws} random laws): squared correlation is unchanged by positive affine maps of the judge, "
      f"and equals 1 - MSE(best affine refit)/Var(Y) with the MSE summed over the joint table, exactly; the refit is a strict minimum")

# ======================================================================= P: how judges are validated today
sys.path.insert(0, str(one("43_*")))
import importlib
papers = importlib.import_module("papers")
R = papers.R
codes = [r[5].split(",") for r in R]
n49 = len(R)
nC = sum("C" in c for c in codes)
nP = sum("P" in c for c in codes)
check("P1", n49 == 49 and nC == 33, f"experiment 43: {nC} of {n49} papers report a correlation with human ratings", ["49", "33"])
check("P2", n49 - nP == 43, f"experiment 43: {n49 - nP} of {n49} papers have no item-level proper score or divergence against human ratings (broad code)", ["43"])
strict = (one("54_*") / "strict_tally_output.txt").read_text()
m = re.search(r"Strict: item-level strictly proper score: (\d+)/49; none: (\d+)/49", strict)
check("P3", m is not None and int(m.group(2)) == 44, f"experiment 54 strict recode: {m.group(2) if m else '?'} of 49 have no item-level strictly proper score", ["44"])

cen = rd(one("45_*") / "census.csv")
Dall = [r for r in cen if r["human_validation"] == "Y" and r["era"] == "llm" and r["lineage"] in ("L1", "L2", "L3", "L4", "L5") and r["coverage"] in ("full", "partial")]
CORR = ("CORR_INST", "CORR_SYS", "CORR_UNSTATED")
nh = sum(r["headline_family"] in CORR for r in Dall)
lo, hi = wilson(nh, len(Dall))
check("P4", len(Dall) == 40 and nh == 16 and round(lo, 3) == 0.263 and round(hi, 3) == 0.554,
      f"experiment 45 census: correlation is the headline agreement number in {nh} of {len(Dall)} sources = {nh / len(Dall):.3f}, Wilson 95% {lo:.3f} to {hi:.3f}", ["40", "16", "26", "55"])
check("P4b", len(cen) == 53 and len(Dall) == 40,
      f"experiment 45 census: {len(cen)} sources coded in all, a purposive sample chosen by lineage (not at random); {len(Dall)} are LLM-era sources read in full or in part that validate a judge against human labels", ["53"])
by = {}
for r in Dall:
    d = by.setdefault(r["lineage"], [0, 0])
    d[1] += 1
    d[0] += r["headline_family"] in CORR
check("P5", by["L1"] == [12, 14] and by["L3"] == [4, 6] and by["L2"] == [0, 8] and by["L4"] == [0, 6] and by["L5"] == [0, 6],
      f"headline correlation by lineage (L1 graded scores, L3 leaderboard/system ranking, L2 pairwise, L4 frontier-lab graders, L5 retrieval/annotation): {by}", ["12", "14", "4", "6", "20"])
p45 = (one("45_*") / "PROTOCOL_AND_PREDICTIONS_11-27-2026.md").read_text()
wl14 = wilson(12, 14)
check("P5b", 12 / 14 >= 0.80 and wl14[0] < 0.65 and "at least 80% of L1 rows have a correlation-family headline and the Wilson lower bound is at least 65%" in p45,
      f"census rule V1, written in the protocol before coding (a correlation headline in at least 80% of graded-score sources with Wilson lower bound at least 65%): observed 12 of 14 = {12 / 14:.3f}, Wilson lower bound {wl14[0]:.3f}, so the bar is missed")
rest = [r for r in Dall if r["lineage"] in ("L2", "L4", "L5")]
lead = sum(r["headline_family"] in ("ACC_AGREE", "KAPPA_ALPHA_ICC", "F1_PRF") for r in rest)
check("P6", len(rest) == 20 and lead == 19 and not any(r["headline_family"] in CORR for r in rest),
      f"of the {len(rest)} pairwise, frontier-lab and retrieval/annotation sources none leads with a correlation and {lead} lead with accuracy, agreement, kappa or F1", ["19"])
sel = [r for r in Dall if "Y" in (r["selection_by_corr"], r["selection_loose"], r["trained_on_human_loss"])]
sel_any = sum(r["selection_by_corr"] == "Y" for r in sel)
sel_corr = sum(r["selection_by_corr"] == "Y" and r["selection_metric"] in ("CORR", "BOTH") for r in sel)
check("P8", len(sel) == 25 and sel_any == 9 and sel_corr == 4,
      f"experiment 45 census: of the {len(sel)} sources that select, tune or train a judge on human data, {sel_any} say they maximized agreement or correlation with human labels and {sel_corr} correlation alone", ["25", "9", "4"])
dist = sum(r["dist_any"] == "Y" for r in Dall)
l303 = next(r for r in cen if r["id"] == "L3-03")
check("P7", dist == 1 and [r["id"] for r in Dall if r["dist_any"] == "Y"] == ["L3-03"] and "Pair Rank Brier Score" in l303["notes"] and "not to individual ratings" in l303["notes"],
      f"{dist} of {len(Dall)} sources scores a human distribution with a proper rule: L3-03 (Arena-Hard), a Brier score on pairwise system-rank forecasts, not on individual ratings; so none scores an item's human rating distribution", ["1"])

# ======================================================================= R: real frozen frontier judges (experiment 46)
E46 = one("46_*")
sm = json.loads((E46 / "results" / "summary.json").read_text())["panels"]["frontier_full|raw"]
smc = json.loads((E46 / "results" / "summary.json").read_text())["panels"]["frontier_full|calibrated"]
sm20 = json.loads((E46 / "results_B20000" / "summary.json").read_text())["panels"]["frontier_full|raw"]
check("R1", sm["n_pairs_x_sets"] == 50 and sm["significant_reversals"] == {"pearson": 0, "spearman": 2, "top1": 0, "kappa": 0},
      f"50 pair-by-dataset comparisons of five frontier judges: significant reversals against the proper score {sm['significant_reversals']} (Pearson 0, Spearman 2)", ["50"])
smL = json.loads((E46 / "results" / "summary.json").read_text())["panels"]["local_plus_frontier_subset|raw"]
check("R1b", smL["n_pairs_x_sets"] == 140 and all(v == 0 for v in smL["significant_reversals"].values()) and len(smL["judges"]) == 8
      and abs(140 * 2 / 4001 - 0.07) < 5e-4 and 140 * 2 / 4001 > 0.05 and 50 * 2 / 4001 < 0.05,
      f"frozen H1 (a significant Pearson reversal in the frontier or local panel): not supported; the panel with three small open-weight judge configurations added has {smL['n_pairs_x_sets']} comparisons and no significant reversal for any metric; "
      f"at the frozen 4,000 draws the smallest two-sided Holm-adjusted p over 140 comparisons is 140 x 2/4001 = {140 * 2 / 4001:.3f} > 0.05 (over the frontier panel's 50 comparisons {50 * 2 / 4001:.4f} < 0.05), so that panel's zero is uninformative", ["140", "0.07", "4,000"])
check("R2", smc["significant_reversals"] == {"pearson": 0, "spearman": 0, "top1": 0, "kappa": 0}, f"after per-judge calibration: {smc['significant_reversals']}")
check("R3", sm20["significant_reversals"] == sm["significant_reversals"], f"five-fold larger resample (B = 20,000) keeps the same counts: {sm20['significant_reversals']}", ["20,000"])
pw = rd(E46 / "results" / "pairwise.csv")
row = {s: next(r for r in pw if r["panel"] == "frontier_full|raw" and r["set"] == s and r["A"] == "gpt-6-astra" and r["B"] == "glm-5.3") for s in ("chaosnli_snli", "chaosnli_mnli")}
f3 = lambda x: f"{abs(float(x)):.3f}"
ok = (float(row["chaosnli_snli"]["d_spearman"]) > 0 and float(row["chaosnli_mnli"]["d_spearman"]) > 0
      and float(row["chaosnli_snli"]["d_skill"]) < 0 and float(row["chaosnli_mnli"]["d_skill"]) < 0
      and float(row["chaosnli_snli"]["holm_spearman"]) < 0.05 and float(row["chaosnli_mnli"]["holm_spearman"]) < 0.05
      and float(row["chaosnli_snli"]["holm_skill"]) < 0.05 and float(row["chaosnli_mnli"]["holm_skill"]) < 0.05
      and f3(row["chaosnli_snli"]["d_spearman"]) == "0.021" and f3(row["chaosnli_mnli"]["d_spearman"]) == "0.034"
      and f3(row["chaosnli_snli"]["d_skill"]) == "0.039" and f3(row["chaosnli_mnli"]["d_skill"]) == "0.098")
check("R4", ok, "GPT-6 Astra minus GLM 5.3: Spearman +0.021 (SNLI) and +0.034 (MNLI) favor Astra; Brier skill -0.039 and -0.098 favor GLM 5.3; all four Holm p < 0.05 "
      f"(Holm p {float(row['chaosnli_snli']['holm_spearman']):.3f} and {float(row['chaosnli_mnli']['holm_spearman']):.3f} for Spearman)", ["0.021", "0.034", "0.039", "0.098", "0.025", "0.042"])
larger = {s_: max(float(row[s_]["holm_spearman"]), float(row[s_]["holm_skill"])) for s_ in ("chaosnli_snli", "chaosnli_mnli")}
check("R4d", f"{larger['chaosnli_snli']:.3f}" == "0.025" and f"{larger['chaosnli_mnli']:.3f}" == "0.042",
      f"GPT-6 Astra vs GLM 5.3: the larger of the two Holm-adjusted p-values (the Spearman difference and the Brier-skill difference) is {larger['chaosnli_snli']:.3f} on SNLI and {larger['chaosnli_mnli']:.3f} on MNLI")
pw_raw = [r for r in pw if r["panel"] == "frontier_full|raw"]
disc = {m_: sum(1 for r in pw_raw if float(r["d_" + m_]) * float(r["d_skill"]) < 0) for m_ in ("pearson", "spearman", "top1", "kappa")}
rv = rd(E46 / "results" / "reversals.csv")
n_rev = {m_: sum(1 for r in rv if r["panel"] == "frontier_full|raw" and r["versus"] == "primary_S" and r["metric"] == m_) for m_ in ("pearson", "spearman", "top1", "kappa")}
check("R4b", len(pw_raw) == 50 and disc == {"pearson": 6, "spearman": 10, "top1": 12, "kappa": 12} and n_rev == disc,
      f"of the 50 comparisons, the point estimates order the two judges the opposite way from the proper score in {disc} cases (Pearson 6, Spearman 10, top-1 12, kappa 12); "
      "the reversals table lists exactly these; only the 2 Spearman cases are significant after Holm", ["6", "10", "12"])
# Sensitivity run outside the frozen protocol: B = 20,000 draws instead of the frozen 4,000.
smL20 = json.loads((E46 / "results_B20000" / "summary.json").read_text())["panels"]["local_plus_frontier_subset|raw"]
rv20 = rd(E46 / "results_B20000" / "reversals.csv")
LF = "local_plus_frontier_subset|raw"
sig20 = [r for r in rv20 if r["panel"] == LF and r["versus"] == "primary_S" and r["metric"] == "pearson" and r["significant"] == "True"]
row4 = [r for r in rv if r["panel"] == LF and r["versus"] == "primary_S" and r["metric"] == "pearson" and r["set"] == "chaosnli_mnli"
        and r["A"] == "qwen2.5-0.5b|native" and r["B"] == "qwen2.5-0.5b|verbalized"]
check("R3b", smL20["n_pairs_x_sets"] == 140 and smL20["significant_reversals"]["pearson"] == 1 and len(sig20) == 1
      and sig20[0]["set"] == "chaosnli_mnli" and sig20[0]["A"].split("|")[0] == sig20[0]["B"].split("|")[0] and sig20[0]["A"] != sig20[0]["B"]
      and f"{max(float(sig20[0]['holm_metric']), float(sig20[0]['holm_S'])):.3f}" == "0.040"
      and len(row4) == 1 and max(float(row4[0]["holm_metric"]), float(row4[0]["holm_S"])) >= 0.05
      and sm20["significant_reversals"]["pearson"] == 0,
      "sensitivity run at B = 20,000 (not the frozen 4,000): the 140-comparison panel has 1 significant Pearson reversal, on MNLI between the native and verbalized readouts of the same small "
      f"open-weight model (Holm p {max(float(sig20[0]['holm_metric']), float(sig20[0]['holm_S'])):.3f}; at the frozen B = 4,000 the same comparison has Holm p "
      f"{max(float(row4[0]['holm_metric']), float(row4[0]['holm_S'])):.3f}, not significant); the five frontier judges alone still have 0 Pearson reversals", ["0.040"])
sk = json.loads((E46 / "results" / "judge_skill_ci.json").read_text())
d350 = {e["judge"]: e for e in sk if e["set"] == "dices350"}
below = sorted(j for j, e in d350.items() if e["skill_raw_ci"][1] < 0)
top1 = [e["top1"] for e in d350.values()]
raw_below = [d350[j]["skill_raw"] for j in below]
check("R5", len(d350) == 5 and len(below) == 4 and "claude-opus-5-5" not in below and d350["claude-opus-5-5"]["skill_raw_ci"][0] > 0
      and f"{min(raw_below):.3f}" == "-0.177" and f"{max(raw_below):.3f}" == "-0.096" and f"{d350['claude-opus-5-5']['skill_raw']:.3f}" == "0.033",
      f"DICES-350, raw judges: four of five have skill below the human base rate with every 95% interval below 0 (range {min(raw_below):.3f} to {max(raw_below):.3f}); "
      f"Claude Opus 5.5 is above (+{d350['claude-opus-5-5']['skill_raw']:.3f}, interval above 0)", ["0.177", "0.096", "0.033"])
pr = [e["pearson"] for e in d350.values()]
check("R5b", f"{min(pr):.2f}" == "0.52" and f"{max(pr):.2f}" == "0.61",
      f"DICES-350: Pearson correlation of the judge's mean rating with the human mean rating, five judges: {min(pr):.3f} to {max(pr):.3f}", ["0.52", "0.61"])
js = rd(E46 / "results" / "judge_scores.csv")
inv = {r["judge"]: float(r["invalid_frac"]) for r in js if r["panel"] == "frontier_full|raw" and r["set"] == "dices350"}
vo = {r["judge"]: 1 - float(r["S_primary"]) / float(r["base_S_primary"]) for r in js if r["panel"] == "frontier_full_valid_only|raw" and r["set"] == "dices350"}
vo_below = sorted(j for j, v_ in vo.items() if v_ < 0)
check("R5c", max(inv.values()) == 0.06 and vo_below == sorted(below) and vo["claude-opus-5-5"] > 0,
      f"DICES-350: invalid answers (counted as a uniform forecast) are {min(inv.values()) * 100:.0f} to {max(inv.values()) * 100:.0f}% of items per judge; on the items where every judge answered validly the same four judges are below the base rate (point estimates) and Opus above", ["0", "6"])
check("R6", f"{min(top1) * 100:.0f}" == "80" and f"{max(top1) * 100:.0f}" == "82",
      f"DICES-350 top-1 agreement with the human majority rating of the five judges: {min(top1):.4f} to {max(top1):.4f}", ["80", "82"])
FRSETS = ("chaosnli_snli", "chaosnli_mnli", "chaosnli_alphanli", "dices350", "dices990")
cells = [e for e in sk if e["set"] in FRSETS]
side = lambda e: "below" if e["skill_raw_ci"][1] < 0 else ("above" if e["skill_raw_ci"][0] > 0 else "ns")
by_set = {st: Counter(side(e) for e in cells if e["set"] == st) for st in FRSETS}
nli_cells = [e for e in cells if e["set"].startswith("chaosnli")]
tot_side = Counter(side(e) for e in cells)
ns_nli = [(e["set"], e["judge"]) for e in nli_cells if side(e) == "ns"]
check("R3c", len(cells) == 25 and sum(by_set[st]["below"] for st in FRSETS[:3]) == 0 and sum(by_set[st]["above"] for st in FRSETS[:3]) == 14
      and ns_nli == [("chaosnli_mnli", "deepseek-v4.1-flash")] and all(e["skill_raw"] > 0 for e in nli_cells)
      and by_set["dices350"]["below"] == 4 and by_set["dices990"]["below"] == 3 and (tot_side["below"], tot_side["above"], tot_side["ns"]) == (7, 16, 2),
      f"experiment 46, raw skill over the human base rate in the 25 judge-set cells (5 judges x 5 sets): ChaosNLI (15 cells) {sum(by_set[st]['below'] for st in FRSETS[:3])} significantly below, "
      f"{sum(by_set[st]['above'] for st in FRSETS[:3])} significantly above, {len(ns_nli)} not significant ({ns_nli[0][1]} on MNLI), and every ChaosNLI point estimate above 0; DICES-350 {by_set['dices350']['below']} below; "
      f"DICES-990 {by_set['dices990']['below']} below; all 25 cells {dict(tot_side)}", ["7", "25", "14", "15"])
kap = [e["kappa"] for e in d350.values()]
check("R4c", len(kap) == 5 and all(k_ > 0 for k_ in kap) and f"{min(kap):.2f}" == "0.26" and f"{max(kap):.2f}" == "0.40",
      f"DICES-350: Cohen's kappa between each judge's most likely label and the human modal label, five judges: {min(kap):.3f} to {max(kap):.3f}, all above 0", ["0.26", "0.40"])
e28 = rd(one("28_*") / "results" / "scores_absolute.csv")
hist350 = sorted({float(r["top1"]) for r in e28 if r["dataset"] == "dices350" and r["method"] == "hist" and int(r["B"]) >= 256})
check("K1", len(hist350) == 1 and abs(hist350[0] - 269 / 350) < 5e-7 and 0.03 < min(top1) - 269 / 350 and max(top1) - 269 / 350 < 0.06,
      f"DICES-350: a constant forecast (the histogram of other items' ratings, experiment 28 method hist at 256 or 320 training items, where it has settled; ties between human modal labels count as wrong) has top-1 agreement {hist350[0]:.4f} = 269/350; "
      f"the five judges are {(min(top1) - 269 / 350) * 100:.1f} to {(max(top1) - 269 / 350) * 100:.1f} points above it (that No is the unique human modal label on 269 of 350 items is checked from the public DICES-350 file by check_always_no_dices350.py)", ["77"])
cal = [e["skill_calibrated"] for e in d350.values()]
calci = [e["skill_calibrated_ci"][0] for e in d350.values()]
check("R7", all(c > 0 for c in calci) and f"{min(cal):.3f}" == "0.053" and f"{max(cal):.3f}" == "0.071",
      f"DICES-350 after a cross-fitted per-judge calibration: all five above the base rate, +{min(cal):.3f} to +{max(cal):.3f}, every interval above 0", ["0.053", "0.071"])
d350p = [r for r in pw if r["panel"] == "frontier_full|raw" and r["set"] == "dices350"]
ao = next(r for r in d350p if r["A"] == "gpt-6-astra" and r["B"] == "claude-opus-5-5")
check("R8", abs(float(ao["d_top1"]) - 0.00095) < 5e-5 and float(ao["lo_top1"]) < 0 < float(ao["hi_top1"])
      and f"{float(ao['lo_top1']):.3f}" == "-0.022" and f"{float(ao['hi_top1']):.3f}" == "0.023"
      and f"{float(ao['d_skill']):.3f}" == "-0.175" and f"{float(ao['lo_skill']):.3f}" == "-0.200" and f"{float(ao['hi_skill']):.3f}" == "-0.152",
      f"DICES-350, GPT-6 Astra minus Claude Opus 5.5: top-1 agreement {float(ao['d_top1']):+.5f} [{float(ao['lo_top1']):+.3f}, {float(ao['hi_top1']):+.3f}]; "
      f"normalized-RPS skill {float(ao['d_skill']):+.3f} [{float(ao['lo_skill']):+.3f}, {float(ao['hi_skill']):+.3f}]", ["0.001", "0.022", "0.023", "0.175", "0.200", "0.152"])
n_top1 = sum(float(r["holm_top1"]) < 0.05 for r in d350p)
holm_skill = sorted(float(r["holm_skill"]) for r in d350p)
n_skill = sum(h < 0.05 for h in holm_skill)
n_skill_025 = sum(abs(h - 0.025) < 1e-3 for h in holm_skill)
check("R9", len(d350p) == 10 and n_top1 == 0 and n_skill == 8 and n_skill_025 == 7 and all(r["d_skill"] == r["d_skill_nrps"] for r in d350p) and abs(50 * 2 / 4001 - 0.025) < 5e-4,
      f"DICES-350, ten judge pairs: {n_top1} differ significantly in top-1 agreement; {n_skill} differ in skill at Holm p < 0.05 ({n_skill_025} at p = 0.025, one at {max(h for h in holm_skill if h < 0.05):.5f}); "
      f"0.025 is the smallest Holm-adjusted p that 4,000 draws allow over 50 comparisons (50 x 2/4001 = {50 * 2 / 4001:.5f})", ["10", "4,000"])

prot = (E46 / "PROTOCOL.md").read_text()
def cell_row(name):
    for l in prot.splitlines():
        if l.startswith(f"| {name} |"):
            return [c.strip() for c in l.strip().strip("|").split("|")]
    raise AssertionError(name)
c_snli, c_mnli, c_alpha, d350_row, d990_row = (cell_row(n) for n in ("ChaosNLI SNLI", "ChaosNLI MNLI-m", "ChaosNLI alphaNLI", "DICES-350", "DICES-990"))
check("D1", c_snli[3] == "100" and c_mnli[3] == "100" and c_alpha[3] == "100" and d350_row[1] == "350" and d350_row[3] == "123" and d990_row[3] == "69-76" and d990_row[1] == "400"
      and "No, Unsure, Yes" in d350_row[2],
      "datasets: ChaosNLI SNLI, MNLI-m and alphaNLI with 100 human labels per item; DICES-350 (350 conversations, 123 raters each, No/Unsure/Yes) and DICES-990 (400 items, 69 to 76 raters per item, quoted in the post as about 70)",
      ["100", "350", "123", "69", "76", "70", "400"])

# ======================================================================= X: robustness to the choice of proper score (experiment 55)
E55 = one("55_*")
s55 = json.loads((E55 / "results" / "summary.json").read_text())
check("X1", s55["h1_dices350_n_significantly_below_base_by_rule"] == {"brier": 4, "ce": 4, "nrps": 4} and s55["h2_cells_holding"] == 15 and s55["h2_cells_total"] == 15,
      f"experiment 55: DICES-350 judges significantly below the base rate under Brier, cross-entropy and nRPS: {s55['h1_dices350_n_significantly_below_base_by_rule']}; best judge's skill in the highest-agreement fifth exceeds its lowest-agreement fifth in {s55['h2_cells_holding']} of {s55['h2_cells_total']} set-by-rule cells", ["15"])
ra = rd(E55 / "results" / "rank_agreement.csv")
taus = [float(r["kendall_tau_b"]) for r in ra]
chg = sorted({r["dataset"] for r in ra if r["same_best"] == "False"})
check("X2", min(taus) == 0.6 and max(taus) == 1.0 and chg == ["chaosnli_alphanli"],
      f"ranking of the five judges by skill agrees across rules: Kendall tau-b {min(taus):.1f} to {max(taus):.1f}; the best judge changes with the rule only on {chg}", ["0.6", "1.0"])
sa = rd(E55 / "results" / "skill_all_rules.csv")
p55 = (one("55_*") / "PROTOCOL.md").read_text()
check("X1b", "half A (defines agreement through its Shannon entropy in nats)" in p55 and "five equal-count entropy quintiles within each set" in p55
      and "What the analyst had seen before freezing" in p55 and "4 of 5 judges below the base rate on DICES-350" in p55,
      "experiment 55 protocol: strata are five equal-count entropy quintiles of the ratings in one random half of the raters, skill is scored on the other half; "
      "the protocol lists what had been seen before it was frozen (including the matched-rule DICES-350 result)", [])
check("X1c", len(sa) == 75 and len({(r["dataset"], r["rule"], r["judge"]) for r in sa}) == 75 and "all 75 declared cells" in p55,
      "experiment 55: 75 judge-by-set-by-rule cells (5 judges x 5 sets x 3 rules); the protocol applies Holm-Bonferroni over all 75 declared cells", ["75"])
sig_up = {r_: sum(1 for r in sa if r["dataset"] == "chaosnli_mnli" and r["rule"] == r_ and r["significant_above_base"] == "True") for r_ in ("brier", "ce", "nrps")}
check("X3", sig_up == {"brier": 3, "ce": 2, "nrps": 5},
      f"MNLI-m, judges significantly above the base rate (of 5): Brier {sig_up['brier']}, cross-entropy {sig_up['ce']}, normalized RPS (not matched to unordered labels) {sig_up['nrps']}", ["5"])
mt = {(r["dataset"], r["rule"]): r["match"] for r in sa}
h2all = rd(E55 / "results" / "h2_cells.csv")
matched_cells = [r for r in h2all if mt[(r["dataset"], r["rule"])].startswith("matched")]
check("X6", len(h2all) == 15 and len(matched_cells) == 9 and all(r["holds"] == "True" for r in h2all),
      f"experiment 55: the best judge scores higher in the highest-agreement fifth than in the lowest in all {len(h2all)} set-by-rule cells, {len(matched_cells)} of them with a rule that the experiment's own table labels matched to the label encoding", ["9"])
h2 = [r for r in rd(E55 / "results" / "h2_cells.csv") if r["dataset"] == "dices350" and r["rule"] == "nrps"][0]
check("X4", h2["best_judge"] == "claude-opus-5-5" and f"{float(h2['skill_Q1']):.2f}" == "0.25" and f"{float(h2['skill_Q5']):.2f}" == "-0.07",
      f"DICES-350, normalized RPS, best judge ({h2['best_judge']}): skill {float(h2['skill_Q1']):+.3f} in the highest-agreement fifth of items, {float(h2['skill_Q5']):+.3f} in the lowest", ["0.25", "0.07", "0.254", "0.072"])

# ======================================================================= X5: the label cost of calibrating the frontier judges (experiment 56)
E56 = one("56_*")
rows56 = json.loads((E56 / "results" / "label_budget_frontier.json").read_text())["rows"]
p56 = (E56 / "PROTOCOL.md").read_text()
dc56 = [r for r in rows56 if r["set"] == "dices350"]
at25 = [r["SS_n"]["25"] for r in dc56]
at100 = [r["SS_n"]["100"] for r in dc56]
sets56 = sorted({r["set"] for r in rows56})
all_pos = {n_: sum(1 for st in sets56 if all(r["SS_n"][n_] > 0 for r in rows56 if r["set"] == st)) for n_ in ("10", "25", "50", "100", "200")}
check("X5", len(rows56) == 25 and len(sets56) == 5 and len(dc56) == 5 and all(v < 0 for v in at25) and all(v > 0 for v in at100)
      and all_pos == {"10": 1, "25": 2, "50": 4, "100": 5, "200": 5}
      and "For each case, 50 draws" in p56 and "one** rating per item" in p56 and "n in {10, 25, 50, 100, 200}" in p56,
      f"experiment 56 (calibrating on n held-out ratings, one per item; mean over 50 draws; budgets 10, 25, 50, 100, 200): on DICES-350 every judge's calibrated skill is negative at n = 25 ({min(at25):+.3f} to {max(at25):+.3f}) "
      f"and positive at n = 100 ({min(at100):+.3f} to {max(at100):+.3f}); sets (of 5) on which every judge's calibrated skill is above 0: {all_pos}, so 100 is the smallest budget tried at which it holds on all five", ["25", "100", "50"])

# ======================================================================= M: power (experiment 44)
am = rd(one("44_*") / "results" / "arm_m_pairs.csv")
nli = [float(r["mdd_N400_k1"]) for r in am if r["dataset"].startswith("chaosnli")]
dic = [float(r["mdd_N400_k1"]) for r in am if r["dataset"].startswith("dices")]
res = sum(r["resolvable_N400_k1"] == "True" for r in am)
check("M1", len(am) == 50 and f"{min(nli):.3f}" == "0.034" and f"{max(nli):.3f}" == "0.065" and f"{min(dic):.3f}" == "0.014" and f"{max(dic):.3f}" == "0.025" and res == 18,
      f"minimum detectable difference between two judges at 400 items with one rating each (two-sided 5%, 80% power): Brier {min(nli):.3f} to {max(nli):.3f} (ChaosNLI), normalized RPS {min(dic):.3f} to {max(dic):.3f} (DICES); {res} of {len(am)} judge pairs resolvable at that size",
      ["0.034", "0.065", "0.014", "0.025", "400", "18"])
kn = (one("44_*") / "results" / "key_numbers.md").read_text()
check("M2", "cells with Holm p < 0.05: 0 of 30" in kn and "Holm-significant interaction cells 0 of 60" in kn,
      "rater-pool contrasts: 0 of 30 pool score differences and 0 of 60 pool-by-judge interaction cells Holm-significant (a power-limited null, not evidence of absence)", ["30", "60"])

# ======================================================================= T2: training and calibration (experiments 41, 47, 48, 53)
tb = [l for l in (one("41_*") / "results" / "k_sweep_summary_table.md").read_text().splitlines() if l.startswith("| ") and "carries signal" not in l and not l.startswith("|---")]
rawgap = [float(l.split("|")[11].strip()) for l in tb]
matched = [float(l.split("|")[12].strip().split(" ")[0]) for l in tb]
check("C1", f"{max(rawgap):.4f}" == "0.1282" and f"{max(matched):.4f}" == "0.0024",
      f"experiment 41 ({len(tb)} view-by-K cells): raw Pearson-trained minus CE-trained test nRPS gap up to {max(rawgap):.4f}; after one validation-fitted map per head at most {max(matched):.4f} (seed mean)", ["0.1282", "0.0024"])
r41 = (one("41_*") / "README.md").read_text()
check("C1c", "seven real multi-annotator rating sets re-binned to two, three and five classes" in r41 and "Seven datasets, 16 views" in r41
      and "12 of 16 in both runs" in r41 and "all five split seeds" in r41 and "linear on 24 hashed-text components" in r41
      and "the largest seed mean is 0.0024" in r41 and "validation-fitted calibration map" in r41,
      "experiment 41: seven public rating sets, 16 views (two, three or five rating levels), linear heads on 24 hashed-text components, five split seeds; a validation-fitted map per head; "
      "matched gap at most 0.0024 (seed mean); the cross-entropy-trained head beats the constant forecast on 12 of 16 views", ["7", "24", "12", "5"])
pq = next(l for l in tb if l.split("|")[1].strip() == "popquorn_offensiveness" and l.split("|")[2].strip() == "2").split("|")
gap_pq = float(pq[7]) - float(pq[5])
check("C1b", (pq[4].strip(), pq[5].strip(), pq[7].strip()) == ("0.1156", "0.1153", "0.2430") and f"{gap_pq:.3f}" == "0.128",
      f"experiment 41, POPQUORN offensiveness at two levels: test nRPS of a constant forecast {pq[4].strip()}, CE-trained head {pq[5].strip()}, raw Pearson-trained head {pq[7].strip()}, raw gap {gap_pq:.4f}", ["0.116", "0.115", "0.243", "0.128"])
n_unord = sum(l.split("|")[1].strip() == "chaosnli_snli_mnli" for l in tb)
tbl41 = (one("41_*") / "results" / "k_sweep_summary_table.md").read_text()
check("C1d", len(tb) == 16 and n_unord == 1 and "Test normalized ranked probability score" in tbl41 and "ridge penalty fixed at 0.01 for every head" in tbl41,
      f"experiment 41: all {len(tb)} views are scored by test normalized RPS with one ridge penalty (0.01) for every head; exactly {n_unord} view (ChaosNLI SNLI+MNLI, three unordered labels) has no label order", ["16"])
rr = json.loads((one("53_*") / "results" / "skill_reanalysis.json").read_text())["rows"]
neg = sum(v["skill_pearson_raw"]["mean_over_seeds"] < 0 for v in rr.values())
check("C2", len(rr) == 16 and neg == 15, f"experiment 53: the raw Pearson-trained head's skill over the base rate is below 0 on {neg} of {len(rr)} views", ["15", "16"])
ce_raw = [v["skill_ce_raw"]["mean_over_seeds"] for v in rr.values()]
n_ce_le0 = sum(x <= 0 for x in ce_raw)
check("C2b", len(ce_raw) == 16 and n_ce_le0 == 4 and f"{max(ce_raw):.3f}" == "0.071",
      f"experiment 53: the raw cross-entropy-trained head's skill over the base rate is at most 0 on {n_ce_le0} of 16 views and peaks at {max(ce_raw):.4f} (range {min(ce_raw):.4f} to {max(ce_raw):.4f})", ["4", "0.071"])
s47 = [e for e in json.loads((one("47_*") / "results" / "label_budget_summary.json").read_text()) if e["design"] == "R1"]
g25 = [e["cells"]["25"]["G_mean"] for e in s47]
nstar = []
for e in s47:
    for key in ("25", "50", "100", "200", "500", "1000", "all"):
        if e["cells"][key]["G_mean"] <= 0.0024:
            nstar.append(e["cells"][key]["n_used"])
            break
check("C3", len(s47) == 16 and f"{min(g25):.4f}" == "0.0097" and f"{max(g25):.4f}" == "0.0225" and min(g25) > 0.0024,
      f"experiment 47: with 25 calibration ratings the calibrated Pearson-trained head is still worse than the CE-trained head on all 16 views by {min(g25):.4f} to {max(g25):.4f} nRPS", ["0.0097", "0.0225", "25"])
n100_200 = sum(100 <= n <= 200 for n in nstar)
check("C4", len(nstar) == 16 and n100_200 == 15 and sorted(nstar)[0] == 50,
      f"experiment 47: the gap falls to 0.0024 or less after {sorted(nstar)[0]} ratings in one view and after 100 to 200 in {n100_200} of 16", ["100", "200", "50"])
sh = json.loads((one("48_*") / "results" / "shift_summary.json").read_text())["arm_a_by_shift"]
n82 = sum(v["n"] for v in sh.values())
reo = sum(v["reopens"] for v in sh.values())
fl = sum(v["fails"] for v in sh.values())
check("C5", (n82, reo, fl) == (82, 53, 33),
      f"experiment 48: a calibration map fitted on one set or rubric and applied to another reopens the proper-score gap in {reo} of {n82} judge-by-shift cases and leaves the judge significantly worse than uncalibrated in {fl}", ["53", "82", "33"])
arm_a = json.loads((one("48_*") / "results" / "arm_a_llm_judges.json").read_text())
frontier5 = {"gpt-6-astra", "claude-opus-5-5", "kimi-k3", "glm-5.3", "deepseek-v4.1-flash"}
isd = lambda name: name.startswith("dices")
cs = lambda f: [c for c in arm_a if f(c)]
dd_cases = cs(lambda c: c["shift"] == "near" and isd(c["source"]) and isd(c["target"]))
far_cases = cs(lambda c: c["shift"] == "far")
to_nli = [c for c in far_cases if isd(c["source"]) and not isd(c["target"])]
to_dices = [c for c in far_cases if not isd(c["source"]) and isd(c["target"])]
f_ = lambda cases: sum(bool(c["calibration_fails_outright"]) for c in cases)
fr_cases = cs(lambda c: c["judge"] in frontier5)
fr_reo, fr_fail = sum(bool(c["reopens"]) for c in fr_cases), f_(fr_cases)
# every frontier case moves a map to a different rating set: a single set (near, far) or the pooled other three sets (leave one set out); the two rubric cases are G-Eval
fr_shift = sorted({c["shift"] for c in fr_cases})
loso_cases = [c for c in fr_cases if c["shift"] == "leave_one_set_out"]
check("C5b", len(arm_a) == 82 and len(fr_cases) == 80 and (fr_reo, fr_fail) == (52, 33) and fr_shift == ["far", "leave_one_set_out", "near"]
      and len(loso_cases) == 20 and all(c["source"] == "pooled_other_three" for c in loso_cases)
      and sorted({c["shift"] for c in arm_a if c["judge"] not in frontier5}) == ["rubric"] and len(dd_cases) == 10 and sum(bool(c["reopens"]) for c in dd_cases) == 0 and f_(dd_cases) == 0
      and len(far_cases) == 40 and f_(far_cases) == 21 and 2 * f_(far_cases) > fl and len(to_nli) == 20 and f_(to_nli) == 19 and len(to_dices) == 20 and f_(to_dices) == 2,
      f"experiment 48: the 82 cases are {len(fr_cases)} frontier-judge cases plus 2 G-Eval rubric cases; the {len(fr_cases)} frontier-judge cases (maps fitted on one rating set, or on the pooled other three, and applied to a new set) reopen the gap in {fr_reo} and fail outright in {fr_fail}; the 10 transfers between DICES-350 and DICES-990 never reopen the gap; "
      f"{f_(far_cases)} of the {fl} outright failures are transfers between the inference sets and the safety sets (DICES to NLI {f_(to_nli)} of {len(to_nli)}, NLI to DICES {f_(to_dices)} of {len(to_dices)})", ["52", "80"])

s49 = json.loads((one("49_*") / "results" / "system_level_drop.json").read_text())
rep = s49["reproduction"]["this_run_exp23_function"]
dd = s49["delta_pearson_minus_calibrated"]
check("C6", s49["n_systems"] == 16 and f"{rep['pearson']:.3f}" == "0.877" and f"{rep['calibrated_pearson']:.3f}" == "0.835" and f"{rep['continuous_calibrated_pearson']:.3f}" == "0.875"
      and dd["documents"]["lo"] < 0 < dd["documents"]["hi"] and f"{dd['documents']['lo']:.2f}" == "-0.05" and f"{dd['documents']['hi']:.2f}" == "0.11",
      f"experiment 49, SummEval, {s49['n_systems']} systems: system-level Spearman {rep['pearson']:.3f} (Pearson-trained head) vs {rep['calibrated_pearson']:.3f} (four-bin calibrated) vs {rep['continuous_calibrated_pearson']:.3f} (continuous calibrator); "
      f"difference {dd['mean']:+.3f}, 95% interval over test documents [{dd['documents']['lo']:+.3f}, {dd['documents']['hi']:+.3f}] (includes 0)", ["0.877", "0.835", "0.875", "0.05", "0.11", "0.049", "0.114", "0.042", "16"])
r49 = (one("49_*") / "README.md").read_text()
check("C6b", "G-Eval GPT-4" in r49 and "960 training summaries" in r49 and "640 test summaries (40 documents x 16 systems)" in r49 and "four-bin calibrator" in r49 and "[-0.049, +0.114]" in r49,
      "experiment 49 setup: SummEval consistency, G-Eval with GPT-4 as the judge, 40 test documents by 16 systems; the drop comes from the four-bin calibrator; interval over documents [-0.049, +0.114]", ["40"])

# ======================================================================= S: scaling hypothesis (experiment 31) via experiment 41's verifier
v = subprocess.run([sys.executable, "-I", str(one("41_*") / "scripts" / "verify_blog_numbers.py"), str(ROOT)], capture_output=True, text=True)
need = ["[B2]", "[B3]", "[B4]", "[B5]", "[B8]", "[B9]", "[B11]", "[B12]"]
have = {t: any(l.startswith("PASS") and t in l for l in v.stdout.splitlines()) for t in need}
check("S1", v.returncode == 0 and all(have.values()),
      f"experiment 41 verifier exit {v.returncode}; required PASS lines {have} (CE-trained curve fits in 9 of 11 panels, Pearson-trained in 1 of 11; 6 of 11 when counting only fits with more than four budgets; three of the nine are four-point fits; largest budgets 256, 1,024, 2,048; CE head pulls ahead after about 4 to 8 ratings)",
      ["9", "11", "1", "6", "256", "1,024", "2,048", "4", "8", "0.66", "0.82"])
for l in v.stdout.splitlines():
    if any(t in l for t in need):
        print("   |", l[:250])

# ======================================================================= L: cited literature number (cited, not machine-checkable here)
cl = (ROOT / "CLAUDE.md").read_text()
check("L1", "arXiv 2503.05965" in cl and '"as much as 31%" worse judges' in cl,
      "cited (Guerdan et al., arXiv 2503.05965, abstract; recorded in the repo's literature note, not recomputed here): forced-choice validation can select judges up to 31% worse than their response-set method", ["31"])

LEAN = one("35_*") / "extended_checks_11-26-2026" / "gemini_plan_claims_audit_11-26-2026"
lc = (LEAN / "lean_check.txt").read_text()
lsrc = (LEAN / "PlanClaims.lean").read_text()
std3 = "[propext, Classical.choice, Quot.sound]"
lean_thms = ["plan_estimator_unbiased", "plan_estimator_unbiased_product", "instance_unbiased", "adaptive_stopping_biased", "Model.anti_proper"]
check("L2", all(f"'PlanAudit.{t}' depends on axioms: {std3}" in lc for t in lean_thms) and "sorryAx" not in lc
      and all(re.search(rf"^theorem {re.escape(t.split('.')[-1])}\b", lsrc, flags=re.M) for t in lean_thms) and "Lean 4.36.0" in cl and "fc4f59e5" in cl,
      "Lean 4 with Mathlib (PlanClaims.lean, experiment 35, plan-claims audit; toolchain Lean 4.36.0, Mathlib fc4f59e5): the per-sample proper score is an unbiased estimator of the risk for finite rating scales "
      "(plan_estimator_unbiased, plan_estimator_unbiased_product, instance_unbiased), the number of ratings must not depend on the ratings (adaptive_stopping_biased), and the collapsed judge's correlation sqrt(rho) exceeds the truthful judge's rho "
      "for finite joint laws (Model.anti_proper); the compiler output lists only propext, Classical.choice, Quot.sound and no sorryAx", [])

# ======================================================================= registry
print()
print("REGISTRY of numeric tokens (unsigned) the post may use, with the claims that support them:")
for t in sorted(REGISTRY, key=lambda s: (len(s), s)):
    print(f"  {t:>8}  {','.join(REGISTRY[t])}")
print()
print(f"{NCHECK[0]} claims checked, {len(FAILED)} FAIL; seed {SEED}")
if FAILED:
    print("FAILED:", ", ".join(FAILED))
sys.exit(1 if FAILED else 0)
