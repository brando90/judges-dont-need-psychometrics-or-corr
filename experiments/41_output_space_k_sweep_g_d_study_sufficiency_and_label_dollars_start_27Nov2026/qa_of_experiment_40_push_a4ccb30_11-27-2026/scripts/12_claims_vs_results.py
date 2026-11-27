"""Check REPORT/README sentences against the committed results.json (point estimates only).

1. "once the nominal budget reaches a few hundred ratings the held-out CE ... is within a few
   thousandths of a nat across k, and the ... interval of the D-study-minus-oracle difference covers 0":
   spread of test CE across the options of each budget >= 256, and intervals that exclude 0 at B >= 256.
2. "At small budgets that k is often infeasible": count of budget rows with no D-study cell.
3. Comparison with the user's study (P7): at budgets where the k = 1 cell is binding (N = B), is the
   k = 1 head better than, tied with, or worse than the D-study k (point estimate of test CE)?
4. Dollar hits that are left-censored (equal to the smallest grid budget).
"""
import json

d = json.load(open("/tmp/e40_cursor/experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/results.json"))
print("1. spread of held-out test CE across k (max-min over the options at each budget), B >= 256:")
excl = []
for name, b in d["datasets"].items():
    cells = b["allocation"]["cells"]
    spreads = []
    for row in b["allocation"]["budgets"]:
        B = row["B"]
        if B < 256:
            continue
        opts = []
        for k in sorted({c["k"] for c in cells}):
            ks = [c for c in cells if c["k"] == k]
            nmax = max(c["N"] for c in ks)
            n = min(nmax, B // k)
            m = [c for c in ks if c["N"] == n]
            if m and n >= 4:
                opts.append(m[0]["test"]["ce"])
        spreads.append((B, max(opts) - min(opts)))
        diff = row["dstudy_minus_oracle_ce"]
        if diff and (diff["lo"] > 0 or diff["hi"] < 0):
            excl.append((name, B, round(diff["point"], 4), round(diff["lo"], 4), round(diff["hi"], 4)))
    print(f"   {name}: " + ", ".join(f"B={B}: {s:.4f}" for B, s in spreads))
print("   D-minus-oracle intervals excluding 0 at B >= 256:", excl)
rows = [r for b in d["datasets"].values() for r in b["allocation"]["budgets"]]
print(f"2. budget rows with the D-study k infeasible: {sum(1 for r in rows if r['dstudy'] is None)} of {len(rows)}")
print("3. k = 1 (binding, N = B) versus the D-study k at the same nominal budget (test CE point estimates):")
better = tie = worse = 0
for name, b in d["datasets"].items():
    cells = b["allocation"]["cells"]
    for row in b["allocation"]["budgets"]:
        B, dc = row["B"], row["dstudy"]
        one = [c for c in cells if c["k"] == 1 and c["N"] == B]
        if not one or dc is None or dc["k"] == 1:
            continue
        delta = one[0]["test"]["ce"] - dc["test"]["ce"]
        tag = "k=1 better" if delta < -1e-4 else ("tie (|diff|<=1e-4)" if abs(delta) <= 1e-4 else "D-study better")
        better += delta < -1e-4; tie += abs(delta) <= 1e-4; worse += delta > 1e-4
        print(f"   {name:26s} B={B:5d} k*={dc['k']:2d}: CE(k=1) - CE(k*) = {delta:+.4f}  {tag}")
print(f"   totals: k=1 better {better}, tie {tie}, D-study better {worse}")
print("4. dollar hits equal to the smallest grid budget (left-censored, true threshold <= 32):")
for name, b in d["datasets"].items():
    h = b["dollars"]
    first = b["allocation"]["budgets"][0]["B"]
    print(f"   {name}: B_ce={h['B_ce']} (censored={h['B_ce'] == first}); B_sqrt_rps={h['B_sqrt_rps']} (censored={h['B_sqrt_rps'] == first})")
