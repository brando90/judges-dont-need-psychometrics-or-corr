#!/usr/bin/env python3
"""Compare Experiment 36's trained heads (commit fcd8db9) with its own train-marginal constant forecast.

Claude, 11-26-2026. Read-only. It reads only `results.json` and tabulates, for each of the 16 matrices, the held-out test
metrics of the cross-entropy-trained head, the Pearson-trained head and the constant forecast that predicts the training
marginal for every prompt (`marginal` in the file; `uniform` is the flat forecast). It checks the claim in SANMI_PITCH.md
that the cross-entropy-trained head beats the Pearson-trained head on held-out CE, sqrt(RPS) and Brier on 9 of 9 ordinal
and 3-class views, and reports three things the pitch does not say:
  (1) how far the cross-entropy-trained head is below the constant forecast;
  (2) in how many matrices the Pearson-trained head has a higher held-out Pearson r than the cross-entropy-trained head
      (a head trained to maximize correlation should win that metric);
  (3) whether the Pearson-trained head's optimizer reported success.
Intervals for the head-minus-constant difference are not in the file (the constant has an interval only for CE and it is not
paired), so none is claimed here.

Usage:
    python3 -I heads_vs_marginal_check.py PATH/TO/results.json
Exit status is nonzero when a check fails.
"""
import json
import sys

p = json.load(open(sys.argv[1]))
m = p["matrices"]
NINE = ["summeval_expert", "chaosnli_snli_mnli", "dices990", "popquorn_offensiveness", "popquorn_politeness",
        "veribench_fsc_rv1", "veribench_fsc_rv1_expert3", "veribench_fsc_native_main", "veribench_fsc_pooled"]
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (("  " + detail) if detail else ""))
    if not ok:
        fails += 1


def f(x):
    return "None" if x is None else f"{x:.3f}"


print("FACT  columns: held-out test CE of the cross-entropy-trained head | CE of the constant (training-marginal) forecast | CE of the flat forecast | "
      "normalized RPS of the cross-entropy-trained head | of the constant forecast | CE and normalized RPS of the Pearson-trained head | "
      "held-out Pearson r of the Pearson-trained head and of the cross-entropy-trained head | Pearson-head optimizer (success, iterations)")
for name, dd in m.items():
    r = dd["rows"]
    ce, pe, mg, un = r["ce"]["point"], r["pearson"]["point"], dd["marginal"]["point"], dd["uniform"]["point"]
    op = r["pearson"]["optimizer"]
    print(f"FACT  {name:30s} train/val/test {dd['n_train']}/{dd['n_val']}/{dd['n_test']:<4d} | CE {ce['ce']:.3f} vs constant {mg['ce']:.3f} (flat {un['ce']:.3f}) | "
          f"nRPS {ce['rps_norm']:.4f} vs constant {mg['rps_norm']:.4f} | Pearson head CE {pe['ce']:.3f}, nRPS {pe['rps_norm']:.4f} | "
          f"r {f(pe['pearson_r'])} vs {f(ce['pearson_r'])} | ({op.get('success')}, {op.get('nit')})")

ok9 = []
for name in NINE:
    r = m[name]["rows"]
    a, b = r["ce"]["point"], r["pearson"]["point"]
    ok9.append(a["ce"] < b["ce"] and a["sqrt_rps"] < b["sqrt_rps"] and a["brier"] < b["brier"])
check("pitch claim: the CE-trained head beats the Pearson-trained head on held-out CE, sqrt(RPS) and Brier on 9 of 9 ordinal and 3-class views", all(ok9), f"{sum(ok9)} of 9")

n = len(m)
not_better_ce = [k for k, d_ in m.items() if d_["rows"]["ce"]["point"]["ce"] >= d_["marginal"]["point"]["ce"]]
not_better_rps = [k for k, d_ in m.items() if d_["rows"]["ce"]["point"]["rps_norm"] >= d_["marginal"]["point"]["rps_norm"]]
print(f"FACT  the cross-entropy-trained head is NOT below the constant forecast on held-out CE in {len(not_better_ce)} of {n} matrices ({', '.join(not_better_ce) or 'none'}) and on normalized RPS in {len(not_better_rps)} ({', '.join(not_better_rps) or 'none'})")
big = [k for k in m if m[k]["n_test"] >= 190 and not k.startswith("veribench")]
gap = {k: m[k]["marginal"]["point"]["ce"] - m[k]["rows"]["ce"]["point"]["ce"] for k in big}
print("FACT  large non-FSC matrices: CE gain of the cross-entropy-trained head over the constant forecast, in nats: " + ", ".join(f"{k} {v:.3f}" for k, v in gap.items()))
print(f"FACT  largest such gain {max(gap.values()):.3f} nats; the flat forecast is {m['summeval_expert']['uniform']['point']['ce']:.3f} for SummEval")
pairs = [(k, d_["rows"]["pearson"]["point"]["pearson_r"], d_["rows"]["ce"]["point"]["pearson_r"]) for k, d_ in m.items()
         if d_["rows"]["pearson"]["point"]["pearson_r"] is not None and d_["rows"]["ce"]["point"]["pearson_r"] is not None]
wins = [k for k, a, b in pairs if a > b]
print(f"FACT  the Pearson-trained head has the higher held-out Pearson r in {len(wins)} of {len(pairs)} matrices ({', '.join(wins)}); in the other {len(pairs) - len(wins)} the cross-entropy-trained head has the higher or equal r")
succ = [d_["rows"]["pearson"]["optimizer"].get("success") for d_ in m.values()]
print(f"FACT  the Pearson-head optimizer reported success in {sum(1 for s in succ if s)} of {len(succ)} matrices; iteration counts {sorted({d_['rows']['pearson']['optimizer'].get('nit') for d_ in m.values()})} (protocol cap 35)")
print("\n" + ("ALL CHECKS PASS" if fails == 0 else f"{fails} CHECK(S) FAILED"))
sys.exit(1 if fails else 0)
