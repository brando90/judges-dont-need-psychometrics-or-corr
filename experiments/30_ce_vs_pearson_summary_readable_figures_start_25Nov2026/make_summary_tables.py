"""Tables for results.md (11-25-2026), computed from the repo's saved results files. Usage: python3 make_summary_tables.py [REPO_ROOT] > tables.md"""
import csv, glob, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
E26, E28, E29 = [Path(glob.glob(str(EXP / f"{n}_*"))[0]) for n in (26, 28, 29)]
rd = lambda p: list(csv.DictReader(open(p)))
SETS = {"chaosnli_snli": "ChaosNLI SNLI", "chaosnli_mnli": "ChaosNLI MNLI-m", "chaosnli_alphanli": "alphaNLI (2 classes)", "dices350": "DICES-350", "dices990": "DICES-990"}
JUDGES = {"gpt-6-astra": "GPT-6 Astra", "claude-opus-5-5": "Claude Opus 5.5", "kimi-k3": "Kimi K3", "glm-5.3": "GLM 5.3", "deepseek-v4.1-flash": "DeepSeek V4.1 Flash"}
f3 = lambda x: f"{x:.3f}"

def tab28(B):
    s = rd(E28 / "results/scores_absolute.csv"); cor = rd(E28 / "results/correlations.csv")
    print(f"| Set | RPS CE-fit | RPS Pearson-fit | RPS uniform | CE CE-fit | CE Pearson-fit | TVD CE-fit | TVD Pearson-fit | Pearson r CE-fit / Pearson-fit | Spearman CE-fit / Pearson-fit |\n|---|---|---|---|---|---|---|---|---|---|")
    for ds, nm in SETS.items():
        v = lambda m, k, src=s: np.mean([float(r[k]) for r in src if r["dataset"] == ds and r["d"] == "1" and r["B"] == str(B) and r["method"] == m])
        c = lambda m, k: np.mean([float(r[k]) for r in cor if r["dataset"] == ds and r["d"] == "1" and r["B"] == str(B) and r["method"] == m])
        print(f"| {nm} | {f3(v('ce','rps'))} | {f3(v('pearson','rps'))} | {f3(v('uniform','rps'))} | {f3(v('ce','ce'))} | {f3(v('pearson','ce'))} | {f3(v('ce','tvd'))} | {f3(v('pearson','tvd'))} | "
              f"{f3(c('ce','pearson_expected_vs_human_mean'))} / {f3(c('pearson','pearson_expected_vs_human_mean'))} | {f3(c('ce','spearman_expected_vs_human_mean'))} / {f3(c('pearson','spearman_expected_vs_human_mean'))} |")

def wins28():
    s = rd(E28 / "results/scores_absolute.csv"); idx = {}
    for r in s: idx[(r["dataset"], r["judge"], r["d"], r["B"], r["method"])] = r
    for d in ("1", "5"):
        out = {}
        for k in ("rps", "ce", "tvd"):
            n = w = 0
            for (ds, j, dd, B, m), r in idx.items():
                if dd == d and m == "ce":
                    n += 1; w += float(r[k]) < float(idx[(ds, j, dd, B, "pearson")][k])
            out[k] = f"{w}/{n}"
        print(f"- d={d} (point estimates, all budgets, all judge-set cells): CE-fit lower than Pearson-fit on RPS {out['rps']}, CE {out['ce']}, TVD {out['tvd']}")

def tab26():
    s = rd(E26 / "results/new5/scores.csv"); dec = rd(E26 / "results/new5/decisions.csv")
    g = lambda j, m, k: float([r for r in s if r["judge"] == j and r["ratings_per_answer"] == "2" and r["training_ratings"] == "256" and r["method"] == m and r["metric"] == k][0]["mean"])
    print("| Judge | RPS CE-fit | RPS Pearson-fit | CE CE-fit | CE Pearson-fit | TVD CE-fit | TVD Pearson-fit | Formal test at 256 ratings (vs Pearson-fit, lower bound) |\n|---|---|---|---|---|---|---|---|")
    for j, nm in JUDGES.items():
        d = [r for r in dec if r["judge"] == j and r["training_ratings"] == "256"][0]
        print(f"| {nm} | {g(j,'ce_fit','rps'):.4f} | {g(j,'pearson_fit','rps'):.4f} | {g(j,'ce_fit','ce'):.3f} | {g(j,'pearson_fit','ce'):.3f} | {g(j,'ce_fit','tvd'):.3f} | {g(j,'pearson_fit','tvd'):.3f} | {d['formal_status']} ({float(d['lb_vs_pearson']):+.4f}) |")
    get = lambda j, B, m, k: float([r for r in s if r["judge"] == j and r["ratings_per_answer"] == "2" and r["training_ratings"] == str(B) and r["method"] == m and r["metric"] == k][0]["mean"])
    exc = []; cnt = {}
    for k in ("rps", "ce", "tvd"):
        n = w = 0
        for j in JUDGES:
            for B in (4, 8, 16, 32, 64, 128, 256):
                n += 1; ok = get(j, B, "ce_fit", k) < get(j, B, "pearson_fit", k); w += ok
                if not ok: exc.append(f"{k.upper()} at {B} ratings for {JUDGES[j]}")
        cnt[k] = f"{w} of {n}"
    print(f"\nCE-fit has lower held-out RPS than Pearson-fit in {cnt['rps']} judge-by-budget cells, lower CE in {cnt['ce']} and lower TVD in {cnt['tvd']}. Cells where Pearson-fit is lower: " + "; ".join(exc) + ".")
    print("\nFormal status by judge and budget:\n\n| Judge | " + " | ".join(str(b) for b in (256, 128, 64, 32, 16, 8, 4)) + " |\n|---|" + "---|" * 7)
    for j, nm in JUDGES.items():
        print(f"| {nm} | " + " | ".join([r for r in dec if r["judge"] == j and r["training_ratings"] == str(b)][0]["formal_status"] for b in (256, 128, 64, 32, 16, 8, 4)) + " |")

def tab29():
    s = rd(E29 / "results/scores.csv"); con = rd(E29 / "results/contrasts.csv")
    P = [("chaos3_qwen", "1", "ChaosNLI, Qwen3-4B, d=1"), ("chaos3_qwen", "5", "ChaosNLI, Qwen3-4B, d=5"), ("alpha2_qwen", "1", "alphaNLI, Qwen3-4B"), ("dices3_qwen", "1", "DICES, Qwen3-4B"),
         ("hs2_qwen", "1", "HelpSteer2, Qwen3-4B"), ("chaos3_llama", "1", "ChaosNLI, Llama-3.1-8B"), ("dices3_llama", "1", "DICES, Llama-3.1-8B")]
    print("| Panel | Budget | RPS CE-LoRA | RPS Pearson-LoRA | CE CE-LoRA | CE Pearson-LoRA | TVD CE-LoRA | TVD Pearson-LoRA | Pearson minus CE RPS [crossed 95%] |\n|---|---|---|---|---|---|---|---|---|")
    for st, d, nm in P:
        B = max(int(r["b_total"]) for r in s if r["study"] == st and r["depth"] == d and r["method"] == "ce_raw")
        g = lambda m, k: float([r for r in s if r["study"] == st and r["depth"] == d and r["b_total"] == str(B) and r["method"] == m][0][k])
        c = [r for r in con if r["study"] == st and r["depth"] == d and r["b_total"] == str(B) and r["metric"] == "rps" and r["target"] == "ce_raw" and r["comparator"] == "pearson_raw"][0]
        print(f"| {nm} | {B} | {g('ce_raw','rps'):.4f} | {g('pearson_raw','rps'):.4f} | {g('ce_raw','ce'):.3f} | {g('pearson_raw','ce'):.3f} | {g('ce_raw','tvd'):.3f} | {g('pearson_raw','tvd'):.3f} | "
              f"{float(c['diff_comparator_minus_target']):+.4f} [{float(c['cross_sim_lo_95']):+.4f}, {float(c['cross_sim_hi_95']):+.4f}] |")

def breakeven():
    b = [r for r in rd(E28 / "results/breakeven_summary.csv")]
    for d in ("1", "5"):
        bb = [r for r in b if r["d"] == d]
        for key, nm in (("N_pearson", "Pearson-fit"), ("N_uniform", "uniform"), ("N_raw", "raw judge")):
            v = [float(r[key]) for r in bb if r[key]]
            print(f"- d={d}, CE-fit vs {nm}: reached in {len(v)} of {len(bb)} set-judge cells; fewest {min(v):g}, median {np.median(v):g}, most {max(v):g}" if v else f"- d={d} vs {nm}: never")

if __name__ == "__main__":
    for h, fn in (("TABLE28_256", lambda: tab28(256)), ("TABLE28_16", lambda: tab28(16)), ("TABLE28_4", lambda: tab28(4)), ("WINS28", wins28), ("TABLE26", tab26), ("TABLE29", tab29), ("BREAKEVEN", breakeven)):
        print(f"\n<!-- {h} -->\n"); fn()
