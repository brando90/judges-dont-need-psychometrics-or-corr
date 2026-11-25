"""Build results_11-25-2026.md (Exp 31) from results/*.csv written by fit_scaling_laws.py and the saved results of Exp 26, 28, 29.
Usage: python3 build_results_md.py [REPO_ROOT] > results_11-25-2026.md"""
import csv, glob, sys, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1]
rd = lambda p: list(csv.DictReader(open(p)))
fits = rd(HERE / "results/fits_11-25-2026.csv"); pairs = rd(HERE / "results/floor_comparisons_11-25-2026.csv")
fl = lambda v: float(v) if v not in ("", None) else float("nan")
RAW = "https://raw.githubusercontent.com/brando90/judges-dont-need-psychometrics-or-corr/main/experiments/" + HERE.name + "/figures/"
MN = {"rps": "RPS", "ce": "CE (nats)", "tvd": "TVD"}
D = 3


def f3(x): return "n/a" if np.isnan(x) else f"{x:.{D}f}"


def floor_s(r):
    if not r["floor"]: return "not fit"
    v = fl(r["floor"]); s = f"{v:.3f} [{fl(r['floor_lo']):.3f}, {fl(r['floor_hi']):.3f}]"
    return s + (" (at 0 bound)" if v < 1e-3 else "")


def b_s(r):
    if not r["floor"]: return "not fit"
    if r["b"] in ("", "nan"): return "no decay (a = 0)"
    v = fl(r["b"]); s = f"{v:.2f} [{fl(r['b_lo']):.2f}, {fl(r['b_hi']):.2f}]"
    return s + (" (at 0.02 bound)" if v <= 0.021 else " (at 3 bound)" if v >= 2.99 else "")


# panel lists (level, id order)
SETS = [("28_chaosnli_snli_d1", "ChaosNLI SNLI"), ("28_chaosnli_mnli_d1", "ChaosNLI MNLI-m"), ("28_chaosnli_alphanli_d1", "alphaNLI (2 classes)"), ("28_dices350_d1", "DICES-350"), ("28_dices990_d1", "DICES-990")]
JUD = [(f"26_{j}_rpa2", n) for j, n in (("gpt-6-astra", "GPT-6 Astra"), ("claude-opus-5-5", "Claude Opus 5.5"), ("kimi-k3", "Kimi K3"), ("glm-5.3", "GLM 5.3"), ("deepseek-v4.1-flash", "DeepSeek V4.1 Flash"))]
PAN = [("29_chaos3_qwen_d1", "ChaosNLI, Qwen3-4B"), ("29_chaos3_qwen_d5", "ChaosNLI, Qwen3-4B, 5 per item"), ("29_alpha2_qwen_d1", "alphaNLI, Qwen3-4B"), ("29_dices3_qwen_d1", "DICES, Qwen3-4B"),
       ("29_hs2_qwen_d1", "HelpSteer2, Qwen3-4B"), ("29_chaos3_llama_d1", "ChaosNLI, Llama-3.1-8B"), ("29_dices3_llama_d1", "DICES, Llama-3.1-8B")]
GROUPS = [("Exp 28, per rating set (mean over 5 judges, 1 rating per item)", SETS), ("Exp 26, per judge (2 ratings per answer)", JUD), ("Exp 29, per LoRA panel", PAN)]
idx = {(r["panel_id"], r["metric"], r["objective"]): r for r in fits}
pidx = {(r["panel_id"], r["metric"]): r for r in pairs}


def table_metric(m):
    print(f"| Panel | Objective | Value at largest N | Floor [95% bootstrap] | Exponent b [95% bootstrap] | Verdict |\n|---|---|---|---|---|---|")
    for gname, lst in GROUPS:
        print(f"| **{gname}** | | | | | |")
        for pid, nm in lst:
            for obj in ("CE", "Pearson"):
                r = idx[(pid, m, obj)]
                print(f"| {nm} | {obj}-{'LoRA' if pid.startswith('29') else 'fit'} | {f3(fl(r['last']))} (N = {r['N_max']}) | {floor_s(r)} | {b_s(r)} | {r['verdict']} |")


def refs_table():
    print("| Panel | N range | Uniform RPS / CE / TVD | Raw judge (Exp 29: untuned base model) RPS / CE / TVD |\n|---|---|---|---|")
    for gname, lst in GROUPS:
        print(f"| **{gname}** | | | |")
        for pid, nm in lst:
            r = {m: idx[(pid, m, "CE")] for m in ("rps", "ce", "tvd")}
            print(f"| {nm} | {r['rps']['N_min']} to {r['rps']['N_max']} ({r['rps']['K']} budgets) | " + " / ".join(f3(fl(r[m]["uniform"])) for m in ("rps", "ce", "tvd")) + " | " + " / ".join(f3(fl(r[m]["raw"])) for m in ("rps", "ce", "tvd")) + " |")


def pair_table():
    print("| Panel | RPS: CE floor / Pearson at largest N / gap [95%] | CE (nats): same | TVD: same | CE floor below raw judge (RPS / CE / TVD, 95% bound) |\n|---|---|---|---|---|")
    for gname, lst in GROUPS:
        print(f"| **{gname}** | | | | |")
        for pid, nm in lst:
            cells = []
            for m in ("rps", "ce", "tvd"):
                p = pidx.get((pid, m))
                cells.append("not fit" if p is None else f"{fl(p['floor_ce']):.3f} / {fl(p['pearson_last']):.3f} / {fl(p['gap']):+.3f} [{fl(p['gap_lo']):+.3f}, {fl(p['gap_hi']):+.3f}]" + (" **>0**" if fl(p["gap_lo"]) > 0 else ""))
            below = " / ".join("n/a" if pidx.get((pid, m)) is None else ("yes" if pidx[(pid, m)]["ce_floor_below_raw"] == "True" else "no") for m in ("rps", "ce", "tvd"))
            print(f"| {nm} | " + " | ".join(cells) + f" | {below} |")


def secondary():
    import json
    sec = json.load(open(HERE / "results/secondary_and_plots_11-25-2026.json"))
    c28 = collections.defaultdict(list)
    for exp, key, nm, meth, v in sec["secondary_corr_28"]:
        ds, j, d = key.split("|")
        if d == "d1": c28[(ds, nm, meth)].append(v)
    print("| Panel | Pearson r, CE | Pearson r, Pearson | Spearman rho, CE | Spearman rho, Pearson |\n|---|---|---|---|---|")
    for pid, nm in SETS:
        ds = pid[3:-3]
        g = lambda n, m: np.mean(c28[(ds, n, m)])
        print(f"| {nm} (Exp 28, at 256 ratings, mean of 5 judges) | {g('r','ce'):.3f} | {g('r','pearson'):.3f} | {g('rho','ce'):.3f} | {g('rho','pearson'):.3f} |")
    c29 = {(a, b, c): d for a, b, c, d in sec["secondary_corr_29"]}
    for pid, nm in PAN:
        k = pid[3:]; k = k[:-3] + "|d" + k[-1]
        print(f"| {nm} (Exp 29, at the largest budget) | {c29[(k,'r','ce')]:.3f} | {c29[(k,'r','pearson')]:.3f} | {c29[(k,'rho','ce')]:.3f} | {c29[(k,'rho','pearson')]:.3f} |")
    print("\nExp 26: no correlations are in its saved scores file, so none are shown.")


def cell_summary():
    print("| Metric | Ratings per item | CE-fit: cells fitting well (floor identified / not) | CE-fit median b among identified [IQR] | Pearson-fit verdicts | Cells where Pearson at largest N is above the CE floor (95%) | CE floor below raw judge (95%) |\n|---|---|---|---|---|---|---|")
    for m in ("rps", "ce", "tvd"):
        for d in ("d1", "d5"):
            cs = [r for r in fits if r["level"] == "cell" and r["panel_id"].endswith(d) and r["metric"] == m]
            pc = [r for r in pairs if r["level"] == "cell" and r["panel_id"].endswith(d) and r["metric"] == m]
            ce = [r for r in cs if r["objective"] == "CE"]; pe = [r for r in cs if r["objective"] == "Pearson"]
            good = [r for r in ce if r["verdict"].startswith("fits")]; ident = [r for r in good if r["floor_identified"] == "True"]
            bs = [fl(r["b"]) for r in ident if r["b"]]
            pv = collections.Counter(r["verdict"].split(";")[0].split(",")[0] for r in pe)
            print(f"| {MN[m]} | {d[1]} | {len(good)} of {len(ce)} ({len(ident)} / {len(good) - len(ident)}) | {np.median(bs):.2f} [{np.percentile(bs,25):.2f}, {np.percentile(bs,75):.2f}] | " + "; ".join(f"{v} {k}" for k, v in pv.most_common()) +
                  f" | {sum(fl(r['gap_lo'])>0 for r in pc)} of {len(pc)} | {sum(r['ce_floor_below_raw']=='True' for r in pc)} of {len(pc)} |")


def plots():
    n = 0
    for gname, lst, stem, hk in (("Exp 28, per rating set (1 rating per item, mean of 5 judges)", SETS, "exp28", "28"), ("Exp 26, per judge (2 ratings per answer)", JUD, "exp26", "26"), ("Exp 29, per LoRA panel", PAN, "exp29", "29")):
        print(f"\n#### {gname}\n")
        for pid, nm in lst:
            key = pid.split("_", 1)[1]
            if stem == "exp29" and pid.endswith("d5"): continue
            n += 1
            name = lambda m: f"fit_{stem}_{m}_{key}.png"
            fl_ = {r["metric"]: r for r in fits if r["panel_id"] == pid and r["objective"] == "CE"}
            fp_ = {r["metric"]: r for r in fits if r["panel_id"] == pid and r["objective"] == "Pearson"}
            print(f"| RPS | CE (nats) |\n|---|---|\n| <img src=\"figures/{name('rps')}\" width=\"400\"> | <img src=\"figures/{name('ce')}\" width=\"400\"> |\n")
            print(f"*Figure {n}. {nm}: held-out RPS (left) and CE (right) against the number of human ratings used to fit, N (lower is better; log x, grid on). Points with 95% error bars from the saved results; blue = CE, orange = Pearson; solid lines = fitted floor + a N^-b over the observed N, dotted = extrapolation to 4 times the largest N, band = 95% parametric-bootstrap band, dotted horizontal = fitted floor, grey = uniform, dashed = raw judge (Exp 29: untuned base model). "
                  f"Fitted RPS law, CE: floor {floor_s(fl_['rps'])}, b {b_s(fl_['rps'])}; Pearson: floor {floor_s(fp_['rps'])}, b {b_s(fp_['rps'])}. TVD file: [{name('tvd')}](figures/{name('tvd')}). {H[hk]}*\n")


H = {"28": "Harness (Exp 28): five judges (GPT-6 Astra, Claude Opus 5.5, Kimi K3, GLM 5.3, DeepSeek V4.1 Flash) through the Stanford AI API Gateway, a 27-transform head fit by CE or by Pearson on N human ratings, held-out items; orchestration and analysis by Claude Sonnet 5.5 in Claude Code.",
     "26": "Harness (Exp 26): the same five judges through the Stanford AI API Gateway chat completions, CE-fit and Pearson-fit heads, 10 folds, analysis by ub9sc in Python (numpy, scipy).",
     "29": "Harness (Exp 29): LoRA fine-tuning of Qwen3-4B-Instruct-2507 and Llama-3.1-8B with a CE or a Pearson objective (raw, no calibration), 10 nested draws, custom PyTorch/HF code on the lab cluster."}

if __name__ == "__main__":
    tmpl = (HERE / "results_template.md").read_text()
    import io, contextlib
    def cap(fn):
        b = io.StringIO()
        with contextlib.redirect_stdout(b): fn()
        return b.getvalue().strip()
    for k, fn in (("REFS", refs_table), ("TAB_RPS", lambda: table_metric("rps")), ("TAB_CE", lambda: table_metric("ce")), ("TAB_TVD", lambda: table_metric("tvd")),
                  ("PAIRS", pair_table), ("SECONDARY", secondary), ("CELLS", cell_summary), ("PLOTS", plots)):
        tmpl = tmpl.replace("{{" + k + "}}", cap(fn))
    for k, v in H.items(): tmpl = tmpl.replace("{{H" + k + "}}", v)
    print(tmpl)
