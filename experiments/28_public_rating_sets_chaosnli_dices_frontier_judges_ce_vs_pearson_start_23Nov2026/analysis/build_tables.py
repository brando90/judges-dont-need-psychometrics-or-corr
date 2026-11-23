"""Build the report tables (markdown and LaTeX) from results/*.csv and the catalog (11-23-2026).
Usage: python3 build_tables.py EXPERIMENT_DIR   -> writes report/tables.json (each table as md and tex strings)"""
import json, re, sys
from pathlib import Path
import pandas as pd

NAMES = {"chaosnli_snli": "ChaosNLI SNLI", "chaosnli_mnli": "ChaosNLI MNLI-m", "chaosnli_alphanli": "ChaosNLI alphaNLI (control)",
         "dices350": "DICES-350", "dices990": "DICES-990"}
JN = {"gpt-6-astra": "GPT-6 Astra", "claude-opus-5-5": "Claude Opus 5.5", "kimi-k3": "Kimi K3", "glm-5.3": "GLM 5.3", "deepseek-v4.1-flash": "DeepSeek V4.1 Flash"}


def esc(x):
    return str(x).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#").replace("$", "\\$")


def tab(header, rows, align=None, size="\\small"):
    md = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)
    al = align or "l" + "r" * (len(header) - 1)
    tex = size + "\n\\begin{tabular}{" + al + "}\\toprule\n" + " & ".join(esc(h) for h in header) + " \\\\ \\midrule\n" + "\n".join(" & ".join(esc(c) for c in r) + " \\\\" for r in rows) + "\n\\bottomrule\\end{tabular}\n"
    return {"md": md, "tex": tex}


def rng(vals, n=5):
    v = [int(x) for x in vals if pd.notna(x)]
    if not v:
        return f"none ({n}/{n})"
    s = f"{min(v)}" if min(v) == max(v) else f"{min(v)}-{max(v)}"
    return s + (f" ({n - len(v)}/{n} none)" if len(v) < n else "")


def main(exp):
    exp = Path(exp); res = exp / "results"; out = {}
    be = pd.read_csv(res / "breakeven_summary.csv"); s = pd.read_csv(res / "scores_absolute.csv"); raw = pd.read_csv(res / "raw_judge_scores.csv")
    # T1 break-even N* across judges
    for d, key in ((1, "t1_d1"), (5, "t1_d5")):
        rows = []
        for ds in NAMES:
            g = be[(be.dataset == ds) & (be.d == d)]
            rows.append([NAMES[ds]] + [rng(g["N_" + c]) for c in ("pearson", "uniform", "both", "hist", "pcal", "raw")])
        out[key] = tab(["Set", "vs Pearson-fit", "vs uniform", "vs both", "vs histogram", "vs Pearson+calibration", "vs raw judge"], rows, "lllllll", "\\footnotesize")
    # T2 mean scores
    rows = []
    for ds in NAMES:
        r = [NAMES[ds]]
        for B in (16, 256):
            g = s[(s.dataset == ds) & (s.d == 1) & (s.B == B)].groupby("method").rps.mean()
            r += [f"{raw[raw.dataset == ds].rps.mean():.3f}" if B == 16 else "", f"{g['uniform']:.3f}", f"{g['hist']:.3f}", f"{g['pearson']:.3f}", f"{g['pcal']:.3f}", f"{g['ce']:.3f}"]
        rows.append(r)
    hdr = ["Set"] + ["raw judge", "uniform", "histogram", "Pearson-fit", "Pearson+cal", "CE-fit"] + ["", "uniform", "histogram", "Pearson-fit", "Pearson+cal", "CE-fit"]
    rows = [[r[0], r[1], *r[2:7], r[1], *r[8:13]] for r in rows]
    out["t2"] = tab(["Set", "raw", "unif", "hist", "Pear", "Pear+cal", "CE", "raw", "unif", "hist", "Pear", "Pear+cal", "CE"], rows, "lrrrrrrrrrrrr", "\\footnotesize")
    # T3 CE, TVD, top1 at B=256 d=1
    rows = []
    for ds in NAMES:
        g = s[(s.dataset == ds) & (s.d == 1) & (s.B == 256)].groupby("method")[["ce", "tvd", "top1"]].mean()
        r = [NAMES[ds]]
        for m in ("pearson", "ce"):
            r += [f"{g.loc[m,'ce']:.3f}", f"{g.loc[m,'tvd']:.3f}", f"{g.loc[m,'top1']:.3f}"]
        rows.append(r)
    out["t3"] = tab(["Set", "Pearson CE", "Pearson TVD", "Pearson top-1", "CE-fit CE", "CE-fit TVD", "CE-fit top-1"], rows, "lrrrrrr", "\\footnotesize")
    # T4 comparator wins
    w = pd.read_csv(res / "comparator_wins.csv"); w = w[w.metric == "rps"]
    rows = []
    for comp in ("pearson", "uniform", "raw", "hist", "pcal", "rps"):
        for ds in NAMES:
            g = w[(w.comparator == comp) & (w.dataset == ds)]
            if len(g):
                rows.append([{"pearson": "Pearson-fit", "uniform": "uniform", "raw": "raw judge", "hist": "histogram", "pcal": "Pearson+calibration", "rps": "RPS-fit (extra)"}[comp], NAMES[ds],
                             f"{len(g)} of 65", f"{int(g.B.min())} to {int(g.B.max())}", ", ".join(sorted({JN[j] for j in g.judge}))])
    out["t4"] = tab(["Comparator significantly better than CE-fit", "Set", "Cells", "Budgets", "Judges"], rows, "llrll", "\\footnotesize")
    # T5 forecast validity and costs
    fs = pd.read_csv(res / "forecast_status.csv")
    rows = []
    cs = pd.read_csv(exp / "forecasts" / "costs_11-23-2026.csv")
    for j, g in fs.groupby("judge"):
        c = cs[cs.judge == j].iloc[0]
        rows.append([JN[j], f"{int(g['uniform_fallback'].sum())} of {int(g['items'].sum())}", f"{int(c.ok_calls)}", f"{int(c.input_tokens):,}", f"{int(c.output_tokens):,}", f"{c.estimated_usd_at_configured_rate:.2f}", f"{c['reference_usd_if_priced_at_1.4_and_4.4']:.2f}"])
    out["t5"] = tab(["Judge", "Uniform fallback", "Calls ok", "Input tokens", "Output tokens", "USD at configured rate", "USD at 1.4/4.4 stand-in"], rows, "lrrrrrr", "\\footnotesize")
    # catalog
    cat = (exp / "CATALOG_11-23-2026.md").read_text().splitlines()
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in cat if re.match(r"\| \d+ \|", l)]
    out["catalog_rows"] = rows
    (exp / "report").mkdir(exist_ok=True); (exp / "report" / "tables.json").write_text(json.dumps(out, indent=1))
    for k in ("t1_d1", "t1_d5", "t2", "t3", "t4", "t5"):
        print(k); print(out[k]["md"])


if __name__ == "__main__":
    main(sys.argv[1])
