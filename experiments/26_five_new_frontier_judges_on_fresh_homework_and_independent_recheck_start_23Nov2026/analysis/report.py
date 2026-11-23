"""Turn the two analysis outputs of experiment 26 into the results tables (markdown and LaTeX) and the learning-curve figure (11-23-2026).

Use after recheck.py has written scores.csv, contrasts.csv and decisions.csv. Needs matplotlib only.
Example: python3 report.py NEW5_DIR EXP21_DIR COSTS.csv FORECASTS.csv OUT_DIR
Difference convention everywhere: comparator minus CE fit on the same held-out answers, so positive means the CE fit is better.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NEW = ["gpt-6-astra", "claude-opus-5-5", "kimi-k3", "glm-5.3", "deepseek-v4.1-flash"]
BUD = [256, 128, 64, 32, 16, 8, 4]
OPEN = {"kimi-k3", "glm-5.3", "deepseek-v4.1-flash"}


def rd(p):
    return list(csv.DictReader(open(p)))


def load(d):
    s = {(r["judge"], int(r["ratings_per_answer"]), int(r["training_ratings"]), r["method"], r["metric"]): float(r["mean"]) for r in rd(Path(d) / "scores.csv")}
    c = {(r["judge"], int(r["ratings_per_answer"]), int(r["training_ratings"]), r["comparator"], r["metric"]): (float(r["improvement"]), float(r["lower_975"]), float(r["upper_95"]), int(r["folds_favor_ce"])) for r in rd(Path(d) / "contrasts.csv")}
    dec = {(r["judge"], int(r["training_ratings"])): r for r in rd(Path(d) / "decisions.csv")}
    return s, c, dec


def f(x, n=4):
    return f"{x:.{n}f}"


def sgn(x, n=4):
    return f"{x:+.{n}f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def tex(head, rows, align=None):
    esc = lambda t: str(t).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("$", r"\$")
    align = align or "l" + "r" * (len(head) - 1)
    hl = " & ".join(esc(h) for h in head) + r" \\"
    lines = [r"\begin{longtable}{" + align + "}", r"\toprule", hl, r"\midrule", r"\endfirsthead", r"\toprule", hl, r"\midrule", r"\endhead"]
    lines += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(lines)


def main(new_dir, old_dir, costs_csv, fc_csv, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    S, C, D = load(new_dir)
    S2, C2, D2 = load(old_dir)
    old_judges = sorted({k[0] for k in S2})
    md, tx = [], {}

    def emit(name, title, head, rows, note="", align=None):
        md.append(f"### {title}\n\n{table(head, rows)}\n" + (f"\n{note}\n" if note else ""))
        tx[name] = tex(head, rows, align)

    # T1: test RPS levels, two-rating primary
    rows = []
    for j in NEW:
        for B in BUD:
            g = lambda m: S[(j, 2, B, m, "rps")]
            rows.append([j, B, f(g("ce_fit")), f(g("pearson_fit")), f(g("identity")), f(g("train_hist")), f(g("uniform"))])
    emit("t1", "Table 1. Held-out RPS (lower is better), primary condition (two ratings per answer), mean of ten folds",
         ["judge", "training ratings", "CE fit", "Pearson fit", "uncalibrated", "training histogram", "uniform"], rows)

    # T2: advantage of CE fit with intervals
    rows = []
    for j in NEW:
        for B in BUD:
            cells = []
            for comp in ("pearson_fit", "uniform", "train_hist"):
                m, lo, hi, nf = C[(j, 2, B, comp, "rps")]
                cells.append(f"{sgn(m)} [{sgn(lo)}, {sgn(hi)}]")
            nf = C[(j, 2, B, "pearson_fit", "rps")][3]
            d = D[(j, B)]
            rows.append([j, B] + cells + [f"{nf}/10", d["formal_status"]])
    emit("t2", "Table 2. CE-fit advantage in RPS (comparator minus CE fit; positive = CE better) with 95% two-sided interval, and the descending formal test",
         ["judge", "training ratings", "vs Pearson fit", "vs uniform", "vs training histogram", "folds CE > Pearson", "formal status"], rows,
         "Formal status: a budget passes if the one-sided lower bounds against both the Pearson fit and uniform are above 0; the descending test stops at the first failure and later budgets are marked not_tested_after_failure (they are not claims). The intervals are the 95% two-sided overlap-adjusted t over ten dependent folds.")

    # T3: test CE and TVD
    rows = []
    for j in NEW:
        for B in BUD:
            cell = []
            for met in ("ce", "tvd"):
                for m in ("ce_fit", "pearson_fit", "uniform"):
                    cell.append(f(S[(j, 2, B, m, met)], 3))
            rows.append([j, B] + cell)
    emit("t3", "Table 3. Held-out CE (nats) and total variation distance, primary condition, mean of ten folds (lower is better)",
         ["judge", "training ratings", "CE: CE fit", "CE: Pearson fit", "CE: uniform", "TVD: CE fit", "TVD: Pearson fit", "TVD: uniform"], rows)

    # T4: every case where correlation is not beaten
    rows = []
    for j in NEW:
        for B in BUD:
            for met in ("rps", "ce", "tvd"):
                m, lo, hi, nf = C[(j, 2, B, "pearson_fit", met)]
                if lo <= 0:
                    rows.append([j, B, met, sgn(m), f"[{sgn(lo)}, {sgn(hi)}]", "Pearson fit better on average" if m < 0 else "CE fit better on average, interval includes 0"])
    emit("t4", "Table 4. Every primary-condition case where the Pearson fit is not significantly worse than the CE fit (95% two-sided interval includes 0 or lies below 0)",
         ["judge", "training ratings", "metric", "Pearson minus CE", "95% interval", "reading"], rows,
         "Metric rps is the primary one; ce and tvd are reported as asked. Same sign convention: negative means the CE fit has the higher (worse) held-out value.")

    # T5: training histogram
    rows = []
    for j in NEW:
        for B in BUD:
            m, lo, hi, nf = C[(j, 2, B, "train_hist", "rps")]
            rows.append([j, B, f"{sgn(m)} [{sgn(lo)}, {sgn(hi)}]", "CE fit better" if lo > 0 else ("histogram better" if hi < 0 else "no clear difference")])
    emit("t5", "Table 5. CE fit versus the training histogram (descriptive; not part of the formal rule)",
         ["judge", "training ratings", "CE-fit advantage in RPS [95% interval]", "reading"], rows)

    # T6: one rating per answer secondary
    rows = []
    for j in NEW:
        cells = []
        for B in BUD:
            if (j, 1, B, "pearson_fit", "rps") not in C:
                cells.append("n/a")
                continue
            m, lo, hi, nf = C[(j, 1, B, "pearson_fit", "rps")]
            cells.append(f"{sgn(m, 3)}{'*' if lo > 0 else ''}")
        rows.append([j] + cells)
    emit("t6", "Table 6. Secondary condition (one rating per answer): CE-fit advantage in RPS over the Pearson fit; * marks a 95% lower bound above 0",
         ["judge"] + [str(B) for B in BUD], rows, "Columns are the number of training ratings; n/a = budget not run in this condition (more ratings than distinct answers available).")

    # T7: experiment 21 judges re-analysed
    rows = []
    for j in old_judges:
        for B in BUD:
            m, lo, hi, nf = C2[(j, 2, B, "pearson_fit", "rps")]
            mu = C2[(j, 2, B, "uniform", "rps")][0]
            d = D2[(j, B)]
            rows.append([j, B, f"{sgn(m)} [{sgn(lo)}, {sgn(hi)}]", sgn(mu), d["formal_status"]])
    emit("t7", "Table 7. The two experiment 21 judges re-analysed with the independent implementation (alpha 0.025 per judge)",
         ["judge", "training ratings", "vs Pearson fit [95% int.]", "vs uniform", "formal status"], rows)

    # costs
    rows = []
    tot = 0.0
    for r in rd(costs_csv):
        tot += float(r["estimated_usd_at_configured_rate"])
        rows.append([r["judge"], "open weights" if r["judge"] in OPEN else "closed", r["calls"], r["ok_calls"], r["input_tokens"], r["output_tokens"], "$" + r["estimated_usd_at_configured_rate"], "$" + r["reference_usd_if_priced_at_1.4_and_4.4"]])
    rows.append(["total", "", "", "", "", "", f"${tot:.2f}", ""])
    emit("t8", "Table 8. Cost of the new judge calls",
         ["judge", "weights", "calls", "ok calls", "input tokens", "output tokens", "est. USD (listed rate)", "ref. USD (1.4/4.4)"], rows,
         "Calls went through the Stanford AI API Gateway; the three open-weight models are listed at $0 there. The last column is a stand-in upper reference for them (GLM 5.3's list rate on another provider), not a charge. No plan route was charged beyond included usage.")

    # parse failures
    cnt = defaultdict(lambda: [0, 0])
    for r in rd(fc_csv):
        cnt[r["judge"]][0] += 1
        cnt[r["judge"]][1] += r["status"] != "valid"
    rows = [[j, cnt[j][0], cnt[j][1]] for j in NEW]
    emit("t9", "Table 9. Forecast parse status (invalid or failed forecasts become uniform and stay in the analysis)", ["judge", "forecasts", "invalid or failed"], rows)

    # figure
    cols = {"ce_fit": "#0072B2", "pearson_fit": "#D55E00", "train_hist": "#009E73", "uniform": "#7f7f7f"}
    lab = {"ce_fit": "CE fit", "pearson_fit": "Pearson fit", "train_hist": "training histogram", "uniform": "uniform"}
    mk = {"ce_fit": "o", "pearson_fit": "s", "train_hist": "^", "uniform": "x"}
    ls = {"ce_fit": "-", "pearson_fit": "--", "train_hist": ":", "uniform": "-."}
    xs = sorted(BUD)
    fig, axes = plt.subplots(2, 5, figsize=(19, 7.5), sharex="col")
    for k, j in enumerate(NEW):
        ax = axes[0, k]
        for m in ("uniform", "train_hist", "pearson_fit", "ce_fit"):
            ax.plot(xs, [S[(j, 2, B, m, "rps")] for B in xs], color=cols[m], marker=mk[m], ls=ls[m], lw=2, ms=6, label=lab[m])
        ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
        ax.set_title(j + (" (open)" if j in OPEN else " (closed)"), fontsize=11)
        ax.set_ylabel("held-out RPS, mean of 10 folds\n(lower is better)" if k == 0 else "")
        ax.set_ylim(0, 0.34)
        ax.grid(True)
        ax = axes[1, k]
        m_ = [C[(j, 2, B, "pearson_fit", "rps")] for B in xs]
        ax.errorbar(xs, [t[0] for t in m_], yerr=[[t[0] - t[1] for t in m_], [t[2] - t[0] for t in m_]], color=cols["ce_fit"], marker="o", lw=2, capsize=4)
        ax.axhline(0, color="black", lw=1)
        ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel("training ratings")
        ax.set_ylabel("held-out RPS: Pearson fit minus CE fit,\nmean and 95% interval (positive = CE better)" if k == 0 else "")
        ax.set_ylim(-0.03, 0.06)
        ax.grid(True)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Experiment 26: UNT short answers, leave-one-assignment-out, two ratings per answer", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "fig_learning_curves_11-23-2026.png", dpi=150)
    (out / "tables_11-23-2026.md").write_text("\n".join(md))
    for k, v in tx.items():
        (out / f"table_{k}.tex").write_text(v + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:6])
