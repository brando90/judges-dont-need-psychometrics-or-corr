"""Write report/REPORT_11-23-2026.tex and .md from report/tables.json. Usage: python3 build_report.py EXPERIMENT_DIR"""
import json, sys
from pathlib import Path

exp = Path(sys.argv[1]); T = json.loads((exp / "report" / "tables.json").read_text())
URL = "https://github.com/brando90/judges-dont-need-psychometrics-or-corr/tree/main/experiments/28_public_rating_sets_chaosnli_dices_frontier_judges_ce_vs_pearson_start_23Nov2026"

COST_RATES = [("0.05", "ChaosNLI, observed (up to 0.50 per batch of ten examples)"), ("0.15", "Chaganty et al., observed (per summary judgment)"), ("1", "illustrative"), ("5", "illustrative")]
NS = [8, 20, 64, 256, 1000]
cost_rows = [[str(n)] + [f"{n * float(r):,.2f}" for r, _ in COST_RATES] for n in NS]

CROSS = [
    ["ChaosNLI SNLI, MNLI-m, alphaNLI (this experiment)", "5 judges, one rating per item", "4 to 8", "4", "SNLI 4; MNLI-m none; alphaNLI 4 (one judge none)"],
    ["DICES-350, DICES-990 (this experiment)", "5 judges, one rating per item", "4", "4 to 8", "DICES-350 8 (4 of 5 judges none); DICES-990 none"],
    ["KonIQ-10k (Codex, exp. 23)", "image quality", "12 (24 with draw resampling, 48 at three per image)", "12", "no budget beats all four controls"],
    ["USR PersonaChat (Codex, exp. 23)", "dialogue quality", "48 (96 at three grades per response)", "48", "n.r."],
    ["USR Topical-Chat (Codex, exp. 23)", "dialogue quality", "none", "none", "n.r."],
    ["MovieLens 100K (Codex, exp. 23)", "movie stars", "CE beats Pearson at all budgets", "none", "none"],
    ["SummEval, GPT-4 forecasts (Codex, exp. 23)", "summary quality", "no sustained CE win at 6 to 768 ratings", "none", "calibrated Pearson and histogram beat CE"],
    ["HelpSteer2 LoRA Qwen3-4B (Codex, exp. 24 and 25)", "fine-tuned judge", "n.r.", "from 16 total ratings", "calibrated Pearson: none up to 8,192"],
    ["HelpSteer2 LoRA Llama-3.1-8B (Codex, exp. 24 and 25)", "fine-tuned judge", "n.r.", "none", "none"],
]
CH = ["Set", "Setting", "CE beats Pearson from N =", "CE beats uniform from N =", "CE beats Pearson+calibration from N ="]


def md_tab(h, rows):
    return "| " + " | ".join(h) + " |\n|" + "|".join("---" for _ in h) + "|\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)


def tex_esc(x):
    return str(x).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#").replace("$", "\\$")


def tex_tab(h, rows, al, size="\\footnotesize"):
    return size + "\n\\begin{tabular}{" + al + "}\\toprule\n" + " & ".join(tex_esc(c) for c in h) + " \\\\ \\midrule\n" + "\n".join(" & ".join(tex_esc(c) for c in r) + " \\\\" for r in rows) + "\n\\bottomrule\\end{tabular}\n"


BOTTOM = ("On five public sets of real human ratings (ChaosNLI SNLI and MNLI-m, an alphaNLI control, DICES-350 and DICES-990) and five frontier judges, "
          "fitting the judge by cross-entropy (CE) beat fitting it by Pearson correlation and beat the uniform forecast on held-out ranked probability score (RPS) "
          "after only 4 to 8 human ratings at one rating per item, or 10 to 20 ratings at five per item, in all 25 set-by-judge cells. "
          "At $5 a rating that is under $100. Scored on RPS, CE and total variation, never on correlation, the Pearson-fit judge was significantly better in 19 of 325 comparison cells, "
          "mostly at two training items, where the tie rule keeps the raw judge. "
          "Two findings limit the claim. (1) The raw frontier judge is already strong on ChaosNLI: CE-fit beats it, sustained, only from 16 to 32 ratings on MNLI-m "
          "(three of five judges), 64 on SNLI (one judge) and 256 on alphaNLI (one judge), and from 8 to 128 on DICES-350 and 16 on DICES-990 (four judges). "
          "For the remaining judge-set pairs it never does within 256. "
          "(2) Pearson-fit followed by a distribution calibration on the same ratings (a rank-then-calibrate recipe) is worse than CE-fit on SNLI and alphaNLI, level with it on MNLI-m at 256 ratings and better than it on DICES "
          "from 64 ratings on (a few judges); so what helps is fitting a distribution with a proper score, not the correlation step. "
          "ChaosNLI SNLI and MNLI-m are subsets chosen for human disagreement, so these break-even sizes may not carry to easier items.")

DERIV = [
    ("R1. Pearson sees only the first moment, up to level and scale.",
     "For a forecast $q_x$ over categories with scores $s_k$, the expected score is $m_q(x)=\\sum_k s_k q_x(k)$. Pearson correlation of $m_q(X)$ with the human mean $\\mu(X)$ satisfies "
     "$\\operatorname{Corr}(a+b\\,m_q,\\mu)=\\operatorname{Corr}(m_q,\\mu)$ for every $a$ and every $b>0$. Two judges whose expected scores differ by such a map get the same score whatever their spread, skew or calibration. "
     "With $K$ categories the map $q\\mapsto m_q$ discards $K-2$ degrees of freedom of the forecast, and a fit that maximizes this correlation cannot choose among forecasts with equal $m_q$. "
     "Here the fit chooses among 27 transforms $q\\propto p^t\\exp(bu_k+cu_k^2)$; transforms that change the level or scale of $m_q$ are invisible to the objective, so which one it picks is decided by noise in the training means."),
    ("R2. CE and RPS are proper; they are minimized only by the full human distribution.",
     "For true distribution $h$ and forecast $q$, $\\mathrm{CE}(q)-\\mathrm{CE}(h)=D_{\\mathrm{KL}}(h\\Vert q)\\ge0$ with equality only at $q=h$. "
     "The ranked probability score $\\mathrm{RPS}(q,y)=\\frac1{K-1}\\sum_{k<K-1}(F_q(k)-\\mathbf 1[y\\le k])^2$ is the sum of Brier scores of the $K-1$ cumulative events, so its expectation is minimized at $F_q=F_h$, again only at $q=h$. "
     "Both therefore reward getting the mean and every higher moment right, and neither is defined through correlation. TVD$=\\tfrac12\\sum_k|h_k-q_k|$ is a metric on distributions, zero only at $q=h$."),
    ("R3. Non-circular evaluation.",
     "Every test item is scored against its full empirical human distribution (100, 123 or 69 to 76 ratings), so mean test CE and RPS equal means over individual held-out human ratings. "
     "Pearson and Spearman are computed for description only. Top-1 agreement is reported because neither objective optimizes it."),
    ("R4. Break-even.",
     "Let $\\Delta_B$ be the mean over test items of (comparator metric minus CE-fit metric) at budget $B$ ratings, positive when CE-fit is better, averaged over 200 training draws. "
     "The simultaneous lower bound $L_B=\\bar\\Delta_B-c\\,\\widehat{\\mathrm{se}}_B$ takes $c$ as the 97.5\\% quantile of the maximum standardized bootstrap deviation across the budgets of one condition. "
     "The break-even $N^\\star=\\min\\{B: L_{B'}>0\\ \\text{for all tested } B'\\ge B\\}$, and for the two primary comparators (Pearson-fit and uniform) both bounds must exceed zero. It is the smallest tested budget, not an interpolation."),
    ("R5. The alphaNLI control.",
     "With $K=2$ the mean $m_q=q(1)$ identifies $q$, so R1 removes no shape information. If CE-fit still beats Pearson-fit there, the gap comes from level and scale, which R1 shows Pearson cannot see, and not from higher moments. "
     "It does (Table 1)."),
]


def build_tex():
    cat = T["catalog_rows"]
    L = []
    A = L.append
    A(r"""\documentclass[10pt]{article}
\usepackage[letterpaper,margin=0.8in]{geometry}
\usepackage{amsmath,amssymb,booktabs,tabularx,microtype,xcolor,graphicx,longtable,array}
\usepackage[colorlinks,allcolors=blue!55!black]{hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\begin{document}
{\small\scshape Public human ratings \quad|\quad Does CE-fitting beat Pearson-fitting on real rating data, and from how many ratings?}\par
{\LARGE\bfseries Break-even number of human ratings: CE-fit vs Pearson-fit on real public rating sets}\par
{\small As of 11-23-2026 \quad|\quad \href{""" + URL + r"""}{Experiment 28, code, forecasts and tables}}\par
\textbf{Bottom line.} """ + tex_esc(BOTTOM).replace("\\$5", "\\$5") + r"""

\textbf{What each metric can and cannot see.} Pearson and Spearman see only the order and linear scale of the expected score: the first moment, up to level and scale (R1). RPS, CE and TVD see the whole forecast distribution against the human distribution (R2). Top-1 agreement sees only the mode. Ordinal coding is an assumption (NLI: contradiction $<$ neutral $<$ entailment; DICES: No $<$ Unsure $<$ Yes); CE and TVD do not depend on it, RPS and Pearson do.

\section*{1. Derivation}""")
    for t, b in DERIV:
        A(r"\textbf{" + t + r"} " + b + "\n")
    A(r"""
\section*{2. Design}
Data: public files read in full, per-item human count vectors. Items run: ChaosNLI SNLI 600 of 1,514, MNLI-m 600 of 1,599, alphaNLI 400 of 1,532, DICES-350 350 of 350, DICES-990 400 of 990 (seeded subsamples, protocol frozen before any judge call). 
Judges (harness: Stanford AI API Gateway chat completions, temperature 0, 2,048 output tokens, one call per item): GPT-6 Astra and Claude Opus 5.5 (closed weights); Kimi K3, GLM 5.3 and DeepSeek V4.1 Flash (open weights). Orchestration and analysis: Claude Sonnet 5.5 in Claude Code, headless. 
Each forecast goes through the 27-transform head of experiment 21, fitted by CE (training CE on individual ratings), by Pearson (correlation of expected score with the training item mean), by RPS (extra), or not fitted (raw). Comparators fitted from the same ratings: uniform, training histogram, Pearson-fit then four-bin distribution calibration. 
Budgets: 4 to 256 ratings at one per item; 10 to 320 at five per item; five folds, 200 training draws per fold, nested budgets, 4,000 item bootstraps. Forecasts, receipts and costs were committed before any label was joined; an independent plain-loop re-implementation reproduced the vectorized scores to $10^{-15}$ for three judge-set pairs.

\section*{3. Results}
\textbf{Table 1. Break-even $N^\star$ on held-out RPS, one rating per item (range over the 5 judges; ``none'' = no sustained win).}\par
""" + tex_tab(["Set", "vs Pearson-fit", "vs uniform", "vs both", "vs histogram", "vs Pearson+calibration", "vs raw judge"], [r for r in _rows(T["t1_d1"]["md"])], "lllllll"))
    A(r"""
\textbf{Table 2. Same, five ratings per item (N is total ratings).}\par
""" + tex_tab(["Set", "vs Pearson-fit", "vs uniform", "vs both", "vs histogram", "vs Pearson+calibration", "vs raw judge"], [r for r in _rows(T["t1_d5"]["md"])], "lllllll"))
    A(r"""
\textbf{Table 3. Mean held-out RPS over the 5 judges, one rating per item; left block 16 ratings, right block 256 ratings (lower is better).}\par
""" + tex_tab(["Set", "raw", "unif", "hist", "Pear", "Pear+cal", "CE", "raw", "unif", "hist", "Pear", "Pear+cal", "CE"], _rows(T["t2"]["md"]), "lrrrrrrrrrrrr") + r"""
\textbf{Table 4. The other metrics at 256 ratings (one per item), mean over judges: CE in nats, TVD, top-1 agreement. CE-fit is better on all three on every set except top-1 on alphaNLI (0.878 against 0.885).}\par
""" + tex_tab(["Set", "Pearson CE", "Pearson TVD", "Pearson top-1", "CE-fit CE", "CE-fit TVD", "CE-fit top-1"], _rows(T["t3"]["md"]), "lrrrrrr") + r"""
\textbf{Table 5. Every comparator that is significantly better than CE-fit on RPS (cells are judge-by-budget-by-condition combinations; 65 per set).} Pearson-fit at 10 ratings is the two-item, five-ratings condition, where almost every transform gets Pearson $\pm1$, so the tie rule keeps the identity, that is the raw judge; the raw judge beats a head fitted from two items.\par
""" + tex_tab(["Comparator better than CE-fit", "Set", "Cells", "Budgets", "Judges"], _rows(T["t4"]["md"]), "llrlp{3.4in}", "\\scriptsize") + r"""
\textbf{Table 6. Cross-set break-even. Codex's thresholds are as published in experiments 23 to 25 and were not re-run; their definitions of the Pearson comparator differ in detail.} n.r. = not reported there.\par
""" + tex_tab(CH, CROSS, "p{2.1in}p{1.0in}p{1.4in}p{0.9in}p{1.4in}", "\\scriptsize") + r"""
\textbf{Table 7. What N ratings cost (USD), under four per-rating prices.} Observed: ChaosNLI paid up to \$0.50 per batch of ten examples; Chaganty et al. about \$0.15 per summary judgment. Illustrative: \$1 and \$5 (assumptions). DICES and HelpSteer2 prices could not be established from public sources.\par
""" + tex_tab(["N ratings", "$0.05", "$0.15", "$1", "$5"], cost_rows, "rrrrr") + r"""
Reading Table 7 with Tables 1 and 2: beating Pearson-fit and uniform takes 4 to 8 ratings (one per item) or 10 to 20 (five per item), at most about \$100 at \$5 a rating. Beating the raw judge takes 16 to 256 ratings ($\approx$\$80 to \$1,280 at \$5) where it happens, and on SNLI, alphaNLI and some judges it does not within 256.

\textbf{Table 8. Forecast validity and judge-call cost (list rate for the closed judges; the open models are listed at 0 and are also shown at a 1.4/4.4 stand-in).} Fallback means unparseable or failed output, scored as uniform by a rule frozen before the run; none of the 11,750 calls errored. Estimated total \$19.69 at configured rates (\$39.0 with the 1.4/4.4 stand-in for the open models), plus \$1.34 in the layer 4 task: under the \$80 stop of the shared \$100 cap.\par
""" + tex_tab(["Judge", "Uniform fallback", "Calls ok", "Input tokens", "Output tokens", "USD configured", "USD stand-in"], _rows(T["t5"]["md"]), "lrrrrrr") + r"""
\begin{center}\includegraphics[width=\linewidth]{../figures/learning_curves_rps.png}\end{center}
\textbf{Figure 1.} Held-out RPS against ratings spent; band is min to max over judges.

\begin{center}\includegraphics[width=\linewidth]{../figures/ce_advantage_rps_d1.png}\end{center}
\textbf{Figure 2.} CE advantage over each comparator, one rating per item; above zero is CE better.

\section*{4. Caveats}
\begin{itemize}\setlength\itemsep{1pt}
\item Ordinal coding is assumed (R1, Design); alphaNLI has two categories and is a control, not a test of shape.
\item ChaosNLI SNLI and MNLI-m keep only items whose original five labels were not unanimous (agree\_3), so they over-represent disagreement; DICES items are adversarial conversations. Break-even sizes may differ on easier items.
\item Ratings here are drawn from 100, 123 or about 73 released ratings per item; test scores condition on that pool, on the item subsample and on the 200 draws.
\item Frozen parsing turned 0 to 7\% of each judge's outputs into the uniform forecast (Table 8; Claude Opus 5.5 often put prose before the JSON). No lenient re-parse was run.
\item Intervals for the raw-judge comparison are the same simultaneous bootstrap; break-even ``vs raw'' compares a fitted head with an unfitted forecast, so it is the practical question, not the correlation-versus-CE question.
\item Pearson-fit here selects among the same 27 transforms as CE-fit; a different Pearson-trained model (for example fine-tuned weights) is not tested, only head fitting (experiments 24 and 25 cover weights).
\item Per-rating prices come from other studies and assumptions (Table 7); no payment was made here.
\item The catalog (next pages) lists 26 sets. Not verified: Jigsaw Unintended Bias (login wall); Chatbot Arena file contents (gated; card only); WebGPT beyond a 2,752-pair sample; SHP beyond 6 of 18 subreddits; licences and selection for LeWiDi-2025 and Pavlick and Kwiatkowski. Only ChaosNLI and DICES were run here; Codex covered KonIQ, SummEval, USR, MovieLens and HelpSteer2.
\end{itemize}

\clearpage\newgeometry{paperwidth=11in,paperheight=8.5in,margin=0.5in}\pdfpagewidth=11in\pdfpageheight=8.5in
\section*{Appendix. Catalog of public rating sets (checked at the source, 11-23-2026)}
\tiny\setlength{\tabcolsep}{2pt}
\begin{longtable}{@{}r p{1.15in} p{1.0in} p{0.75in} p{0.8in} p{0.8in} p{0.9in} p{0.8in} p{1.2in} p{1.5in}@{}}
\toprule """ + " & ".join(tex_esc(x) for x in ["#", "Name", "Task", "Items", "Ratings per item (min / median / max)", "Scale", "Individual ratings released", "Licence", "Link", "Selection bias"]) + r""" \\ \midrule\endhead
""" + "\n".join(" & ".join((r"\href{" + c + "}{" + tex_esc(c.replace('https://', '')[:60]).replace('/', '/\\allowbreak{}').replace('-', '-\\allowbreak{}') + "}") if i == 8 else tex_esc(c) for i, c in enumerate(r)) + r" \\" for r in cat) + r"""
\bottomrule\end{longtable}
\clearpage\restoregeometry\pdfpagewidth=8.5in\pdfpageheight=11in
\end{document}
""")
    return "\n".join(L)


def _rows(md):
    lines = md.splitlines()[2:]
    return [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]


def build_md():
    L = []; A = L.append
    A("# Break-even number of human ratings: CE-fit vs Pearson-fit on real public rating sets\n\nAs of 11-23-2026. Experiment 28. [Code, forecasts and tables](" + URL + ")\n")
    A("**Bottom line.** " + BOTTOM + "\n")
    A("**What each metric can and cannot see.** Pearson and Spearman see only the order and linear scale of the expected score: the first moment, up to level and scale (R1). RPS, CE and TVD see the whole forecast distribution against the human distribution (R2). Top-1 agreement sees only the mode. Ordinal coding is an assumption (NLI: contradiction < neutral < entailment; DICES: No < Unsure < Yes); CE and TVD do not depend on it, RPS and Pearson do.\n")
    A("## 1. Derivation\n")
    for t, b in DERIV:
        A("**" + t + "** " + b.replace("\\%", "%").replace("\\,", " ") + "\n")
    A("## 2. Design\n\nSee the PDF (Design section): five judges through the Stanford AI API Gateway (GPT-6 Astra, Claude Opus 5.5, Kimi K3, GLM 5.3, DeepSeek V4.1 Flash), orchestration and analysis by Claude Sonnet 5.5 in Claude Code; 27-transform head fitted by CE, Pearson or RPS; budgets 4 to 256 (one rating per item) and 10 to 320 (five per item); 5 folds, 200 draws, 4,000 bootstraps; protocol and forecasts committed before any label join.\n")
    A("## 3. Results\n")
    A("**Table 1. Break-even N on held-out RPS, one rating per item (range over 5 judges).**\n\n" + T["t1_d1"]["md"] + "\n")
    A("**Table 2. Same, five ratings per item (N is total ratings).**\n\n" + T["t1_d5"]["md"] + "\n")
    A("**Table 3. Mean held-out RPS over 5 judges, one rating per item; first six columns 16 ratings, last six 256 ratings.**\n\n" + T["t2"]["md"] + "\n")
    A("**Table 4. Other metrics at 256 ratings (one per item), mean over judges.**\n\n" + T["t3"]["md"] + "\n")
    A("**Table 5. Every comparator significantly better than CE-fit on RPS (65 cells per set).**\n\n" + T["t4"]["md"] + "\n")
    A("**Table 6. Cross-set break-even (Codex's thresholds as published in experiments 23 to 25, not re-run).**\n\n" + md_tab(CH, CROSS) + "\n")
    A("**Table 7. Cost of N ratings (USD).**\n\n" + md_tab(["N ratings"] + [f"${r} ({n})" for r, n in COST_RATES], cost_rows) + "\n")
    A("**Table 8. Forecast validity and judge-call cost.**\n\n" + T["t5"]["md"] + "\n")
    A("Figures: `../figures/learning_curves_rps.png`, `../figures/learning_curves_ce.png`, `../figures/ce_advantage_rps_d1.png`.\n")
    A("## 4. Caveats\n\n- Ordinal coding is assumed; alphaNLI is a control.\n- ChaosNLI SNLI and MNLI-m are agree_3 subsets (disagreement over-represented); DICES items are adversarial.\n- Test scores condition on the item subsample and the 200 draws.\n- Frozen parsing turned 0 to 7% of each judge's outputs into the uniform forecast; no lenient re-parse was run.\n- Break-even against the raw judge compares a fitted head with an unfitted forecast.\n- Only head fitting is tested, not weight training.\n- Per-rating prices come from other studies and assumptions; no payment was made.\n- Catalog gaps: Jigsaw Unintended Bias (login wall), Chatbot Arena file contents (gated), WebGPT beyond a sample, SHP beyond 6 of 18 subreddits, some licences.\n")
    A("## Appendix. Catalog of public rating sets\n\nSee [CATALOG_11-23-2026.md](../CATALOG_11-23-2026.md) (26 sets, checked at the source).\n")
    return "\n".join(L)


(exp / "report").mkdir(exist_ok=True)
(exp / "report" / "REPORT_11-23-2026.tex").write_text(build_tex())
(exp / "report" / "REPORT_11-23-2026.md").write_text(build_md())
