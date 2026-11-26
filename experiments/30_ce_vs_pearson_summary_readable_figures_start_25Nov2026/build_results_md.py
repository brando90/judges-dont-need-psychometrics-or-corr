"""Builds results_11-25-2026.md from make_summary_tables.py output plus the fixed text. Usage: python3 build_results_md.py [REPO_ROOT]  (run make_summary_figures.py first)."""
import re, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parents[1])
tabs = subprocess.run([sys.executable, str(HERE / "make_summary_tables.py"), ROOT], capture_output=True, text=True, check=True).stdout
blk = {m.group(1): m.group(2).strip() for m in re.finditer(r"<!-- (\w+) -->\n(.*?)(?=\n<!-- |\Z)", tabs, re.S)}
C = "https://github.com/brando90/judges-dont-need-psychometrics-or-corr/commit/"
E28 = "../28_public_rating_sets_chaosnli_dices_frontier_judges_ce_vs_pearson_start_23Nov2026"
E26 = "../26_five_new_frontier_judges_on_fresh_homework_and_independent_recheck_start_23Nov2026"
E29 = "../29_four_objectives_lora_and_shape_control_chaosnli_dices_helpsteer2_start_23Nov2026"
E24 = "../24_trained_open_weight_judge_helpsteer2_ce_vs_pearson_start_21Nov2026"
E25 = "../25_helpsteer2_lora_scale_and_second_family_ce_vs_pearson_start_21Nov2026"
H28 = "Harness (Exp 28): judges GPT-6 Astra, Claude Opus 5.5, Kimi K3, GLM 5.3 and DeepSeek V4.1 Flash through the Stanford AI API Gateway; orchestration and analysis by Claude Sonnet 5.5 in Claude Code."
H26 = "Harness (Exp 26): the same five judges through the Stanford AI API Gateway chat completions; analysis by ub9sc in Python (numpy, scipy)."
H29 = "Harness (Exp 29): LoRA training (rank 8, top 4 blocks) of Qwen3-4B-Instruct-2507 and Llama-3.1-8B-Instruct on the mercury2 GPU box; light QA by claude-opus-5-5-high through the Antigravity CLI."
SETS = [("chaosnli_snli", "ChaosNLI SNLI"), ("chaosnli_mnli", "ChaosNLI MNLI-m"), ("chaosnli_alphanli", "alphaNLI (2 classes)"), ("dices350", "DICES-350"), ("dices990", "DICES-990")]
JUDGES = [("gpt-6-astra", "GPT-6 Astra"), ("claude-opus-5-5", "Claude Opus 5.5"), ("kimi-k3", "Kimi K3"), ("glm-5.3", "GLM 5.3"), ("deepseek-v4.1-flash", "DeepSeek V4.1 Flash")]
P29 = [("chaos3_qwen_d1", "ChaosNLI, Qwen3-4B, d=1"), ("chaos3_qwen_d5", "ChaosNLI, Qwen3-4B, d=5"), ("alpha2_qwen_d1", "alphaNLI, Qwen3-4B"), ("dices3_qwen_d1", "DICES, Qwen3-4B"),
       ("hs2_qwen_d1", "HelpSteer2, Qwen3-4B"), ("chaos3_llama_d1", "ChaosNLI, Llama-3.1-8B"), ("dices3_llama_d1", "DICES, Llama-3.1-8B")]


def img(f, alt, w="49%"): return f'<a href="figures/{f}"><img src="figures/{f}" width="{w}" alt="{alt}"></a>'


def pairs(items, left, right, alt_l, alt_r):
    """One heading and one side-by-side pair of single-panel plots per item (left file pattern, right file pattern)."""
    return "\n\n".join(f"**{name}**\n\n<p>{img(left.format(k=k), f'{name} {alt_l}')} {img(right.format(k=k), f'{name} {alt_r}')}</p>" for k, name in items)


def twos(items, pat, alt):
    out = []
    for i in range(0, len(items), 2):
        out.append("<p>" + " ".join(img(pat.format(k=k), f"{name} {alt}") for k, name in items[i:i + 2]) + "</p>")
    return "\n\n".join(out)


S28 = pairs(SETS, "single_exp28_rps_{k}.png", "single_exp28_ce_{k}.png", "held-out RPS", "held-out CE")
S28T = ", ".join(f"[{n}](figures/single_exp28_tvd_{k}.png)" for k, n in SETS)
S26 = pairs(JUDGES, "single_exp26_rps_{k}.png", "single_exp26_ce_{k}.png", "held-out RPS", "held-out CE")
S26T = ", ".join(f"[{n}](figures/single_exp26_tvd_{k}.png)" for k, n in JUDGES)
SBE = twos(SETS, "single_exp28_breakeven_{k}.png", "break-even")
S29 = pairs(P29, "single_exp29_rps_{k}.png", "single_exp29_ce_{k}.png", "held-out RPS", "held-out CE")
S29T = ", ".join(f"[{n}](figures/single_exp29_tvd_{k}.png)" for k, n in P29)
md = f"""# Corr vs CE: current results (11-25-2026) | QA update 11-26-2026

**As of 11-26-2026:** g6-H1 is corrected beside the historical hypothesis verdict; the [experiment 29 QA note]({E29}/QA_CORRECTIONS_11-26-2026.md) supplies reproducible counts.

**Related (11-25-2026):** opinion piece [The Bitter Judges Lesson: Learn AI Judges the Way You Learn Any Other ML Task](https://cs.stanford.edu/people/brando9/2026/11/25/bitter-judges-lesson.html) (the argument; this repo is the evidence).

Repo `main` at commit [143ba39]({C}143ba39) (Nov 24) when this was written. No experiment was run; every number and figure is computed from the saved results files of Experiments 26, 28 and 29 by the two scripts in this folder (`make_summary_figures.py`, `make_summary_tables.py`, built into this file by `build_results_md.py`). Every plot has grid lines on. Lower is better for RPS, CE and TVD.

## Bottom line

**CE-fit beats Pearson-fit, meaning lower held-out RPS, CE and TVD against the full human rating distribution (never higher held-out correlation).** In Exp 26 it is clean on RPS only: lower RPS in 35 of 35 judge-by-budget cells (point estimates), but lower CE in 32 of 35 and lower TVD in 32 of 35. In Exp 28 it wins on RPS in 25 of 25 set-by-judge cells from only 4 to 8 human ratings (one per item; significant), and CE and TVD agree on every set at 256 ratings. For CE-trained against Pearson-trained LoRA judges it wins in 7 of 7 panels at the largest budget in Exp 29 (RPS, CE and TVD; significant on RPS). The weaker comparisons are the raw judge with no fitting (CE-fit wins sustained only from 8 to 256 ratings, and never within 256 for 11 of 25 set-by-judge cells), Pearson-then-calibrate on DICES from 64 ratings, and the formal test in Exp 26, which passes for only 2 of 5 judges at 256 ratings. Nothing numerically new since Nov 23; Exp 29 (the Antigravity runs) is complete.

![Exp 28 held-out RPS per set](figures/fig_exp28_rps_per_set.png)

*Figure 1. Overview grid of the five single-panel plots below (Exp 28, held-out RPS, lower is better) against human ratings used to fit, per rating set. Line = mean over 5 judges, band = min to max judge, dashed = raw judge with no fit. {H28}*

## What "beat" means, exactly

- **Metric.** A beats B on a metric when A has the lower mean held-out value of that metric. The metrics are RPS (primary), CE in nats and TVD, each computed against the full empirical human rating distribution of every held-out item. Higher held-out correlation never decides a win; Spearman and Pearson r appear only as secondary columns.
- **What is compared.** A judge fit or trained with a CE objective (CE-fit, CE-LoRA) against the same judge fit or trained to maximize Pearson correlation (Pearson-fit, Pearson-LoRA), on the same data and the same budget.
- **Two strengths of "beat".** Point estimate: the mean metric is lower (this is what "35 of 35" on RPS in Exp 26 means; on CE and TVD the Exp 26 count is 32 of 35 each). Significant: the lower bound of (Pearson-fit minus CE-fit) is above 0. Exp 28 break-even: 97.5% simultaneous bootstrap bound above 0 at every tested budget from N up, against Pearson-fit and uniform. Exp 26 formal test: descending from 256 ratings, one-sided bound at alpha 0.01 per judge, stopping at the first failure. Exp 29: crossed simultaneous 95% cluster-bootstrap band above 0.
- **Unit of ratings.** The budget counts individual human ratings used to fit or train, not items. d = 1 is one rating per item, d = 5 is five per item.

## What is new since Nov 23 and what is unchanged

| Item | Status |
|---|---|
| Exp 26 and Exp 28 | Unchanged since Nov 23 (commits [570d3a9]({C}570d3a9), [5b7257f]({C}5b7257f)). |
| Exp 29 (LoRA, four objectives, Qwen3-4B and Llama-3.1-8B on mercury2) | Complete Nov 23 evening: [d6930ed]({C}d6930ed) (Qwen, 921 runs), [721edb3]({C}721edb3) (Llama, 480 runs; 1,401 total). Includes chaos3_llama, dices3_llama and hs2_qwen. |
| Exp 27 (identification preprint) | Unchanged; Stanford archive bh388th6008. |
| Newest earlier commit | [143ba39]({C}143ba39), Nov 24: README title and intro only. |
| This folder (Exp 30) | New Nov 25: readable figures, tables computed from the saved results, this file. No new experiment. |

## Table 1. Exp 28: five public rating sets, five frontier judges (mean over judges, one rating per item)

{H28} Fit = 27-transform head chosen by CE or by Pearson. Commit [5b7257f]({C}5b7257f).

At 256 ratings:

{blk['TABLE28_256']}

At 16 ratings:

{blk['TABLE28_16']}

At 4 ratings:

{blk['TABLE28_4']}

Win counts over all budgets and judge-set cells (point estimates):

{blk['WINS28']}

![Exp 28 held-out CE per set](figures/fig_exp28_ce_per_set.png)

*Figure 2. Exp 28, held-out CE in nats (lower is better), per rating set. Line = mean over 5 judges, band = min to max judge. {H28}*

![Exp 28 held-out TVD per set](figures/fig_exp28_tvd_per_set.png)

*Figure 3. Exp 28, held-out TVD (lower is better), per rating set. {H28}*

**Single-panel plots, Exp 28, one pair per rating set** (left held-out RPS, right held-out CE; lower is better; click for full size; TVD single-panel files: {S28T}). {H28}

{S28}

## Break-even and cost (Exp 28)

{blk['BREAKEVEN']}

![Exp 28 break-even](figures/fig_exp28_breakeven.png)

*Figure 4. Exp 28 break-even: fewest human ratings (one per item) from which CE-fit has lower held-out RPS than the comparator and keeps it (simultaneous 97.5% bound above 0). Each marker is one judge; x = not reached within 256. {H28}*

**Single-panel break-even plots, one per rating set** ({H28})

{SBE}

Cost, from the experiments' own cost files (estimated at the gateway list rate, Kimi K3 is listed at 0 there): Exp 28 used 11,750 judge calls for an estimated 19.69 USD (22.55 USD if priced at the 1.4 and 4.4 USD per million token reference); Exp 26 used 800 calls for an estimated 1.34 USD (1.20 USD at the reference). The human side is the cost that matters: CE-fit needs 4 to 8 human ratings per rating set to beat Pearson-fit and uniform.

## Table 2. Exp 26: fresh homework answers, 160 answers, five frontier judges (256 ratings, two per answer, mean of ten folds)

{H26} Commit [570d3a9]({C}570d3a9). Uniform: RPS 0.2497, CE 1.792, TVD 0.767.

{blk['TABLE26'].split(chr(10)+chr(10)+'Formal status')[0]}

Formal status by judge and budget (the test stops at the first failure; later budgets are then not tested):

{blk['TABLE26'].split('Formal status by judge and budget:')[1].strip()}

Under the definition of "beat" above, Exp 26 is a clean win on RPS only. Claude Opus 5.5 at 256 ratings has TVD 0.334 for CE-fit against 0.309 for Pearson-fit (Pearson-fit lower). Spearman and Pearson r are not in this experiment's tables.

![Exp 26 held-out RPS per judge](figures/fig_exp26_rps_per_judge.png)

*Figure 5. Overview grid (single-panel plots below). Exp 26, held-out RPS (lower is better) against human ratings used to fit (two per answer), per judge; mean of 10 folds, band = 95% interval, dashed = raw judge with no fit. {H26}*

![Exp 26 held-out CE per judge](figures/fig_exp26_ce_per_judge.png)

*Figure 6. Exp 26, held-out CE in nats. {H26}*

![Exp 26 held-out TVD per judge](figures/fig_exp26_tvd_per_judge.png)

*Figure 7. Exp 26, held-out TVD. {H26}*

**Single-panel plots, Exp 26, one pair per judge** (left held-out RPS, right held-out CE; lower is better; TVD single-panel files: {S26T}). {H26}

{S26}

## Table 3. Exp 29: LoRA fine-tuning, raw CE-trained against raw Pearson-trained, largest budget

{H29} The repo does not name the builder model for Exp 29. Commits [d6930ed]({C}d6930ed) and [721edb3]({C}721edb3). Means over 10 draws; hs2_qwen reuses Exp 24 predictions.

{blk['TABLE29']}

![Exp 29 held-out RPS curves](figures/fig_exp29_rps_lora_curves.png)

*Figure 8. Overview grid (single-panel plots follow). Exp 29, held-out RPS (lower is better) against human ratings, raw (uncalibrated) LoRA judges, mean over 10 draws. {H29}*

![Exp 29 held-out CE curves](figures/fig_exp29_ce_lora_curves.png)

*Figure 9. Exp 29, held-out CE in nats. {H29}*

![Exp 29 held-out TVD curves](figures/fig_exp29_tvd_lora_curves.png)

*Figure 10. Exp 29, held-out TVD. {H29}*

**Single-panel plots, Exp 29, one pair per panel** (left held-out RPS, right held-out CE; lower is better; TVD single-panel files: {S29T}). {H29}

{S29}

The repo's own Exp 29 figures: [four objectives and K=2 control]({E29}/figures/fig1_four_objectives_and_k2_control.png), [disagreement slices and logit rescue]({E29}/figures/fig2_disagreement_slices_and_logit_rescue.png). Pre-registered hypotheses (Exp 29): H1 confirmed (raw CE and RPS beat raw Pearson in 7 of 7 panels) (11-25 text; corrected 11-26-2026: **the registered across-budget H1 is only partly supported**; 7/7 is the narrower largest-budget result, while three-category RPS wins are CE-trained 26/27 point and 22/27 crossed, RPS-trained 24/27 point and 17/27 crossed); H2, H3 and H4 partial; details in [RESULTS_11-23-2026.md]({E29}/RESULTS_11-23-2026.md).

## Exp 24 and Exp 25 (HelpSteer2 Qwen3-4B LoRA, complete Nov 21 and 22)

| Exp | Result at largest budget | Harness / commit |
|---|---|---|
| 24 | Raw CE RPS 0.15213 against Pearson 0.18376; calibrated CE 0.15200 against Pearson 0.15309 (9 of 10 draws) | Qwen3-4B LoRA; [54de212]({C}54de212) |
| 25 | Raw CE wins 20 of 20 draws (Qwen) and 10 of 10 (Llama-3.1-8B) at 64 or more ratings; calibrated Qwen converges (0.15080 against 0.15089), calibrated Llama tied (0.16005 against 0.15893) | Qwen3-4B and Llama-3.1-8B LoRA, built by Antigravity (Gemini 4 Carbon); [59e97d4]({C}59e97d4) |

![Exp 24 learning curves]({E24}/figures/helpsteer2_lora_learning_curves.png)

*Figure 11. Exp 24, HelpSteer2 LoRA learning curves (the repo's own figure).*

## Caveats (read before quoting)

- The derivation (Pearson sees only the first moment; CE and RPS are proper) is an infinite-data result. "CE is better with limited data" is supported empirically in the cells above, not derived.
- Exp 28 and 26 are frontier judges with calibration heads; Exp 29 is LoRA training. They answer slightly different questions.
- Against the raw judge, CE-fit wins sustained only from 8 to 256 ratings in 14 of 25 set-by-judge cells (one rating per item); on ChaosNLI the raw frontier judge is already strong. The sets are chosen for human disagreement.
- Pearson-then-calibrate beats CE-fit on DICES from 64 ratings (mean RPS at 256: DICES-350 0.221 against 0.226; DICES-990 0.194 against 0.198) and on MNLI-m (0.161 against 0.163) in Exp 28; Spearman and Pearson r are close between CE-fit and Pearson-fit, so correlation alone barely separates them.
- Older judge-pool split (Nov 21): 15 native-probability judges favored CE selection, 44 old 3-vote judges favored Pearson selection.
- Exp 26: at 4 ratings CE is not significantly better for 4 of 5 judges.
- Prior exposure in Exp 29: 265/713 (chaos3), 110/412 (alpha2) and 172/300 (dices3) test items were scored by frontier judges in Exp 28; the LoRA never trained on them.
"""
(HERE / "results_11-25-2026.md").write_text(md)
(HERE / "README.md").write_text("# Experiment 30: CE vs Pearson summary with readable figures (started 11-25-2026)\n\nNo new experiment: tables and plots computed from the saved results of Experiments 26, 28 and 29. Read [results_11-25-2026.md](results_11-25-2026.md). Rebuild: `python3 make_summary_figures.py ../..` then `python3 build_results_md.py ../..` (needs numpy, matplotlib).\n")
print(len(md))
