# Exp 31: scaling laws for judges, CE-fit vs Pearson-fit (RPS, CE, TVD against human ratings) | 11-25-2026

**Related (11-25-2026):** opinion piece [The Bitter Judges Lesson: Learn AI Judges the Way You Learn Any Other ML Task](https://cs.stanford.edu/people/brando9/2026/11/25/bitter-judges-lesson.html) (the argument; this repo is the evidence).

Run by ultimate_brando9 for Brando Miranda, task AT-3569. Free CPU re-analysis of the saved results of Exp 26, 28 and 29 of this repo (no model calls, no spend). It answers Brando's Sanmi-meeting note of Nov 25, 11:00 AM ("we should be doing scaling laws for judges, it's not special in any way") and his Nov 11 line ("just ce and lots of data scaling laws as always"). Script: `fit_scaling_laws.py`; tables and this file: `build_results_md.py`; numbers: `results/*.csv`.

## Bottom line

Held-out RPS, CE and TVD of CE-fit and CE-LoRA judges fall with the number of human ratings N as an ordinary saturating power law, error = floor + a * N^-b, with exponent b about 0.6 to 0.9: median b over the Exp 28 judge-by-set cells with an identified floor is 0.66 for RPS, 0.82 for CE and 0.59 for TVD (1 rating per item), and 0.64 to 0.89 for CE-LoRA on the three Exp 29 panels with eight budgets (ChaosNLI Qwen3-4B 0.89 [0.76, 0.99], HelpSteer2 Qwen3-4B 0.76 [0.64, 0.89], ChaosNLI Llama-3.1-8B 0.64 [0.49, 0.78]). At b = 0.66 each doubling of N removes about 37% of the error still above the floor. The CE floors sit well under uniform (RPS floor 0.46 to 0.83 of uniform RPS across the five Exp 28 sets, 0.69 to 0.76 across the Exp 29 panels). Pearson-fit and Pearson-LoRA curves are not described by this law: they are still falling at the largest N tested with a log-like decline (b at or under 0.1, floor not identified), flat (DICES-990 stays at 0.26 RPS against uniform 0.27), or non-monotone (all 25 judge-by-set cells with five ratings per item). At the largest N tested the Pearson-fit sits above the CE-fit floor in 20 of 25 (RPS), 25 of 25 (CE) and 21 of 25 (TVD) Exp 28 cells (1 rating per item, 95% bound). So the Pearson-fit curve has not reached where the CE curve saturates, and we cannot say from the data where, or whether, it saturates. Exp 26's saved intervals are too wide to identify any law (every judge is "flat within noise"). Not answerable here: the human leave-one-vote-out RPS floor (the labels are sealed and not in the repo, so floors are compared with uniform and the raw judge), any judge-size axis within one model family, and anything beyond N = 256 (Exp 26, 28) or 2,048 (Exp 29).

**"Beat" and "fits" here.** Beat = lower held-out RPS, CE and TVD against the full human rating distribution; correlations are secondary columns only. "Fits" = a saturating power law with no significant rise, a decline beyond noise, and an RMS residual of at most 10% of the curve's range (rules in Method).

## Reference values

Uniform forecast and raw judge (no fit) per panel. N is the number of human ratings used to fit.

{{REFS}}

## Table 1. Fitted law, held-out RPS (lower is better)

error = floor + a * (N / N_min)^-b, floor and a at least 0, b between 0.02 and 3; intervals are 95% parametric bootstrap. "at 0 bound" or "at 0.02 bound" means the fit sits on the edge of the allowed range: that curve has no identified floor.

{{TAB_RPS}}

## Table 2. Fitted law, held-out CE in nats (lower is better)

{{TAB_CE}}

## Table 3. Fitted law, held-out TVD (lower is better)

{{TAB_TVD}}

## Table 4. Is the Pearson curve at the CE floor? (floor comparison per panel)

Columns give the CE-fit floor, the Pearson-fit value at the largest N tested, and the gap between them with a 95% bootstrap interval; **>0** = the Pearson curve is still above where CE saturates at the largest N. The fitted Pearson floors are not compared directly because most sit on the 0 bound (Tables 1 to 3). Last column: whether the CE-fit floor's upper 95% bound is below the raw judge's value.

{{PAIRS}}

## Table 5. Correlations (secondary columns)

At the largest N tested. CE-fit and Pearson-fit differ by less than 0.05 in every row, so correlation barely separates them; RPS, CE and TVD do (Tables 1 to 4).

{{SECONDARY}}

## Table 6. All 25 judge-by-set cells of Exp 28 (not only the set means)

Same fit per judge and set cell; the full per-cell numbers are in `results/fits_11-25-2026.csv` and `results/floor_comparisons_11-25-2026.csv`.

{{CELLS}}

## Where a power law does not fit

- **Pearson-fit and Pearson-LoRA, mostly a slow log-like decline.** In four of five Exp 28 sets (SNLI, MNLI-m, alphaNLI, DICES-350) and in Exp 29 on ChaosNLI Qwen (b 0.09) and HelpSteer2 Qwen (b 0.05) the best fit has b at or under 0.1 and a floor near 0. No RPS floor near 0 is plausible for human ratings that disagree, so these are extrapolations, not ceilings, and the curve has not saturated by the largest N.
- **Pearson-LoRA is flat, then drops.** HelpSteer2 Qwen RPS: 0.246, 0.233, 0.239, 0.221, 0.221, 0.218, 0.197, 0.184 at N = 16 to 2,048. The drop starts after N = 512 and is still going at 2,048, which a monotone power law with a floor cannot describe.
- **Pearson-fit with five ratings per item is non-monotone in all 25 cells.** The smallest budget (2 items) is better than the next one: ChaosNLI SNLI with Claude Opus 5.5 has RPS 0.110 at N = 10, 0.145 at N = 20 and 0.128 at N = 320; the raw judge scores 0.109 in that cell. This is consistent with the fit staying near the raw judge when it has only two items (my inference, not tested).
- **Pearson-fit is flat on DICES-990** (RPS 0.26 from N = 4 to 256, no decline beyond noise) and on most TVD cells with one rating per item (16 of 25).
- **CE curves with an early steep drop and noisy middle.** CE-LoRA on HelpSteer2 Qwen (RMS residual 12% of its range) and ChaosNLI Llama (10.5%) is marked poor fit: a large fall between N = 16 and 64, then small steps (ChaosNLI Qwen RPS 0.172 at N = 64, 0.171 at 128, 0.155 at 256, 0.156 at 512).
- **CE-fit floor not identified in 17 of 25 Exp 28 cells for RPS (one rating per item).** The curve fits but is still bending at N = 256, so its floor interval reaches 0.
- **Exp 26.** All ten curves per metric are "flat within noise": the saved fold-level standard errors are about 0.01 (95% intervals about plus or minus 0.02), against a total decline of about 0.02 for CE-fit. The point estimates decline smoothly (GPT-6 Astra CE-fit RPS 0.0848, 0.0796, 0.0741, 0.0705, 0.0678, 0.0664, 0.0660 at N = 4 to 256), so the shape is plausible, but the intervals cannot identify b or the floor.
- **Four-budget panels (Exp 29 alphaNLI, both DICES) have one degree of freedom,** so their fits are indicative only; the five-per-item ChaosNLI Qwen panel has three budgets and is not fit.

## What the data cannot answer

- **Human floor.** The leave-one-vote-out human RPS needs the per-item vote counts, which are sealed and not in the repo; Exp 29's saved all-dev histogram is not a floor (it ignores the item). Floors are compared with uniform and with the raw judge. A judge can only be said to be near the human limit once that value is computed on the labels, which is free CPU work for whoever holds them.
- **Judge-size axis.** In Exp 26 and 28 the five judges are five different model families with no size ladder, so exponent against size cannot be fit. Exp 29 has Qwen3-4B and Llama-3.1-8B only: two points, and family and size are confounded.
- **Range.** N stops at 256 ratings (Exp 26, 28) and 2,048 (Exp 29). Floors are extrapolations beyond the last point; Pearson curves in particular have not stopped falling.
- **Independence.** The bootstrap treats the points of a curve as independent. Points share test items and nested draws, so shape noise is probably overstated and level noise understated. For Exp 28, the Pearson-fit SE is approximated as the square root of the squared CE-fit SE plus the squared SE of the Pearson-minus-CE difference (the file does not hold the Pearson SE alone); set means use the mean of the five judges' SEs (the fully correlated case).
- **Higher moments.** The fits give the sample size at which RPS, CE and TVD saturate for these judges. They do not separately show how many ratings it takes to learn a second or third moment of the rating distribution.

## Method

- **Inputs, saved results only.** Exp 28: `scores_absolute.csv` (per set, judge, ratings per item d, budget B, method), SE from `contrasts.csv` (the paired test-item bootstrap SE of the uniform-minus-CE difference, which equals the CE-fit SE because uniform has no fit noise), raw judge from `raw_judge_scores.csv`, correlations from `correlations.csv`. Exp 26: `results/new5/scores.csv` (mean and 95% interval, 2 ratings per answer main, 1 per answer in the CSV only). Exp 29: `results/scores.csv`, raw LoRA methods `ce_raw` and `pearson_raw`, SE = draw SD / sqrt(10 draws), N = `b_total`.
- **Fit.** Weighted least squares with weights 1/SE^2, exact over a grid of 150 values of b from 0.02 to 3 (floor and a solved in closed form, both at least 0); N is scaled by the smallest N, so a is the excess at the smallest N.
- **Intervals.** 2,000 parametric bootstrap draws per curve, each point Normal(value, SE^2), refit, percentile 95%. The plot band is the same bootstrap.
- **Verdicts.** non-monotone = any step up larger than 1.96 times the SE of the two points; flat = decline from first to last point under 2 SEs; poor fit = RMS residual above 10% of the first-to-last range; floor not identified = floor interval wider than that range; b at or under 0.2 adds "floor is an extrapolation"; 4 points = indicative; 3 points = not fit.
- **Figures.** One single-panel plot per curve pair and metric in `figures/` (66 files, `fit_<exp>_<metric>_<panel>.png`); grid on in every plot; the 5-ratings-per-item Exp 28 plots and Exp 26 single-rating plots are included as files and in the CSVs.
- **Reproduce.** `python3 fit_scaling_laws.py` (about 2 minutes on 4 CPUs; numpy and matplotlib), then `python3 build_results_md.py > results_11-25-2026.md`.

## Figures

Each figure caption gives the harness and model.

{{PLOTS}}
