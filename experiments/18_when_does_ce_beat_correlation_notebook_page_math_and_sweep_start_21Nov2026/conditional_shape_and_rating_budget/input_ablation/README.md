# What can a judge infer from each of the notebook's four input sets? · 11-21-2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/18_when_does_ce_beat_correlation_notebook_page_math_and_sweep_start_21Nov2026/conditional_shape_and_rating_budget/input_ablation/README.md>

Date: 11-21-2026

**TLDR:** In this eight-state synthetic example, hiding the reference leaves the ideal predicted mean unchanged on every case. Pearson stays at 1 while the average total variation distance to the full conditional rating distribution rises from 0 to 0.12. Conversely, knowing the candidate and reference gives a constant expected rating but predicts some of the variation in the *shape* of the distribution.

## Exact population scores

Here $x$ represents the task, $c$ the candidate $\hat y$, and $g$ the reference $y^\star$. A judge sees only the inputs named in its row. Each row predicts the true conditional rating probabilities given that information. The expectation covers all eight equally likely synthetic artifacts and all six possible rating outcomes, weighted by their true probabilities. There is no sampling, fitting, validation, prior human exposure, or held-out human test in this calculation. Intervals and p-values do not apply.

| Inputs available to the population oracle | Cross-entropy (CE), nats ↓ | Normalized ranked probability score (RPS) ↓ | Total variation distance (TVD) to full conditional truth ↓ | Pearson vs. full conditional means ↑ |
|---|---:|---:|---:|---:|
| Task, candidate, reference | 1.71513 | 0.184830 | 0.000 | 1.000 |
| Task, candidate | 1.76354 | 0.187710 | 0.120 | 1.000 |
| Task | 1.77458 | 0.190249 | 0.141 | 0.789 |
| Candidate, reference | 1.74757 | 0.191564 | 0.081 | Undefined (constant mean) |
| No inputs, constant marginal forecast | 1.79176 | 0.194444 | 0.157 | Undefined (constant mean) |

CE is the expected negative natural logarithm of the probability assigned to the observed rating, from zero to infinity. RPS averages the squared error of cumulative probabilities across the five thresholds on the 0–5 scale and ranges from zero to one. TVD is half the sum of the absolute differences between the two six-category probability vectors and ranges from zero to one. Here TVD uses the known full-input truth. Pearson in this table is the correlation, over the eight artifacts, between the oracle's expected rating and the full-input expected rating, with range −1 to 1 when defined. An undefined value indicates that the oracle predicts the same mean on all artifacts. The full-input row gives the irreducible CE and RPS. [Machine-readable scores](population_scores.csv)

## Construction and checks

Take $x,c,g$ to be independent fair signs. For ratings $r=0,\ldots,5$, use the full conditional probabilities

$$
p(r\mid x,c,g)=\tfrac16 +(0.045x+0.035xc)u_r+0.06cg\,v_r,
$$

where $u=(-1,-0.6,-0.2,0.2,0.6,1)$ and $v=(1,0,-1,-1,0,1)$. The task and the task–candidate interaction determine the mean $2.5+7(0.045x+0.035xc)$. The candidate–reference interaction shifts mass between central and extreme ratings while preserving the mean. Each forecast for partially observed inputs is the average full distribution over all compatible hidden-input assignments. Exhaustive marginalization yields the population minimizer for CE and RPS given those inputs. There is no optimizer, initialization, tie selection, or search budget.

Hiding $g$ removes the $0.06cg\,v$ term. Its half absolute sum is exactly 0.12. Its cumulative differences are proportional to $(1,1,0,-1,-1)$, so the expected RPS rises by exactly $4(0.06)^2/5=0.00288$. Seeing only $c,g$ retains the shape term while averaging away the mean signal. This explains why that row improves both CE and RPS on the constant baseline despite its undefined Pearson correlation. CE favors the candidate-and-reference oracle over the task-only oracle in this example, whereas RPS favors the task-only oracle. Their input sets are incomparable, and the two scores weigh their errors differently.

The [enumeration script](enumerate_inputs.py) checks positivity, normalization, mean preservation, all five RPS excesses against the cumulative-squared-error identity, and the analytic Pearson value for the task-only oracle. All probabilities and RPS/TVD calculations use exact rational arithmetic. Logarithms, square roots and printed scores use floating point. Run `python3 enumerate_inputs.py` from this directory to regenerate the table.

## What this establishes for the project

The existing open-weight judge prompt explicitly supplies the Python task, candidate and gold reference. I searched this repository's experiment code and reports, its paper sources, and the sister repository's experiment documentation for reference omission and input-mask comparisons. I found the all-input prompt, reference-file provenance checks, and a separate restriction on what *solvers* receive. I did not find a matched study of these four input sets against real human ratings. [Open-weight judge prompt](../../../09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026/scripts/build_prompts.py) · [Frontier input provenance audit](../../../10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026/README.md) · [Solver packet instructions](https://github.com/brando90/trustworthy-theorem-judges/blob/main/experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026/support/COORDINATOR_README.md)

The calculation isolates the predictive information in each input under the specified law. It does not estimate how much the actual Lean reference helps a judge predict expert ratings or how many training ratings either objective requires. The parent experiment separately measures finite-sample fitting with matched information.
