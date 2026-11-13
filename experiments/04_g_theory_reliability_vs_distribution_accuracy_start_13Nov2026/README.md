# 04: Generalizability-theory reliability versus human-rating distribution accuracy

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/README.md>

**TLDR:** Test whether generalizability-theory reliability coefficients or human-variance matching are sufficient to identify an accurate human-rating judge. Separately test whether variance decomposition improves evaluation-design choices; a counterexample to sufficiency would not refute that use of the theory.

**Status:** PLANNED — source reading and experimental design completed on 11-13-2026; no simulations, judge fitting or human-data analysis run. The start date records the beginning of this planning work.

**Goal:** Find the strongest defensible counterexample to treating aggregate reliability as sufficient evidence of judge trustworthiness, and identify what generalizability theory adds beyond direct predictive evaluation.

**Key uncertainty:** Does a proposed benefit concern prediction of human ratings, or precision of an aggregate score under repeated sampling? These are different targets. This experiment studies the published framework and synthetic examples, not any unpublished study or slide deck.

## Repository context

This repository asks how to choose and train a judge that predicts a specified population's conditional human-rating distribution. Its supported thesis is narrower than its title: scalar correlation alone does not identify that distribution. Proper probability losses address distributional prediction; they do not by themselves establish construct validity, representative human sampling or stable benchmark rankings.

- [00_synthetic_pearson_vs_kl_squashed_judge_0to5_start_07Nov2026](../00_synthetic_pearson_vs_kl_squashed_judge_0to5_start_07Nov2026/README.md): the historical marginal-histogram illustration, not conditional identification evidence.
- [01_conditional_binomial_distribution_identification_start_08Nov2026](../01_conditional_binomial_distribution_identification_start_08Nov2026/README.md): controlled conditional-distribution study; scalar mean-based criteria cannot distinguish its same-mean forecasts. This motivates testing whether adding variance information resolves the ambiguity.
- [02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026](../02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/RESULTS_comparison.md): the completed matched human-rating comparison does **not** establish a cross-entropy advantage. Its [checkpoint](../02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/ckpt.md) was inspected before this plan. Older instructions call this experiment 03; use its current path, and preserve its frozen evidence.
- [trustworthy-theorem-judges](https://github.com/brando90/trustworthy-theorem-judges) owns practical judge development, human-rating collection and sample-size planning. This study adds no competing collection, participant assignment or collaborator email.

Prospective human-rating comparisons retain normalized individual-rating **ranked probability score (RPS)** as primary, with **cross-entropy (CE)** and **total variation distance (TVD)** secondary; see the [current definitions](../02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/EVALUATION_CRITERION_11-09-2026.md). Existing experiments keep their original endpoints. No paper claims or results are changed by creating this folder.

The user requested the identifier `04_`; it was available locally and on `origin/main`. Existing experiments were not renumbered, and the gap at 03 is intentional.

## What the supplied paper actually says

Shavelson, Webb and Rowley, **Generalizability Theory**, *American Psychologist* (1989), describes:

- A **generalizability study (G study)**: estimate variation attributable to objects, raters, tasks, occasions and their interactions, as appropriate to the design.
- A **decision study (D study)**: use those estimates to predict precision for a proposed measurement design, such as more tasks versus more raters.
- **Relative decisions**: distinguish or rank objects; **absolute decisions**: interpret their score levels. Different variance components count as error for these decisions.

G and D name studies, not a universal pair of coefficients. No contemporary implementation is being reproduced or evaluated here. The paper itself warns about restricted generalization, dependence, unstable variance estimates and statistically optimized weights that lack conceptual justification. See [SOURCES.md](SOURCES.md) for page-specific evidence and reading scope.

## Planned tests, in priority order

1. **Same mean and variance, different distributions.** Construct distinct positive categorical forecasts with identical first two conditional moments; test whether second-moment reliability summaries tie while expected proper losses distinguish them.
2. **Repeatable but wrong.** Test task-dependent deterministic errors and a shared score offset. Distinguish absolute reliability within a measurement system from absolute agreement with an external human target.
3. **A correct distribution need not maximize a sampled-score reliability coefficient.** Compare predicting the human distribution with sampling a rating from it; do not confuse human disagreement with instability of the probability forecast.
4. **A fair positive control for D studies.** Compare cost-matched designs under known variance components. Then separately test dependence and population shift, which the paper already flags as limitations.
5. **Optional retrospective application.** Only if existing ratings identify the needed variance components, compare diagnostics on authorized historical data. This cannot establish fresh human generalization.

**Decision criterion:** A verified counterexample rejects only the corresponding sufficiency claim. Retain the D-study use case if its design predictions work under their assumptions. Stop a broad “G theory is unnecessary” claim if the results support complementarity instead.

## Files and continuation

```text
README.md                   Context, scope and planned tests
PLAN.md                     Definitions, controls, proposed design and verdict rules
SOURCES.md                  Paper identity, page map and attribution boundaries
source_manifest.json        Supplied PDF retrieval details and SHA-256 digest
results.md                  Status; no experimental results yet
CKPT_g_theory_plan.md        Resumable planning checkpoint
sources/local/              Ignored published-paper reading copy and extraction
```

Next implementation: follow [PLAN.md](PLAN.md), beginning with the finite-support moment counterexample and independent formula tests. Use local deterministic arithmetic before any larger simulation. Future numerical claims must be generated by scripts in this folder; no numerical outcome is inferred from this plan.

**TLDR-end:** [judge objectives: 04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026] The experiment tests the limits of reliability as evidence of human-rating accuracy, while giving variance-based evaluation design a fair positive control. Planning is complete; the experimental verdict is pending.
