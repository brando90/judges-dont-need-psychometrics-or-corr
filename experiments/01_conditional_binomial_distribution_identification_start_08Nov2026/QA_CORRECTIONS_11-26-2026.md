# Experiment 01: verified QA corrections (11-26-2026)

**Date: 11-26-2026.** Codex CLI, GPT-6 Astra ultra (Vals account). CPU Python/raw-output checks; no Lean checking, no new model calls. Original protocols, result tables and raw outputs remain preserved.

**g1-M1 — CONFIRMED, sampled test ratings.** Old reading-guide text: “with no sampled test ratings.” Corrected 11-26-2026: the main test losses are exact expectations against the known distribution, and separate sampled-test diagnostics also exist. There are 9 test trace examples and 6,912 candidate rows with sampled-test cross-entropy (CE). Test ratings do not enter fitting or selection. The original guide sentence is retained as historical wording.

**g1-L1 — CONFIRMED, integration status.** Old text: “main publication pending” / “PENDING.” Corrected 11-26-2026: paper integration was completed on 11-17-2026 in commits `3bfd866` and `18f529e`; this does not establish completion of every requested external review or acceptance/publication by a venue.

**g1-L4 — CONFIRMED, reproduction paths.** Old commands used `experiments/02_conditional_binomial_distribution_identification/` (absent). Corrected 11-26-2026 commands, from the repository root:

```bash
python3 -m pytest -q -p no:cacheprovider experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/test_study.py experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/test_validated_scorer.py
python3 experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/verify_results.py
python3 experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/make_report.py
```

The final two commands are historical reproduction entry points that write their usual outputs; phase 2 did not run them against preserved evidence.

**g1-L5 — CONFIRMED, sampled-test metric panel.** Old table: Pearson, Spearman and Kendall before any proper score. Corrected 11-26-2026: use the matched sampled-test panel below. Ranked probability score (RPS) is the mean squared cumulative probability error over five thresholds [0,1], lower better; CE is mean negative log probability of sampled individual ratings (nats), lower better. Total variation distance (TVD) was not recorded for this target. Correlations compare forecast means with the same sampled item-rating means, range [-1,1], higher better. All 21 criterion/rating-count groups use the same 256 replications, 48 held-out test items and equal weighting across selected ties; these rows were reconstructed from candidate and selection CSVs, not substituted from population scores.

**g1-M3 — historical evidence caveat (11-26-2026).** Historical human-data intervals repeated in the full report remain aggregate-only; see experiment 02's dated correction note. This is missing reconstruction evidence, not a demonstrated wrong estimate.

## Recomputed evidence and matched metric panel (11-26-2026)

Date: 11-26-2026. Read-only CPU reconstruction; no model calls.

g1-M1: test trace records = 9 ; candidate rows with sampled-test CE = 6912
g1-L1: integration commits
3bfd866 2026-11-17 Paper & one-pager: close SK feedback points 1-5, dual AAAI/arXiv build (~1.5x appendix + S1-S4 supp), sync Rule 89
18f529e 2026-11-17 Add 11-17 weekly update, AAAI-27 Exp 09/10/04 integration, and v25 Sanmi one-pager
g1-L4: historical command directory exists: False ; corrected directory exists: True
g1-L5: selected sampled-test scores, 256 replications; equal average over ties. TVD not recorded for this target.

| Dataset / judge | RPS | CE (nats) | TVD | Spearman | Pearson |
|---|---:|---:|---:|---:|---:|
| 1 ratings/item; brier selected | 0.104050707 | 1.331926262 | not recorded | 0.781759721 | 0.791991024 |
| 1 ratings/item; ce selected | 0.103938299 | 1.329865408 | not recorded | 0.781759721 | 0.791991024 |
| 1 ratings/item; kendall selected | 0.133588390 | 1.984490520 | not recorded | 0.781759721 | 0.791991024 |
| 1 ratings/item; nmse selected | 0.133588390 | 1.984490520 | not recorded | 0.781759721 | 0.791991024 |
| 1 ratings/item; pearson selected | 0.133588390 | 1.984490520 | not recorded | 0.781759721 | 0.791991024 |
| 1 ratings/item; rps selected | 0.104084169 | 1.332631617 | not recorded | 0.781759721 | 0.791991024 |
| 1 ratings/item; spearman selected | 0.133588390 | 1.984490520 | not recorded | 0.781759721 | 0.791991024 |
| 2 ratings/item; brier selected | 0.104009403 | 1.328960642 | not recorded | 0.864226637 | 0.878114143 |
| 2 ratings/item; ce selected | 0.104009403 | 1.328960642 | not recorded | 0.864226637 | 0.878114143 |
| 2 ratings/item; kendall selected | 0.133860619 | 1.988252929 | not recorded | 0.864226637 | 0.878114143 |
| 2 ratings/item; nmse selected | 0.133860619 | 1.988252929 | not recorded | 0.864226637 | 0.878114143 |
| 2 ratings/item; pearson selected | 0.133860619 | 1.988252929 | not recorded | 0.864226637 | 0.878114143 |
| 2 ratings/item; rps selected | 0.104009403 | 1.328960642 | not recorded | 0.864226637 | 0.878114143 |
| 2 ratings/item; spearman selected | 0.133860619 | 1.988252929 | not recorded | 0.864226637 | 0.878114143 |
| 3 ratings/item; brier selected | 0.103692706 | 1.326472497 | not recorded | 0.898856683 | 0.913626168 |
| 3 ratings/item; ce selected | 0.103692706 | 1.326472497 | not recorded | 0.898856683 | 0.913626168 |
| 3 ratings/item; kendall selected | 0.133563126 | 1.986653684 | not recorded | 0.898856683 | 0.913626168 |
| 3 ratings/item; nmse selected | 0.133563126 | 1.986653684 | not recorded | 0.898856683 | 0.913626168 |
| 3 ratings/item; pearson selected | 0.133563126 | 1.986653684 | not recorded | 0.898856683 | 0.913626168 |
| 3 ratings/item; rps selected | 0.103692706 | 1.326472497 | not recorded | 0.898856683 | 0.913626168 |
| 3 ratings/item; spearman selected | 0.133563126 | 1.986653684 | not recorded | 0.898856683 | 0.913626168 |


Reproduce the evidence with `python3 docs/qa_11-25-2026/phase2/verify_g1_g2_remaining.py 1` from the repository root. HIGH-item checks are separately saved as `verify_g1-H1.py` (experiment 05), `verify_g2-H1.py` / `verify_g2-H2.py` (experiment 11), and `verify_g2-H3.py` (experiment 07), with their printed `.txt` outputs.
