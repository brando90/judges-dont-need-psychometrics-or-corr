# Experiment 39: Proper-score design versus D-study budget allocation

**Doc link:** <https://docs.google.com/document/d/1WtzpWNuUbkf3eGLOERMIvcyI7Sjec1hLfIZ1rJKfKvE/edit>

**TLDR:** A CPU head-to-head of proper-score design and G/D-theory allocation on ChaosNLI SNLI, MNLI-m, and alphaNLI, and on DICES-350 and DICES-990, plus the harder Phase B of experiment 04. The two targets do not share the same optimal split of a fixed rating budget, and the corrected effects are small. Proper-score design does not replace D-theory. **Corrected 11-27-2026:** the first version of Study A overstated the proper-score selector; see the [v2 results](v2_matched_budget_corrected_11-27-2026/RESULTS_V2.md) and the [correction note](CORRECTION_NOTE_EXP39.md). Results: [results.md](results.md). Reading guide: [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md).

**As of:** 11-27-2026. Started 11-26-2026. Local CPU, `$0`, no model calls. Protocol frozen before measurement.

## Question

Once a learner is trained and evaluated with a proper score, do we still need a generalizability study (G-study) or a decision study (D-study) to allocate a fixed rating budget $B = n_{\mathrm{items}} \times n_{\mathrm{raters}}$?

The two official targets under test are:

1. A distributional forecast of new-item $p(y\mid x)$, scored by held-out test ranked probability score (RPS), with cross-entropy (CE) and total variation distance (TVD) as secondary checks.
2. Precision of a mean rating (per-item universe score, and separately the grand mean).

Experiment [04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026](../04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/README.md) showed that reliability is not sufficient for distribution accuracy, and that D-study allocation works under its assumptions, but left the harder close-design Phase B unrun. Experiment [34_g_d_theory_vs_ce_where_reliability_agrees_and_fails_start_25Nov2026](../34_g_d_theory_vs_ce_where_reliability_agrees_and_fails_start_25Nov2026/README.md) showed where the coefficients agree with cross-entropy and where a perfectly repeatable forecast still loses on RPS. This experiment asks whether a proper-score selector and a D-study selector pick the same $(n_{\mathrm{items}}, n_{\mathrm{raters}})$ split, on public multi-annotator data and on the missing Phase B grid.

## Evaluation setup

Study A uses the Experiment 28 27-transform CE-fit of five frozen frontier-judge probabilities (no new application programming interface (API) calls). A frozen 30% test split is never used to choose a design or a transform. U1 is the per-item D-study (always more raters). U2 is the grand-mean D-study (more items when $\widehat{v}_{u}>0$). The proper-score selector minimizes expected held-out RPS by 5-fold $\times$ 8 inner draws on an 80-item, 16-rating pilot. 150 replications. Candidate designs: $n_{i}\in\lbrace 16,32,64\rbrace$, $n_{r}\in\lbrace 2,4,8,16\rbrace$. In v1, “beats” meant a lower mean and a paired Student-$t$ 95% interval that excludes 0. In v2 it means a lower mean and a Holm-adjusted two-way (replication × test item) bootstrap p-value below 0.05.

Study B matches the Experiment 04 Phase B crossed G-study, interpolates $v(t)=(1-t)\mathrm{C1}+t\mathrm{C2}$, and asks how often constrained residual maximum likelihood (REML) mis-selects among $8\times 2$, $4\times 4$, and $2\times 8$ as the facet-mixing parameter $t$ varies. 200 replications (256 at the two reproduction cells).

## Result (corrected 11-27-2026)

Study A was re-run after an independent review ([v2 results](v2_matched_budget_corrected_11-27-2026/RESULTS_V2.md)). On held-out RPS, the most-items rule has a lower mean than the most-raters rule in 18 of 20 dataset-budget cells, significant after Holm correction in 5 of 20 and at most 3.0% of total RPS. A pilot-based proper-score selector beats the most-raters rule in 2 of 20 cells after Holm and never beats the most-items rule; a pilot D-study with a reliability target never beats the most-items rule either. All six pre-declared predictions held; the decision rules under which a pilot would add value (D2, D3, D4) were not met. U1 and U2 are analytic corner rules at a fixed budget, so their item-mean and grand-mean MSE orderings (20 of 20 and 18 of 20 cells) follow from the formulas, not from a data-driven choice. The first version's headline (proper-score selection beats U1 in 10 of 20 cells) is withdrawn; its tables remain in [results.md](results.md) marked "v1". Figures: [v2 fig1](v2_matched_budget_corrected_11-27-2026/figures/fig1_v2_test_rps_by_selector.svg), [v2 fig2](v2_matched_budget_corrected_11-27-2026/figures/fig2_v2_paired_gains_holm.svg); Study B: [fig2](figures/fig2_study_b_reml_accuracy_and_regret.svg).

On Study B (not re-run), REML recovers the published C1 and C2 oracles. The protocol claim of equal $v_{\Delta}$ at $t=1/2$ is false (exact-fraction check in [checks/](checks/check_vdelta_closed_forms.py) and a core-Lean proof in [checks/lean/VDelta.lean](checks/lean/VDelta.lean)); $4\times 4$ is uniquely optimal there, and the close calls are at $t=4/15$ and $t=11/15$, where REML picks the oracle in 50% to 64% of replications (a coin flip between two nearly tied designs). Six of 55 cells are nominally worse than equal allocation; none survives Holm over 55 cells.

```bash
python3 src/fetch_raw_data.py          # downloads and checks the six raw files
python3 src/prepare_data.py            # builds data/processed/study_a_bundle.npz
python3 src/study_a.py                 # v1 Study A (superseded), about 5.6 minutes
python3 src/study_b.py
python3 src/verify.py                  # skips the row-count check if the v1 CSV is absent
python3 checks/check_vdelta_closed_forms.py
~/.elan/bin/lean checks/lean/VDelta.lean   # core Lean 4, no Mathlib
python3 src/make_figures.py
# corrected Study A: see v2_matched_budget_corrected_11-27-2026/RESULTS_V2.md for its commands
```

**TLDR-end:** [judge objectives: experiment 39] For new-item RPS a fixed rating budget is mildly better spent on more items than on more raters (18 of 20 cells by mean, 5 significant after Holm, at most 3% of RPS), and a 1,280-rating pilot did not pay for itself against that fixed rule; for universe-score precision the formulas still prefer more raters per item. Proper scores name the target; they do not choose an allocation by themselves.
