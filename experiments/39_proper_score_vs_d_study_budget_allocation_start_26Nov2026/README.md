# Experiment 39: Proper-score design versus D-study budget allocation

**Doc link:** <https://docs.google.com/document/d/1WtzpWNuUbkf3eGLOERMIvcyI7Sjec1hLfIZ1rJKfKvE/edit>

**TLDR:** A CPU head-to-head of proper-score design and G/D-theory allocation on ChaosNLI SNLI, MNLI-m, and alphaNLI, and on DICES-350 and DICES-990, plus the harder Phase B of experiment 04. The two targets do not share the same optimal split of a fixed rating budget. Proper-score design does not replace D-theory. Results: [results.md](results.md). Reading guide: [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md).

**As of:** 11-27-2026. Started 11-26-2026. Local CPU, `$0`, no model calls. Protocol frozen before measurement.

## Question

Once a learner is trained and evaluated with a proper score, do we still need a generalizability study (G-study) or a decision study (D-study) to allocate a fixed rating budget $B = n_{\mathrm{items}} \times n_{\mathrm{raters}}$?

The two official targets under test are:

1. A distributional forecast of new-item $p(y\mid x)$, scored by held-out test ranked probability score (RPS), with cross-entropy (CE) and total variation distance (TVD) as secondary checks.
2. Precision of a mean rating (per-item universe score, and separately the grand mean).

Experiment [04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026](../04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/README.md) showed that reliability is not sufficient for distribution accuracy, and that D-study allocation works under its assumptions, but left the harder close-design Phase B unrun. Experiment [34_g_d_theory_vs_ce_where_reliability_agrees_and_fails_start_25Nov2026](../34_g_d_theory_vs_ce_where_reliability_agrees_and_fails_start_25Nov2026/README.md) showed where the coefficients agree with cross-entropy and where a perfectly repeatable forecast still loses on RPS. This experiment asks whether a proper-score selector and a D-study selector pick the same $(n_{\mathrm{items}}, n_{\mathrm{raters}})$ split, on public multi-annotator data and on the missing Phase B grid.

## Evaluation setup

Study A uses the Experiment 28 27-transform CE-fit of five frozen frontier-judge probabilities (no new application programming interface (API) calls). A frozen 30% test split is never used to choose a design or a transform. U1 is the per-item D-study (always more raters). U2 is the grand-mean D-study (more items when $\widehat{v}_{u}>0$). The proper-score selector minimizes expected held-out RPS by 5-fold $\times$ 8 inner draws on an 80-item, 16-rating pilot. 150 replications. Candidate designs: $n_{i}\in\lbrace 16,32,64\rbrace$, $n_{r}\in\lbrace 2,4,8,16\rbrace$. “Beats” means a lower mean and a paired Student-$t$ 95% interval that excludes 0.

Study B matches the Experiment 04 Phase B crossed G-study, interpolates $v(t)=(1-t)\mathrm{C1}+t\mathrm{C2}$, and asks how often constrained residual maximum likelihood (REML) mis-selects among $8\times 2$, $4\times 4$, and $2\times 8$ as the facets become harder to tell apart. 200 replications (256 at the two reproduction cells).

## Result

On every comparable Study A cell, the modal item-mean oracle is the U1 design and the modal RPS oracle is the max-item design also picked by U2 and by the modal proper-score selector. The primary selector significantly lowers held-out test RPS relative to U1 in 10 of 20 comparable cells, never significantly loses, and is inconclusive in the other 10. It does not beat U2 on RPS (0 wins, 2 losses, 18 ties). U1 beats U2 on item-mean mean squared error (MSE) in 20/20 cells. Full tables and intervals: [results.md](results.md). Figures: [fig1](figures/fig1_study_a_allocation_and_rps.svg), [fig2](figures/fig2_study_b_reml_accuracy_and_regret.svg).

On Study B, REML recovers the published C1 and C2 oracles. The protocol claim of equal $v_{\Delta}$ at $t=1/2$ is false (numerics-only exact-fraction check in [checks/](checks/check_vdelta_closed_forms.py)); $4\times 4$ is uniquely optimal there, and the close calls are at $t=4/15$ and $t=11/15$. A small pilot can be significantly worse than equal allocation when $4\times 4$ is clearly best.

```bash
python3 src/prepare_data.py
python3 src/study_a.py
python3 src/study_b.py
python3 src/verify.py
python3 checks/check_vdelta_closed_forms.py
python3 src/make_figures.py
```

**TLDR-end:** [judge objectives: experiment 39] Proper-score design and a grand-mean D-study agree on spending a rating budget on more items when the target is new-item RPS; a per-item D-study is still the right tool when the target is universe-score precision. The two oracles disagree systematically.
