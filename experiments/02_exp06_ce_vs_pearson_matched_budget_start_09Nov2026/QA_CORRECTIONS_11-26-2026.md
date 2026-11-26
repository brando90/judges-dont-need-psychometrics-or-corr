# Experiment 02: verified QA corrections (11-26-2026)

**Date: 11-26-2026.** Codex CLI, GPT-6 Astra ultra (Vals account). CPU Python/raw-output checks; no Lean checking, no new model calls. Original protocols, result tables and raw outputs remain preserved.

**g1-M2 / g1-M3 — CONFIRMED evidence limitations; reconstruction remains open.** Old matched-comparison CE difference: “−0.260401 [−1.457729, 0.198472].” Verified 11-26-2026 point difference from five per-partition outputs: **−0.260401278759**; saved endpoints agree to rounding. Old cached-selection CE difference “+0.770 [−0.895, 1.317]” has point difference **+0.769587959519**. Old weighted-training difference “−0.010409 [−0.061911, +0.005933]” has point difference **−0.010409135349**. The distance-follow-up “+0.023822 [−0.036327, +0.053333]” has only aggregate outputs here. None of these historical intervals is independently reconstructed in phase 2: private ratings, predictions and per-bootstrap held-out scores are absent. The 136 interval endpoints in the main 17-metric panel remain aggregate-verifiable only. Fit receipts establish fits and failures, not the missing score draws. Point estimates are not declared wrong; all original estimates and intervals remain unchanged.

**g1-L2 — CONFIRMED, integration status.** Old wording: “still needs a paper-integration decision” and “The paper remains unchanged.” Corrected 11-26-2026: paper integration is recorded as completed on 11-17-2026 in the experiment index; paper sources and integration commits are present. Continued experiment ownership does not imply integration is still pending.

**g1-L6 — CONFIRMED, metric presentation.** Old reports isolate “Training Pearson”, “Mean within-fold Pearson” and correlation-only held-out tables. Corrected 11-26-2026: the matched held-out panel below reports ranked probability score (RPS), cross-entropy (CE) and total variation distance (TVD) before Spearman and Pearson for every method whose per-partition scores are saved. The original historical tables remain unchanged. Training correlations are optimization diagnostics; matching training RPS/CE/TVD was not saved as a complete panel, so held-out proper scores must not be described as training scores. RPS averages five squared cumulative-distribution errors [0,1], CE is negative log probability of each individual rating (nats), and TVD is half the absolute probability difference from the empirical histogram [0,1], lower better. Correlations of expected rating with human mean are secondary [-1,1], higher better. The held-out rows average the five fixed overlapping partitions of 75 artifacts, 25 tasks and 195 ratings with task-family separation; uncertainty is not recomputed from partition variation.

## Recomputed evidence and matched metric panel (11-26-2026)

Date: 11-26-2026. Read-only CPU reconstruction; no model calls.

g1-M2: CE-trained minus Pearson-trained CE from five per-partition scores = -0.2604012787586951
Saved refit interval (not independently reconstructed): {'estimate': -0.2604012787586951, 'hi': 0.1984719895967459, 'invalid_draws': 2, 'lo': -1.4577286817584236, 'p_value': None, 'valid_draws': 998}
g1-M3: cached CE-selected minus Spearman-selected CE = 0.7695879595190789
g1-M3: weighted-trained minus Pearson-trained weighted loss = -0.010409135349498166
Private dataset and per-bootstrap held-out scores are absent from this checkout; aggregate endpoints are not fresh quantile reconstructions.
g1-L6: held-out panels below precede separate training-correlation diagnostics; training proper-score panel unavailable in saved outputs. Held-out rows average five overlapping partitions of 75 artifacts, 195 ratings, 19 task families; not 375 independent artifacts.

| Dataset / judge | RPS | CE (nats) | TVD | Spearman | Pearson |
|---|---:|---:|---:|---:|---:|
| Matched mixtures / ce | 0.139079625 | 5.491440838 | 0.512387331 | 0.670348558 | 0.766431783 |
| Matched mixtures / constant | 0.187207614 | 1.928000882 | 0.696790026 | -0.347284877 | -0.404177125 |
| Matched mixtures / pearson | 0.143634477 | 5.751842117 | 0.518075329 | 0.687232133 | 0.753947181 |
| Historical cached selection / brier | 0.206695037 | 8.852204448 | 0.651110484 | 0.614889156 | 0.639813260 |
| Historical cached selection / ce | 0.203198699 | 8.867520625 | 0.651110423 | 0.671126890 | 0.678107271 |
| Historical cached selection / kendall | 0.179258073 | 8.097932666 | 0.599999467 | 0.769241233 | 0.777381067 |
| Historical cached selection / pearson | 0.179258073 | 8.097932666 | 0.599999467 | 0.769241233 | 0.777381067 |
| Historical cached selection / rps | 0.215406058 | 8.748787212 | 0.642666061 | 0.591601899 | 0.600683635 |
| Historical cached selection / spearman | 0.179258073 | 8.097932666 | 0.599999467 | 0.769241233 | 0.777381067 |
| Historical cached selection / tvd | 0.203613525 | 8.826751787 | 0.649332677 | 0.665106691 | 0.670724933 |
| Historical cached selection / weighted_pmf | 0.204917302 | 8.789750148 | 0.644888281 | 0.585994747 | 0.619198836 |
| Historical fitted heads / brier | 0.116141995 | 1.496750505 | 0.481920187 | 0.652331877 | 0.736207653 |
| Historical fitted heads / ce | 0.117427876 | 1.527141302 | 0.466798805 | 0.658669566 | 0.735569296 |
| Historical fitted heads / pearson | 0.181758394 | 1.897422144 | 0.672101185 | 0.142945852 | 0.056961528 |
| Historical fitted heads / rps | 0.118395973 | 1.555052242 | 0.494567754 | 0.645333066 | 0.727838926 |
| Historical training follow-up / constant | 0.187207614 | 1.928000882 | 0.696790026 | -0.347284877 | -0.404177125 |
| Historical training follow-up / mixture_pearson | 0.143634477 | 5.751842117 | 0.518075329 | 0.687232133 | 0.753947181 |
| Historical training follow-up / mixture_weighted_pmf | 0.134226886 | 5.659607095 | 0.504989630 | 0.669716565 | 0.768025066 |
| Historical training follow-up / softmax_brier | 0.116142324 | 1.496756583 | 0.481921380 | 0.652331877 | 0.736206400 |
| Historical training follow-up / softmax_ce | 0.117428590 | 1.527168296 | 0.466800220 | 0.658669566 | 0.735566756 |
| Historical training follow-up / softmax_rps | 0.118396047 | 1.555056167 | 0.494566659 | 0.645333066 | 0.727835447 |
| Historical training follow-up / softmax_weighted_pmf | 0.117855648 | 1.524401262 | 0.487215005 | 0.650816477 | 0.733365286 |


Reproduce the evidence with `python3 docs/qa_11-25-2026/phase2/verify_g1_g2_remaining.py 2` from the repository root. HIGH-item checks are separately saved as `verify_g1-H1.py` (experiment 05), `verify_g2-H1.py` / `verify_g2-H2.py` (experiment 11), and `verify_g2-H3.py` (experiment 07), with their printed `.txt` outputs.
