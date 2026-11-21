# When does cross-entropy beat Pearson for rating forecasts? (11-21-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/18_when_does_ce_beat_correlation_notebook_page_math_and_sweep_start_21Nov2026/conditional_shape_and_rating_budget/README.md>

Date: 11-21-2026

A forecast that gets the mean rating exactly right can still give the wrong probability of disagreement. Cross-entropy (CE) can learn this difference from individual ratings. How much that helps on held-out artifacts depends on the candidate judges, the variation in rating distributions, how ties are broken, and the amount of data.

## Evidence from historical expert ratings

Selecting one of 15 saved judges by training CE gave lower average held-out ranked probability score (RPS) than selecting by training Pearson in all six annotation budgets. The pointwise 95% family-reweighting intervals favor CE at three and seven training families. At fifteen families the data allow an advantage for either selector. These results concern **selecting among fixed candidates**.

| Training families | Ratings per training artifact | Constant test RPS | CE-selected test RPS | Pearson-selected test RPS | Test RPS of Pearson-selected minus CE-selected [95% interval] |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 0.2050 | 0.1500 | 0.1831 | +0.0331 [+0.0168, +0.0573] |
| 3 | 2 | 0.2073 | 0.1472 | 0.1787 | +0.0315 [+0.0166, +0.0644] |
| 7 | 1 | 0.1943 | 0.1424 | 0.1580 | +0.0156 [+0.0028, +0.0540] |
| 7 | 2 | 0.1939 | 0.1396 | 0.1524 | +0.0128 [+0.0007, +0.0523] |
| 15 | 1 | 0.1875 | 0.1320 | 0.1479 | +0.0159 [-0.0200, +0.0314] |
| 15 | 2 | 0.1870 | 0.1319 | 0.1493 | +0.0174 [-0.0226, +0.0327] |

Normalized individual-rating RPS is the average squared error of the five cumulative probabilities on the 0–5 rating scale. Its range is 0–1, lower is better, and positive differences favor CE. Each selector exhaustively evaluated the same 15 candidates on the same training data with a fixed tie order. The constant forecast uses the smoothed training rating histogram. Results average 250 matched splits with four test families each. There were 19 unique task families, 25 tasks, 75 artifacts and 195 previously studied expert ratings. Each test artifact contributed the mean loss over all its available ratings. We then averaged the artifacts equally within each test split and the 250 splits equally. The 30 artifacts with two ratings and 45 artifacts with three ratings therefore carry the same per-artifact weight whenever they share a test split. No raters were held out. The 95% intervals use 1,000 paired Bayesian bootstrap draws over families with selection repeated. They condition on this candidate pool, rater panel, split schedule and rating subsamples. They are pointwise intervals with no hypothesis test (p-val=n/a). The supplied reference matched the annotation-era hash for 15 artifacts. Reference provenance is unresolved for the other 60. [Protocol](historical_rating_budget/PROTOCOL_11-21-2026.md) · [Human results and uncertainty](historical_rating_budget/RESULTS_11-21-2026.md) · [Full held-out metric panel](historical_rating_budget/scores.csv)

**Can a Pearson-selected judge win at test correlation?** It can, but selecting on Pearson does not ensure it. At fifteen training families and two ratings per artifact, the CE-selected judge had higher mean within-split test Pearson (0.745 vs 0.692) and Spearman (0.691 vs 0.634). Both measure tracking of artifact means, on a scale from −1 to 1. In the earlier five-judge scalar-vote pool the correlation-selected judge had the higher test Pearson and Spearman, and lower test RPS than the CE-selected judge (0.180 vs 0.205). A follow-up held the human data fixed and applied both split protocols to each pool. Cross-entropy selection had lower held-out RPS throughout the 15-judge native-probability pool. Pearson selection had lower held-out RPS throughout the 44-judge three-vote pool. Pool membership, prompts, model families and native forecast format vary together. The comparison identifies a reversal across these historical pools. [Matched-protocol comparison](../reconcile_single_judge_selection_21Nov2026/RECONCILE_11-21-2026.md) [Current metric panel](historical_rating_budget/scores.csv) · [Earlier scalar-vote pool and controls](../../16_rps_selected_arm_and_probability_floor_on_scalar_pools_start_20Nov2026/results.md)

## How the sample-size result depends on what is fitted

An earlier study fitted a 48-parameter probability model to the same historical ratings. With three training families (31 ratings on average), held-out RPS was 0.1848 for CE training and 0.1814 for Pearson training. At five families (52 ratings on average), it was 0.1771 and 0.1815 respectively. Positive differences persisted at the larger tested sizes. The reported 95% intervals resample overlapping split averages and only measure variation from the split schedule on this cache. The Pearson optimization was nonconvex. These measurements show a change in ordering for that fit and pool. They cannot prescribe a human sample size. [Earlier study, training-size curve and numerical controls](../../10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026/RESULTS_frontier_ce_vs_corr.md#4-when-does-proper-loss-beat-correlation-real-data-learning-curves--disagreement-strata)

In our controlled simulation, the two judges exhaustively searched the same 12 forecasts. Each of 256 independent replications had 384 fresh synthetic test artifacts and three simulated ratings per test artifact. Points below give test RPS of the Pearson-trained judge minus the CE-trained judge, with pointwise 95% paired percentile bootstrap intervals over 2,000 resamples of replications (p-val=n/a).

- **Eight artifacts, one rating each.** When Pearson's default tie-break picked the true neutral shape, Pearson won by 0.00349 RPS (difference −0.00349 [−0.00402, −0.00295]). With the same nonlinear mean curve and an observed feature that changed spread at fixed mean, CE won by 0.00303 [+0.00235, +0.00367].
- **128 artifacts, 16 ratings each.** When spread varied, CE recovered the true distribution in all 256 replications. The Pearson tie-break kept the neutral shape. The gap was +0.00509 [+0.00500, +0.00518]. Both methods had identical test Pearson and Spearman. Using 512 independent validation ratings and RPS to break Pearson ties recovered the true shape in all replications too. Under neutral truth the default methods tied.
- **128 total training ratings.** For both methods and both shapes, the intervals allowed either allocation to win when comparing 128 artifacts × one rating with eight artifacts × 16 ratings. Sampling more artifacts matters especially when the informative artifact type is rare. The derivation explains why repeated ratings of other artifacts cannot supply that missing information. [Sweep, plot and metric panel](results.md) · [Finite-sample derivation](MATH_11-21-2026.md)

## What is linear in the objective?

Let qᵢ(r) be the judge's predicted probability of rating r on artifact i. Define mᵢ = Σᵣ r qᵢ(r), the predicted mean rating, and let hᵢ be its observed mean rating. Standardize m and h separately across the n artifacts using population-divisor standard deviations. For nonconstant vectors, the Pearson loss is exactly

$$
1-\rho(m,h)=\frac{1}{2n}\sum_i [z(m_i)-z(h_i)]^2.
$$

Each term depends on the means and standard deviations of the entire sample. Ordinary CE instead contributes −log qᵢ(r) for an individual rating r. Converting qᵢ to mᵢ is a linear projection. A neural judge can predict an arbitrarily nonlinear mean, and the normalization in Pearson is itself nonlinear. Any change to category probabilities that preserves the expected rating leaves mean-based Pearson unchanged. For an interior distribution on six ratings, four independent directions per artifact preserve both total mass and expected rating. The derivation also treats correlating the probability bins themselves, a different objective with different invariances. [Derivation, population theorem and finite-sample bounds](MATH_11-21-2026.md)

The notebook's four possible input sets give a second illustration. In an exact eight-state example, hiding the reference preserved Pearson of 1 against the true means, while total variation distance (TVD) from the full conditional distribution rose from 0 to 0.12 and expected RPS rose from 0.18483 to 0.18771. This is an exact synthetic calculation. Real theorem-judge input ablations require matched forecasts. [Four-input calculation](input_ablation/README.md)

## Which evaluation answers which question?

Sanmi's phrase was **distributional accuracy**. For a forecast of individual ratings, use held-out individual-rating RPS as primary and CE as a complementary proper loss. CE measures the negative natural log probability of the rating in nats. TVD, half the summed absolute probability differences, is useful beside them, with the reference distribution specified. TVD against two or three ratings is noisy even for the true forecast. For a scalar rating interpreted in points, also measure mean absolute error and signed bias. For ranking, report rank agreement. Top-one accuracy discards the other five probabilities and does not address the distributional question.

[Photo, verbatim transcriptions and checksums](assets/ARCHIVE_11-21-2026.md) · [Gemini review and corrections](review/REVIEW_11-21-2026.md) · [Metric and reproduction guide](HOW_TO_READ_RESULTS.md) · [Google Doc](https://docs.google.com/document/d/1dcu-9M48NlBrCf77CgZj1Of8J0Hnj7_yOh9hC_nbbaE/edit)

The synthetic design was fixed locally before scores were measured and preserved byte for byte when the parallel runs were consolidated under experiment 18. Its original self-link uses the temporary directory name. The [protocol at its current location](PROTOCOL.md) has the original checksum.
