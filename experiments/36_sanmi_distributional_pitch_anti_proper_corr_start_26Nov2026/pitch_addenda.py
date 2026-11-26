"""Static sections added after the main run. Numbers in POWER are copied from results_power_mdd.json."""

ELICITATION = r"""## Elicitation hierarchy

Sanmi's question was what accuracy, cross-entropy, and moments each identify. The rows below are ordered by how much of \(p^*(y \mid c)\) the population minimizer is forced to match. A loss is Bayes-consistent for a target functional when every population minimizer returns that functional. Bartlett, Jordan, and McAuliffe (2006) call the 0–1 case classification calibration. Gneiting and Raftery (2007) characterize strictly proper scoring rules: the Bayes act is the full distribution.

| Row | Elicitation | Bayes act | Properness |
| --- | --- | --- | --- |
| 1 | Hard 0–1 accuracy | A point mass on the mode, \(\delta_{\mathrm{mode}}\). Accuracy computed on a sample from the judge is still maximized by always emitting the mode. | Bayes-consistent for the mode only. The rest of the probability mass function is free. |
| 2 | Scalar mean squared error | The conditional mean \(\mu_Y(c)\). | Bayes-consistent for the mean. Silent on spread and on a second mode. |
| 3 | Pearson \(\operatorname{Corr}(X, Y)\) | An affine-uncalibrated mean. For a judge sampled independently of \(Y\) given \(c\), the maximizer collapses to \(X_\delta = a\mu_Y(c)+b\) with \(a>0\), and \(\sqrt{\rho}>\rho\). | Not strictly proper. Not Bayes-consistent for \(p^*\). |
| 4 | Cross-entropy or Brier | The full probability mass function, \(\hat q = p^*\). | Strictly proper and class-symmetric. A wrong class is not charged by how far it sits on the rubric. |
| 5 | Ranked probability score, or the continuous ranked probability score | The full cumulative distribution. The loss grows with the distance of the realized rating. | Strictly proper, and ordinal-distance-sensitive. The continuous ranked probability score is the version for a real-valued \(Y\). |

**Symmetric agreement versus asymmetric target matching.** An inter-rater coefficient treats the two scores as exchangeable. Swapping the raters does not change Pearson correlation, an intraclass correlation, Krippendorff's \(\alpha\), or a generalizability coefficient. A proper score does not swap. It scores a forecast \(\hat q\) against a draw from a target \(p^*(y \mid c)\). That is asymmetric target matching. The psychometric coefficient remains available as a description of the human panel. It is not the loss that trains the judge, and it is not the loss that says whether the judge matched \(p^*\).
"""

COT = r"""## Chain-of-thought logits, and a model-size axis that was not run

The single-pass numbers above are a linear softmax: one map from the prompt features to a distribution on the rating classes. A chain-of-thought judge is a different object. If the model first writes a rationale \(r\) and the rating logits are read at the end of that text, those logits are \(q(y \mid c, r)\), the rating distribution conditional on one sampled trajectory. The marginal the proper scores want is
\[
q(y \mid c) = \sum_r q(r \mid c)\, q(y \mid c, r).
\]
One rationale does not return that sum. Two procedures do. A direct forward pass, with the rating token next and no rationale, reads \(q(y \mid c)\) from the logits of the rating tokens. \(M\) independent chain-of-thought samples, \(M \in \{1, 5, 20\}\) in the rollout above, are a Monte Carlo estimate of the same marginal. The linear-head figure is that comparison without a live language model. `scripts/snap_llm_logit_readout.py` states the same split for a GPU node.

A second scaling axis is model size \(P\), beside the annotation budget \(B = N \times k\). The GPU script records a planned grid, Qwen3-4B and Llama-3.1-8B, and does not run it. No held-out number in this folder is a function of \(P\).
"""

POWER = r"""## Test-set power and minimum detectable difference (DICES-990)

This complements the unbiased test risk at one rating per prompt. The two judges are fixed: the cross-entropy head and the Pearson head, trained once on the DICES-990 train split (594 conversations, seed 0). Only the test design changes. A cell is \(N_{\mathrm{test}}\) held-out conversations and \(k_{\mathrm{test}}\) ratings from each. There are 200 random subsets. Each subset gets an item-clustered percentile 95% interval from 200 resamples of its conversations. Power is the fraction of subsets whose interval excludes 0. The minimum detectable difference is the median half-width of those intervals.

For cross-entropy, Brier, and sqrt(RPS), the difference is the Pearson-trained head minus the cross-entropy-trained head. Positive means the cross-entropy head is better. For Pearson \(r\), the difference is the cross-entropy head's correlation minus the Pearson head's correlation.

At one test rating per prompt, cross-entropy reaches power 0.80 between 50 prompts (power 0.77, minimum detectable difference 0.440 nats) and 100 prompts (power 0.96, minimum detectable difference 0.317). Brier reaches it at 150 prompts (power 0.84, minimum detectable difference 0.070). sqrt(RPS) reaches it at all 198 prompts (power 0.90, minimum detectable difference 0.048). Pearson \(r\) does not. At \(k_{\mathrm{test}}=1\) and \(N_{\mathrm{test}}=198\) its power is 0.62 and its minimum detectable difference is 0.023, while the median correlation gap in those subsets is 0.028. Correlation first reaches power 0.80 at \(k_{\mathrm{test}}=4\) with all 198 prompts (power 0.82), or at \(k_{\mathrm{test}}=8\) with 150 prompts (power 0.85).

The proper scores separate these two heads with fewer test ratings than the correlation does, because the proper-score gaps are large and the correlation gap is small. The figure is `figures/eval_power_mdd_dices990.png`. The full grid is `tables/eval_power_mdd_dices990.md`.

![Test-set power on DICES-990](figures/eval_power_mdd_dices990.png)
"""
