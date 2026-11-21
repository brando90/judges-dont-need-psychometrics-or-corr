# A non-circular evaluation protocol for CE-trained vs correlation-trained judges (11-21-2026)

Dated 11-21-2026. Written before any new data were looked at. By ultimate_brando9 (Brando's AI assistant), Claude Sonnet 5.5, Claude Code headless harness. No model was called and no experiment was run for this note; the only computation is a 40-line numeric illustration of the algebra (Section 2, `toy_circular.py` and `toy_circular_output.txt` in this folder). Builds on experiment 18 (maths R1–R7, W1–W7), experiment 20 (limited data), experiment 21 (Codex's frozen fresh-family protocol, commit ff2f07a) and Sanmi's feedback (repo note, commit 10897ca). Typeset version: [EVAL_PROTOCOL_11-21-2026.pdf](EVAL_PROTOCOL_11-21-2026.pdf) (same content, same numbers).

## Bottom line (11-21-2026)

1. **Scoring a Pearson-trained judge with Pearson correlation is circular twice.** (a) The training objective and the test metric are the same function of the judge's mean ratings, so the Pearson-trained judge is favoured by construction. (b) That function sees only the mean rating per item, up to a shift and a positive rescaling (R1, R2), so it ties or inverts judges that differ a lot in level, scale, spread and shape. In the toy of Section 2 a judge with ρ = 1.000 has ranked probability score (RPS) 0.189, no better than the uniform forecast (0.192), while a judge with ρ = 0.949 has RPS 0.108.
2. **Primary metrics:** held-out ranked probability score (RPS), then cross-entropy (CE) and total variation distance (TVD), each against the human rating distribution, always next to the uniform forecast. **Metrics neither objective optimises:** top-1 agreement with the human modal rating, calibration of the forecast cumulative probabilities, calibration slope of human mean on judge mean; pairwise preference accuracy for ranking uses. Pearson, Spearman and Lin's concordance correlation are secondary columns.
3. **Decision rule:** find the smallest number N\* of human ratings at which the CE-trained judge beats both the Pearson-trained judge and the uniform forecast on held-out RPS, with an interval, over held-out task families. If the rating budget B buys at least N\* ratings (B/c ≥ N\*, c = cost of one rating), train with the proper loss; below N\* do not fit a flexible head by either loss.
4. **Both ways stay open.** Where Pearson wins (ranking uses, picking one judge among a few, very small N as in experiment 20) the protocol reports it in the same table.

## 1. Limits of each metric (read this first)

| Metric | Reads | Blind to | Proper for p\*? | Favoured by construction |
|---|---|---|---|---|
| Pearson ρ between judge means and human means | one mean $`m_i`$ per item, up to a shift and a positive rescale (R1, R2) | level, scale, spread, skew, all shape | no | the Pearson-trained judge (its own objective) |
| Spearman | ranks of the means | same, plus the size of gaps | no | none directly (rank version of Pearson) |
| Lin's concordance (CCC) $`\frac{2s_{mt}}{s_m^2+s_t^2+(\bar m-\bar t)^2}`$ | means with level and scale | spread, shape | no | none; penalises level and scale that Pearson ignores |
| **RPS** (primary) | the whole forecast CDF at all K−1 thresholds | little; noisy when each item has few ratings | yes, strictly | the CE-trained judge in the population limit (same minimiser p\*); not its training loss |
| **CE** | each category probability | very large penalties on near-zero probabilities, so the floor ε matters (Sanmi) | yes, strictly | the CE-trained judge (its own objective) |
| **TVD** | L1 distance to the empirical rating histogram | with k ratings per item the expected score is biased (k = 1: $`1-\sum_r p^\ast(r)p(r)`$, R6) | no for sparse labels | none |
| Top-1 agreement | the forecast's modal rating | everything except the mode | no | none |
| Calibration (reliability, slope) | whether forecast cumulative probabilities and means match what raters did | ranking quality | no | none |
| Pairwise preference accuracy | the sign of mean differences between item pairs | level, scale, spread (reads means only, like Pearson) | no | none directly |

## 2. Why Pearson scoring of Pearson-trained and CE-trained judges is circular

**Notation.** Items $`i=1,\dots,n`$; ratings $`r\in\{0,\dots,K-1\}`$; the judge forecasts a distribution $`p_i`$; humans have a distribution $`p^\ast_i`$, observed through $`k_i`$ ratings $`y_{i1},\dots,y_{ik_i}`$. Means $`m_i=\sum_r r\,p_i(r)`$ (judge) and $`t_i=\sum_r r\,p^\ast_i(r)`$ (humans); averages $`\bar m,\bar t`$; $`\tilde t_i=t_i-\bar t`$. $`F_q(k)=\sum_{r\le k}q(r)`$. Pearson: $`\rho(m,t)=\frac{\sum_i(m_i-\bar m)(t_i-\bar t)}{\sqrt{\sum_i(m_i-\bar m)^2\sum_i(t_i-\bar t)^2}}`$.

**Step 1 (R1: Pearson reads only the means).** The numerator is $`\sum_i(m_i-\bar m)(t_i-\bar t)=\sum_i\tilde t_i m_i-\bar m\sum_i\tilde t_i=\sum_i\tilde t_i m_i`$ because $`\sum_i\tilde t_i=0`$. For a fixed target the denominator is a function of $`m`$ alone. So $`\rho=F_t(m_1,\dots,m_n)`$: two judges with the same means have the same ρ whatever their $`p_i`$.

**Step 2 (R2: shift and positive rescale do not matter).** For $`b>0`$, $`\rho(a+bm,t)=\rho(m,t)`$: the shift cancels in the centering and $`b`$ multiplies numerator and denominator. Every judge with $`m_i=a+bt_i`$, $`b>0`$, has ρ = 1.

**Step 3 (the objective is the metric).** The Pearson-trained judge maximises ρ on training items, i.e. $`F_t`$ on the training means. Scoring it with Pearson on test items evaluates the same functional on new items. Empirical risk minimisation of a metric is expected to win on that metric on test data; that says nothing about whether the output is useful as a rating or a distribution. A fair test needs a metric that is not the training objective of exactly one arm, or one that no arm optimises.

**Step 4 (the metric cannot see what CE learns).** CE and RPS are proper: their expected value over $`Y\sim p^\ast`$ is minimised only at $`p=p^\ast`$ (R6), which fixes the mean and every higher moment. Pearson gives the same value 1 to $`p=p^\ast`$ and to any judge with $`m=a+bt`$, however wrong its level, scale, spread and shape.

**Numeric illustration** (exact expectations, not an experiment; `toy_circular.py`). K = 6, five items with human means t = (0.5, 1.5, 2.5, 3.5, 4.5) and human distributions Binomial(5, t/5). Expected scores under those distributions; lower is better for RPS, CE, TVD. Judge A compresses the means, $`m_i=2.5+0.1(t_i-2.5)`$. Judge B has means $`t_i+e_i`$, e = (+0.5, −0.5, +0.5, −0.5, 0).

| Forecast | Pearson ρ (higher better) | RPS | CE (nats) | TVD | rank by Pearson / by RPS |
|---|---|---|---|---|---|
| Truth p\* | 1.000 | 0.0953 | 1.234 | 0.000 | 1 (tie) / 1 |
| A: compressed means, ρ = 1 | 1.000 | 0.1891 | 1.983 | 0.397 | 1 (tie) / 3 |
| B: true means plus errors | 0.949 | 0.1083 | 1.341 | 0.169 | 3 / 2 |
| Uniform | undefined (constant means) | 0.1922 | 1.792 | 0.427 | – / 4 |

Pearson ranks A above B; RPS ranks B far above A, and A is no better than uniform on RPS and worse than uniform on CE. This is the algebra above in numbers, not evidence about real judges.

**The symmetric caveat.** RPS is not neutral either: the CE-trained judge is consistent for p\*, which is what RPS measures. That is the substance of the claim ("read the output as a distribution, score it with a proper rule"), not a side effect, but it is why the protocol adds metrics that neither objective optimises (Section 3) and reports where Pearson wins. A second circularity is choosing a judge or prompt by Pearson and then reporting Pearson; the protocol selects and reports with different data and different metrics (Section 6, selection arms).

## 3. Metrics, step by step

For a held-out item i with ratings $`y_{ij}`$ and forecast $`q=p_i`$. Every score is averaged over the item's ratings, then over items in a held-out family, then over families. Lower is better unless stated.

- **RPS** (primary; normalised ranked probability score, range 0 to 1): $`S(q,y)=\frac1{K-1}\sum_{k=0}^{K-2}\big(F_q(k)-\mathbf 1\{y\le k\}\big)^2`$. Reference: the uniform forecast. Ordered scales only.
- **CE** (nats): $`-\log q_\varepsilon(y)`$, $`q_\varepsilon=(1-\varepsilon)q+\varepsilon/K`$, ε = 6×10⁻⁶ as in experiment 21, and the table recomputed with ε = 10⁻² (Sanmi's floor point). Reference: log K for uniform.
- **TVD** (range 0 to 1): $`\tfrac12\sum_r|q(r)-\hat h_i(r)|`$, $`\hat h_i`$ the item's empirical histogram. Descriptive when $`k_i`$ is small (R6). With about 100 ratings per item (ChaosNLI, DICES) the histogram is close to $`p^\ast_i`$ and TVD is meaningful.
- **Top-1 agreement** (higher better): 1 if $`\arg\max_r q(r)`$ is among the human modal ratings of the item, else 0 (forecast ties share credit).
- **Calibration of cumulative probabilities:** for each threshold k, sort held-out items by $`F_q(k)`$, cut into B = 5 equal-count bins, take the weighted mean |mean forecast − observed frequency of y ≤ k|; average over thresholds (ECE). **Calibration slope:** least-squares fit $`\hat t_i=a+b\,m_i`$ on held-out items ($`\hat t_i`$ the human mean); ideal (a, b) = (0, 1). This measures what Pearson ignores.
- **Pairwise preference accuracy** (higher better): over item pairs in the same family with human-mean gap ≥ δ, the share where sign($`m_i-m_j`$) = sign($`\hat t_i-\hat t_j`$), ties in m counting ½; δ = 1 rating point (for K = 2, any difference), fixed now. It reads means only, so it cannot separate shape; it is the metric for ranking uses.
- **Secondary:** Pearson, Spearman, Lin's CCC $`=2s_{mt}/(s_m^2+s_t^2+(\bar m-\bar t)^2)`$ (higher better).
- **Target noise.** With one to three ratings per item the observed histogram is a poor stand-in for $`p^\ast_i`$. On datasets with about 100 ratings per item, train on a few ratings subsampled from each item but score against the full histogram (expected RPS and CE under $`\hat h_i`$). Report both the sparse and the dense scoring, labelled.

## 4. The "who wins" table (layout, filled by each run)

One block per dataset, judge and training size N (human ratings). Cell: difference with a 95% interval and the winner. For lower-is-better metrics the difference is *test metric of the comparator minus test metric of the CE-trained judge*; for higher-is-better metrics it is *test metric of the CE-trained judge minus test metric of the comparator*. Positive always means the CE-trained judge is better. Absolute values of both baselines are printed beside it.

| Held-out metric | vs Pearson-trained judge | vs Pearson-trained judge with affine calibration | vs uniform forecast | vs training histogram |
|---|---|---|---|---|
| **RPS** (primary) | Δ [lo, hi], winner | same | same | same |
| **CE**, floors 6×10⁻⁶ and 10⁻² | same | same | same | same |
| **TVD** | same | same | same | same |
| Top-1 agreement | same | same | same | same |
| Calibration ECE | same | same | same | same |
| Calibration slope (distance of b from 1) | same | same | same | same |
| Pairwise preference accuracy | same | same | same | same |
| CCC, Spearman, Pearson (secondary) | same | same | same | same |

Then a summary table: rows dataset × judge; columns N\* under RPS, CE, TVD and under the neutral metrics; a column "where Pearson wins".

## 5. Sanmi's points mapped to what the protocol does

Sanmi's feedback of 11-16-2026 (repo note, commit 10897ca): his five points, then the framing and prompts-vs-weights notes.

| Sanmi's point | What the protocol does |
|---|---|
| 1. Human data: the frontier mixture is null; both judges in the open-pool win lose to the constant forecast; single-judge selection favours correlation; add an RPS-selected arm | Selection arms: pick the best judge by Pearson, by CE and by RPS on the training ratings, score the pick on held-out families. A comparison where both arms lose to uniform is labelled "both worse than uniform", never a win. CE reported at two floors. |
| 2. Toy baselines: a Pearson-trained softmax pins only the mean, so it loses on RPS; the fair rival is score-then-calibrate; add affine-calibrated or MSE baselines; report RPS against the constant and the truth, not ratios of excess | Arms include Pearson-trained with affine calibration (least-squares a, b on training items, then exponential tilting of the judge's distribution to the calibrated mean) and an MSE-trained mean head with the same tilting. Every table gives absolute RPS next to uniform and, where known, the truth. |
| 3. Small n: fitting many weights to a scale-free objective on 16 tasks overfits; the supportable claim is "do not fit flexible models by maximising correlation" | Training-size sweep from 4 to 256 ratings. Each arm gets its own ridge strength from one grid, chosen by inner leave-one-family-out cross-validation on that arm's own loss (no single shared ridge). Failures at small N are reported (experiment 20: CE loses at 4 items). |
| 4. Use: distribution accuracy is the target when the output is read as a rating or distribution; ranking uses need rank agreement; item signal vs rater severity | Two use-cases with their own primary metric: distribution reading → RPS; ranking → pairwise accuracy and Spearman. Where raters have identifiers (DICES) fit an ordinal model with rater effects and score against the severity-adjusted expected rating as a second target; with two to three raters the mean is the target and severity is marked unidentified. |
| 5. References: Lin's concordance, ICC consistency vs agreement, ordinal likelihood models with rater facets as proper-loss judges | CCC is a secondary column; ICC (agreement form) between judge and human mean is reported when ≥ 2 raters per item; the ordinal-likelihood head is an extra proper-loss arm. |
| Framing: narrower claim, drop "don't need psychometrics" | The claim under test is Sanmi's wording: correlation measures association with mean ratings and cannot certify agreement in level and scale, let alone spread; when output is read as a rating or distribution, select and report with a proper scoring rule such as RPS and keep correlation as a ranking diagnostic. |
| Prompts vs weights: tune prompts and harness; coarse selection; run after the workshop deadline | A selection mode: choose among a small set of prompts or judges by Pearson, CE or RPS on N ratings and report the selection regret on held-out families. Planned after the workshop deadline, as Sanmi suggested. |

## 6. Arms, baselines, training sizes and intervals

**Arms** (all fitted on the same training ratings, same features, optimiser and starting point): uniform forecast; training histogram with 0.5 pseudocount per bin; Pearson-trained; Pearson-trained with affine calibration; MSE-trained mean head with tilting; CE-trained; optionally RPS-trained and the ordinal-likelihood head. A frozen judge's weights never change: the fitted part is a small head (experiment 21 uses a 27-point grid of transformations and exhaustive search, so the optimum is guaranteed). Every caption states the objective, search space and budget.

**Training size.** N ∈ {4, 8, 16, 32, 64, 128, 256} human ratings, as in experiment 21, reported as ratings and as items × ratings per item (one rating per item and two or three per item are separate rows; experiment 20 found different thresholds for them).

**Splits and intervals.** Hold out whole task families (assignments, genres, prompt groups); at least 8. Per family average the paired difference over 200 seeded training draws; across the F families the one-sided 97.5% bound is mean − $`t_{F-1,0.975}\cdot s\sqrt{1/F+1/(F-1)}`$ (the overlap-adjusted form of experiment 21; F = 10 gives 9 degrees of freedom), plus the $`2^F`$ sign-flip permutation check, family win counts and variation across draws. Draws within a family are never counted as independent evidence. Judges are fixed in advance and α = 0.05 is split by their number.

## 7. Break-even rule: when is CE worth the cost of the ratings?

For a dataset, judge and metric (RPS first), at each grid size N let $`d_N^{P}`$ and $`d_N^{U}`$ be the held-out RPS of the Pearson-trained judge minus that of the CE-trained judge, and the same for the uniform forecast. N *passes* if the lower interval bound of both is above 0. Then

$$N^\ast=\min\{N\in\text{grid}:\ N'\text{ passes for every grid size }N'\ge N\}$$

(test from the largest size down and stop at the first failure, as in experiment 21; if no size passes, report "N\* > 256"). Report N\* in ratings and in items.

**Cost.** Let c be the cost of one human rating (money or minutes) and B the budget, so $`N_B=\lfloor B/c\rfloor`$ ratings. **Rule:** if $`N_B\ge N^\ast`$, train with a proper loss; the break-even cost of the ratings is $`C^\ast=c\,N^\ast`$. If $`N_B<N^\ast`$, do not fit a flexible head by either loss: use the uniform forecast or the training histogram, or the Pearson-trained judge with affine calibration if it passes against uniform at $`N_B`$. Neutral-metric gate: CE must not lose to Pearson on top-1 agreement, calibration or pairwise accuracy by more than a tolerance fixed before the run (0.02 in each metric's units). Brando's rule, in his words (11-21-2026): "If for the practical 'cost' of getting human ratings/scores/judgements N is sufficiently large to use CE then we shouldn't be using Corr."

**Already measured** (experiment 20, softmax-linear head, synthetic, 3 ratings per item): CE beats the Pearson-trained judge from 16 items (48 ratings; +0.034, 96% of seeds), ties at 8 items, loses at 4; with one rating per item it crosses at 32 items (32 ratings, 94%). Real 75 artifacts: from 7 training families up in the closed-weight 15, all 59 and historical 44 pools (+0.027 to +0.058); at 3 families it ties and it loses in the open-weight 39 pool (−0.037). These are not N\* under this protocol (different splits and baselines).

**Experiment 21** (Codex, frozen before this note) already uses RPS first, uniform and Pearson baselines, 4–256 ratings and whole-assignment holdouts. This note does not change it. Top-1 agreement, calibration and pairwise accuracy can be computed from its saved forecasts at no cost; they would be labelled post hoc there.

## 8. Data checklist for public rating sets

1. **Individual ratings released** (not only the mean, majority or a filtered mean).
2. **Ratings per item**, and whether it varies; enough to estimate a distribution (about ≥ 5) or to subsample from.
3. **Scale K** and whether it is ordered (RPS needs order; for unordered labels use CE and TVD).
4. **Rater identity:** anonymous id, same raters across items, rater metadata (needed for rater-facet models).
5. **Item structure:** prompt, task or family id (for family holdouts); whether pairs share a prompt.
6. **Licence:** commercial use, redistribution, attribution.
7. **Selection bias:** how items and raters were chosen (for example items picked for disagreement) and the rater quality filter.
8. **Same object as the judge sees:** the judge gets exactly what raters saw.
9. **Contamination:** possible exposure of public items to model pretraining (unknown unless stated).
10. **Sensitive content** and terms for model use.

**Shortlist of public sets already checked** (read 11-21-2026 from each set's README, data card or paper; DICES row counts computed from the released CSV files; HelpSteer2 counts from the released file).

| Set | Releases | Scale, rater identity | Licence | Watch for |
|---|---|---|---|---|
| ChaosNLI ([paper](https://arxiv.org/abs/2010.03532), [repo](https://github.com/easonnie/ChaosNLI)) | 100 new labels for each of 4,645 examples: 3,113 from SNLI and MNLI, 1,532 from Abductive-NLI, as label counts per example (`label_counter`), plus the earlier original labels; model logits for several models | K = 3 (entailment, neutral, contradiction) for SNLI and MNLI; K = 2 for Abductive-NLI. No rater id in the released format | CC BY-NC 4.0 (non-commercial), per the repo README | SNLI and MNLI parts are only items whose majority label matched three of the five original labels (borderline agreement), so they lean toward disagreement; the Abductive-NLI part is the entire development set, so it is not selected. Mechanical Turk workers. With K = 2 the mean fixes the whole distribution (W1), so Abductive-NLI can test level and scale only |
| DICES ([paper](https://arxiv.org/abs/2306.11247), [repo](https://github.com/google-research-datasets/dices-dataset)) | DICES-350: 350 conversations, each rated by all 123 raters (43,050 rows). DICES-990: 990 conversations, 69 to 76 ratings each, 172 raters (72,103 rows). One row per (rater, conversation), with rater metadata, safety categories and a gold label | Yes / No / Unsure per category (unordered or weakly ordered); anonymous rater ids with demographics | CC BY 4.0 | Adversarial safety conversations with a contested ground truth; unordered labels, so CE and TVD, RPS only with a stated ordering; the authors apply a rater-quality filter (104 of 123 raters in the 350 set) but release all |
| HelpSteer2 ([paper](https://arxiv.org/abs/2406.08673), [data](https://huggingface.co/datasets/nvidia/HelpSteer2)) | 21,362 prompt-response samples with five attributes (helpfulness, correctness, coherence, complexity, verbosity). The main label is the mean of the three most agreeing annotators (3.41 annotators on average); a separate `disagreements` file (added 1 Oct 2024) gives individual annotations as lists: 23,652 rows with 2 to 6 annotations each (1,162 with 2, 16,390 with 3, 1,314 with 4, 4,724 with 5, 62 with 6) | Integer 0 to 4, K = 5, ordered; annotators not identified | CC BY 4.0 | Main labels are filtered means, so use the `disagreements` lists; the number of annotations depends on difficulty (more annotators were added when the first three disagreed), so ratings per item are not independent of the item |

Not yet surveyed: other public rating sets. The literature survey is a separate task, planned for Monday; it will fill this table with the same columns.

## 9. What this note does not do

It runs nothing and cites no new results. It fixes definitions and decision rules on 11-21-2026; changes after any new data are looked at are added as dated amendments, never as edits. Open points: the 23,652 rows of HelpSteer2's `disagreements` file do not equal the 21,362 main samples, so matching rows to main samples needs the prompt and response text; ChaosNLI's per-worker identity is not released, so rater-facet models are not possible there.
