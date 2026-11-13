# 04: Falsifiable plan for reliability versus distribution accuracy

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/PLAN.md>

**TLDR:** First test whether matching means, variances and reliability coefficients can leave human-rating distributions wrong. Then test generalizability theory on its own evaluation-design objective, including a condition where it should succeed; do not turn a prediction-target mismatch into a claim that the theory is false.

**Executed 11-13-2026:** see [RESULTS_reliability_vs_accuracy.md](RESULTS_reliability_vs_accuracy.md); the executable protocol is [protocol.json](protocol.json). This plan is kept as written.

**Status:** PROPOSED — NOT RUN, 11-13-2026. Mathematical constructions below are predictions to verify, not measured results or machine-checked proofs. Proposed grids and tolerances are agent choices, not deployment standards. Commit executable definitions, configuration and expected outcomes before measurement; later changes become a new version.

## 1. Claims and attribution

A **generalizability (G) study** estimates variance components; a **decision (D) study** uses them to assess a proposed measurement design. For distribution prediction, this plan uses normalized **ranked probability score (RPS)**, **cross-entropy (CE)** and **total variation distance (TVD)**, defined below.

| Claim to examine | Whose claim? | Evidence that would reject it | What rejection would not establish |
| --- | --- | --- | --- |
| Matching human rating variance maximizes reliability coefficients | A hypothesis to test, not a claim attributed to the paper | Equal variance with different object/error allocations and therefore different coefficients | That human variance is irrelevant to distribution prediction |
| High reliability coefficients identify the correct human-rating distribution | A possible stronger interpretation to test; not claimed in the supplied paper | Reliable biased forecasts, or distinct distributions sharing all relevant first and second moments | That G studies or D studies are invalid |
| An aggregate coefficient certifies each task | A possible misuse, not a claim attributed to the paper | A subgroup with large prediction error hidden by aggregate summaries | That task-specific uncertainty cannot be studied |
| Variance components can guide allocation of measurement effort | Explicit paper claim | Under the specified model and cost, predictions systematically fail on independent repetitions | Failure under a different population or dependent observations is not an in-model refutation |
| Proper losses make all psychometric tools unnecessary | A broad reading of the repository title, not its supported thesis | Accurate D-study allocation when distribution loss alone does not specify the allocation | That proper losses cease to be appropriate for distribution prediction |

The strongest initial target is **sufficiency**, not a contest over which field is correct. Preserve evidence against our preferred interpretation, including successful D-study predictions.

## 2. Define the target before measuring

### Probability prediction

For a fixed rated artifact x (task, solver output and reference version), let p_x(y) be the conditional distribution of rating y from the specified human population and q_x(y) a judge's forecast on y in {0,1,2,3,4,5}. Distinguish:

1. **Human conditional variance:** Var(Y | x), disagreement in that population.
2. **Across-artifact variance of means:** Var_x(E[Y | x]).
3. **Marginal rating variance:** Var(Y) = Var_x(E[Y | x]) + E_x Var(Y | x).
4. **Judge repeat variability:** variation in q_x itself across calls, or in a scalar rating returned by a stochastic judge. These are separate quantities.
5. **Sampling uncertainty of an aggregate:** variation in an estimated system score when tasks, raters or occasions are resampled.

Variance is a central second moment, not generally the raw second moment E[Y²]. Matching only marginal variance fixes neither the allocation in the variance identity nor task-conditioned predictions. Matching conditional variance alone does not fix conditional means or the full joint dependence across raters. A deterministic probability vector can correctly represent variable human ratings without changing between calls.

### Object of measurement and facets

A **facet** is a source of measurement variation, such as rater identity or occasion. A coefficient must name the **object of measurement** whose differences are of interest, the target population and what is averaged.

- **Artifact-level design:** artifacts are objects; human raters and rating occasions are facets. A G coefficient summarizes precision across this artifact population; it is not a separate verdict for each artifact. Task-family clustering remains relevant.
- **System-level design:** solver systems are objects; tasks and raters are facets. Each system's average over tasks and raters is the measured score. Task-by-system interactions now enter measurement error.
- **Fixed-benchmark versus new-task generalization:** treating tasks as fixed or random changes the estimand. A coefficient across systems requires variation across systems; a single system does not identify a between-system component.

Do not infer a design merely from the phrase “aggregate over items.” This plan evaluates the published framework under explicitly constructed models; it makes no claim about the design or results of any unpublished study.

### Working crossed random-effects model

For object u and two random facets a and b, use the balanced model

```math
X_{uab}=\mu+U_u+A_a+B_b+(UA)_{ua}+(UB)_{ub}+(AB)_{ab}+E_{uab}.
```

Each effect has mean zero; effects are mutually independent in the base condition. The residual E includes the highest-order interaction and unmodeled error when there is no within-combination replication. Let v_u, v_a, v_b, v_ua, v_ub, v_ab and v_e denote the corresponding nonnegative variance components. For an average over n_a and n_b sampled facet levels:

```math
v_{\delta}=v_{ua}/n_a+v_{ub}/n_b+v_e/(n_a n_b),
```

```math
v_{\Delta}=v_{\delta}+v_a/n_a+v_b/n_b+v_{ab}/(n_a n_b),
```

```math
E\rho^2=v_u/(v_u+v_{\delta}),\qquad
\Phi=v_u/(v_u+v_{\Delta}).
```

Here Eρ² is the **relative generalizability coefficient** and Φ is the **absolute dependability coefficient**. Φ is this plan's notation for the paper's absolute analogue; the notation does not identify any unpublished implementation. These formulas concern this random-facet average, not every nested or fixed-facet design. The paper's pages 924–926 supply the construction; implement and verify the exact averaging factors rather than trusting extracted equation text.

Always report error variances and their square roots alongside the ratios. With nonnegative components and nonzero denominators, Φ ≤ Eρ². A zero denominator means undefined, not perfect reliability. High reliability is relative to a chosen universe score, which need not equal the desired human target.

## 3. Phase A — cheapest sufficient counterexamples

**Proposed resources:** deterministic local arithmetic; no model calls, human collection or graphics processors. Bound the first implementation to one small script and its tests, not a training pipeline. Use exact rational arithmetic for moment identities where possible; generate every displayed numerical value from saved script output.

### A1. Same mean and variance, wrong distribution

Let p_x be any strictly positive distribution over the six categories. Define

```math
v=(-1,3,-3,1,0,0),\qquad q_x=p_x+\epsilon_x v,
\quad 0<\epsilon_x\le \min_y p_x(y)/6.
```

With category vector y=(0,1,2,3,4,5) and elementwise multiplication, check that the entries of v, yv and y²v each sum to zero, that q_x is normalized and positive, and that q_x differs from p_x. Thus the proposed pair preserves conditional mean and variance while changing distribution shape. Use varying p_x means across artifacts so correlations and between-object variance are nondegenerate. Reuse the existing binomial construction by reference, without modifying its frozen files.

**Predictions to verify:** scalar correlations of the predicted means with the true means tie; conditional means and variances match; expected RPS and CE favor p_x. For a rating panel generated independently conditional on each artifact, verify that the two complete panels have the same first and second moments, hence the same population moment-based variance components and coefficients. Matching univariate moments alone is not enough for a panel with shared rater effects; prove the joint moment condition for the chosen design.

Distinguish **population coefficient equality** from equality of finite-sample estimates: different higher moments can change estimator uncertainty. If analyzing deterministic forecast means instead of sampled ratings, label that separate analysis; it cannot measure within-forecast human disagreement.

**Verdict rule:** an exact moment-preserving construction with positive expected excess RPS rejects identification of the full distribution by those moment summaries. It is a constructed example, not an empirical frequency estimate or a new criticism of a claim made by the paper.

### A2. Perfect repeatability, wrong human scores

Construct a varying deterministic scalar judge that is wrong on a predeclared subset of artifacts. With no facet variation, its own nondegenerate relative and absolute reliability coefficients should be maximal. Its external mean absolute error and proper losses should expose the errors. Include a constant judge as a separate degenerate control: no between-object variation, undefined coefficient when both numerator and denominator vanish.

For a shared-bias control, choose human ratings supported on {1,2,3} and shift the judge's entire distribution up by one category to {2,3,4}, without clipping. Analyze human and judge panels separately. Predict invariant internal variance components and coefficients, but positive judge-minus-human mean bias and worse expected proper losses. CE may be infinite because of zero probabilities; retain infinity rather than silently smoothing it. A separate positive-probability condition may be declared in advance if finite CE is needed.

**Important boundary:** an absolute coefficient includes varying rater severity under its design; it is not universally blind to offsets. A shared offset across a separately analyzed panel changes its grand mean, not its variance components. Pooling human and judge ratings as different raters can expose that offset. Report which analysis is performed.

### A3. Oracle prediction versus sampled-score reliability

Use the exact human distribution q_x=p_x as the oracle forecast. Evaluate both a deterministic return of this probability vector and independent categorical samples from it. The oracle should minimize expected proper losses, not make them zero: irreducible human disagreement remains. For independent samples with no shared facet effects, the relative coefficient for an n-rating mean has the form

```math
\frac{\operatorname{Var}_x(m_x)}
{\operatorname{Var}_x(m_x)+\mathbb E_x[s_x^2]/n},
\quad m_x=\mathbb E[Y\mid x],\quad s_x^2=\operatorname{Var}(Y\mid x).
```

It need not be maximal when s_x² is positive. A deterministic report of m_x has no repeat sampling error but does not represent p_x unless accompanied by its distribution. Change between-artifact spread while keeping within-artifact variance fixed to examine population dependence of the ratio; report absolute error as well.

**Verdict rule:** distinguish “matches human variability,” “estimates the human mean precisely,” and “has a repeatable output.” Do not maximize a sampled-score coefficient and then claim to have optimized human-distribution accuracy.

### A4. Equal total variance, different design recommendations

Hold the total variance and between-object component fixed, but move nuisance variance between rater and occasion components, including their object interactions. Predict different benefits from adding raters versus adding occasions. This directly tests whether a single variance-matching diagnostic contains enough information to answer the D-study allocation question.

## 4. Measurement contract for Phase A

Use normalized individual-rating RPS, `sum_{k=0}^4 (Q_x(k)-1{y<=k})²/5`, averaged over ratings within each artifact and then equally over artifacts. CE is `-ln q_x(y)`, with no implicit floor. Synthetic population TVD is `sum_y |q_x(y)-p_x(y)|/2`; empirical TVD later uses observed human frequencies and is explicitly a different, noisy comparison. Track Pearson, Spearman and Kendall separately when defined; mean absolute error; bias = judge mean minus human mean (positive means lenient); both means; conditional variance discrepancy; repeat variability; variance components; and both coefficients.

| Control | Distributional prediction | Scalar/variance prediction | Internal reliability prediction |
| --- | --- | --- | --- |
| Exact distribution oracle | Lowest expected RPS/CE; population TVD zero | Correct means and conditional variances | Depends on object spread, facet design and whether scores are sampled |
| Same-moment perturbation | Positive excess expected RPS/CE and population TVD | Same moments and tied mean-based correlations | Same population components when joint second moments match |
| Shared-offset panel | Worse RPS; CE may be infinite | Unchanged conditional variance; nonzero bias; mean-based correlations unchanged | Unchanged when analyzed as a separately shifted panel |
| Repeatable task-dependent errors | Worse proper losses on affected tasks | External error visible | Maximal if nonconstant across objects and no facet error |
| Constant deterministic judge | Depends on target; cannot be an oracle for varying distributions | Constant predicted mean; correlations undefined | Undefined in the all-zero-variance case |
| Training-marginal forecast | Population baseline; not generally conditionally optimal | Constant mean; conditional mismatch possible | Analyze output format explicitly; do not label constancy as validity |
| Shuffled artifact forecasts | Same marginal forecast collection, wrong conditional pairing in a nontrivial construction | Marginal matching may survive; task accuracy need not | Not predicted from marginal matching alone; compute under the declared design |
| Missing/invalid forecast | Failure retained; no invented loss or imputed distribution | Missing | Not a valid reliability observation |

The implementation must attach a code path and numeric or symbolic expected value to every emitted metric before running. Resolve any unpredicted value before interpretation. No language-model prompt is delivered in Phase A: literal compliance means returning the exact specified probability vector, which must survive serialization and parsing unchanged. Trace one artifact from construction through forecast, optional rating sample, metric and aggregation.

## 5. Phase B — give D studies a fair test

**Question:** Under a known model, does variance-based allocation choose a more precise aggregate than a rule that only matches marginal variance or allocates effort equally?

Use the crossed model above, explicitly on a continuous score scale for this variance-component control; do not round or clip Gaussian samples into ordinal ratings and then claim the same variance formulas remain exact. This phase does not compute categorical RPS or CE.

**Proposed synthetic configuration, to freeze in executable form:**

- Independent, zero-mean Gaussian effects; grand mean zero.
- Variance vector in the order `(u,a,b,ua,ub,ab,e)`: `(1,0.6,0.1,0.6,0.1,0.1,0.5)`, plus a second condition swapping a and b and their object interactions. These are illustrative agent-chosen settings, not estimates from humans or the paper.
- Calibration G study: 64 objects crossed with 8 levels of each facet; 256 independent replications, seeds 0 through 255. These are proposed simulation sizes, not a human-study power calculation.
- Feasible D-study designs: n_a,n_b in {2,4,8}. Compare designs with the same n_a*n_b when per-rating cost is equal; separately report cost frontiers instead of comparing unequal budgets as if matched.
- Fit variance components from calibration data using a prespecified constrained restricted maximum likelihood estimator. Compare with known population components and an independently implemented balanced analysis-of-variance calculation; retain raw negative estimates and label any constrained alternative.
- Choose the design using calibration data only. Evaluate it on independently drawn facet levels/observations for the same held-out design evaluation, keeping the object universe and target fixed. Replicate the complete estimate–choose–evaluate procedure, not just the last coefficient calculation.
- Baselines: equal allocation, more a levels, more b levels, and the oracle allocation using known components. A direct mixed-effects implementation is a computational equivalence check, not a different scientific theory.

**Endpoints:** observed squared error relative to each known universe score (absolute); centered errors across objects (relative); predicted versus observed error variance; Eρ² and Φ; excess decision error relative to the oracle allocation; cost. The D-study design endpoint is intentionally different from distributional RPS. Judge selection, if subsequently studied, still uses the repository's common validation RPS.

**Then, separately, test known vulnerabilities:** duplicate/correlated raters, heterogeneity across predeclared task groups, and a changed task population. Keep the same total nominal rating budget. Do not treat duplicate measurements as independent humans. Report precisely which independence, exchangeability or stationarity assumption changed; the paper already discusses these issues (pages 926–928).

**Decision rules:**

- If population formulas match independent simulation within Monte Carlo uncertainty and allocation lowers paired error at equal cost, retain the design-usefulness claim for that model.
- If a formula fails with known components, inspect the implementation and estimand before asserting a theory failure. A poor component estimate in a small calibration study is distinct from a wrong population formula.
- If forecasts degrade only after dependence or population shift, report robustness limits, not an in-model refutation.
- If uncertainty spans no allocation benefit, report inconclusive evidence; do not claim equivalence or superiority.

For stochastic headlines, report paired differences and 95% intervals over independent simulation replications, plus Monte Carlo uncertainty for coverage estimates. No confirmatory hypothesis test is specified: `p-val=n/a`. Exact identities receive exact verification, not confidence intervals. No universal “reliable enough” threshold is adopted; an application threshold requires an externally justified decision cost.

## 6. Optional Phase C — existing human data, only if informative

First inspect the authorized historical rating schema and crossing pattern: task families, solver systems, artifacts, rater identities, repeat counts, missingness and previous exposure. Do not unseal test ratings, change the collection, or turn an absence of repeated occasions into a zero occasion-variance estimate.

If components are identifiable, analyze predeclared designs on matched human/judge input bytes and report forecast losses beside reliability and external absolute agreement. Use the current applied evaluator's metric definitions. Report intraclass correlation for absolute agreement, ICC(A,1), beside consistency, ICC(C,1), when the design supports them; these are not automatically the same as the two-facet coefficients above. Cluster uncertainty at task family and model sampled raters appropriately; a task-only bootstrap is conditional on the observed raters.

Any selection compares the same candidate pool, data, model family, information access and search budget. Fit on train, select all hyperparameters by validation RPS, and evaluate the same held-out panel. Previously exposed historical ratings support a retrospective analysis only. A reliability-based selector, if added, is a separately labeled diagnostic comparison, not a silent replacement of the agreed primary criterion.

If the existing data do not identify the desired components or lack sufficient candidate variation, stop this phase with that reason. Coordinate any future data requirement with the [sole collection owner](https://github.com/brando90/trustworthy-theorem-judges/tree/main/experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026); do not start a second rating workflow here.

## 7. Implementation and stopping rules

1. Implement `scripts/check_counterexamples.py` and `scripts/test_metrics.py` for Phase A. Check normalization, positivity, moments, expected-loss identities, coefficient invariance, zero denominators, and serialization. Do not overwrite prior experiments.
2. Save exact inputs and per-artifact outputs, software versions, command, commit and hashes. Generate `RESULTS_reliability_vs_accuracy.md` from saved outputs. Keep experimental results explicitly separate from this plan.
3. Implement Phase B only after the formulas and attribution checks pass. Save the proposed grid and expected metrics before execution; keep all failed fits and missing repetitions, with conditional estimates labeled.
4. Decide whether Phase C adds information beyond the analytic result and the existing historical comparisons. Reuse authorized data before proposing new collection or model calls.

**Continue:** verified sufficiency counterexamples plus working positive controls support a narrow report about distinct targets. **Redirect:** if the practical claim is only better design allocation, study that question rather than distribution identification. **Stop the broad debunking claim:** successful allocation or a mismatch between our target and the source claim rules out presenting this work as a general refutation of psychometrics. **Leave empirical outcomes pending:** no claim about deployment, finite-sample objective superiority or human generalization follows from a toy construction alone.

**TLDR-end:** [judge objectives: 04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026] Test whether reliability and moment matching are insufficient for accurate human-rating distributions, while directly checking the legitimate design-allocation benefit. The experiment can reject a sufficiency claim without rejecting generalizability theory.
