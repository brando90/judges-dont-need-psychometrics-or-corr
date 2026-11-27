# Measurement contract — experiment 42, 11-27-2026

This is a simulated-population testbed. Passing a cut here does not certify a judge on human raters.

## What a match would mean

The exact risk is the expected ranked probability score (RPS), cross-entropy (CE), and total variation distance (TVD) of a train-fitted forecast under a known law.

- **Oracle.** The true conditional distribution of the named target. It minimizes expected RPS and CE.
- **Intended distributional learner.** Add-one cross-entropy on the persona the deployment names. On the polarized target it should approach the polarized law as the quota grows, and it should be calibrated to that persona (bias near 0, TVD near 0).
- **Pooled learner.** Add-one after ignoring the persona label. It estimates the equal train mixture. On the polarized target it should keep a fixed shape error. At very small quotas it can beat the persona split, because the split sees only one quarter of the ratings.
- **Pearson binomial.** Matches the bin mean. Pearson correlation of its means with the true bin means should stay near 1 on every target whose bin means are an affine function of 1.6, 2.0, …, 3.6, including the strict persona (a shift of −0.8). TVD on the polarized law should stay large.
- **Lenient point and Spearman point.** Point masses. Own-source Eρ² should be 1 whenever the six locations are not all equal. RPS should be worse than the constant-shape binomial.
- **Maximum Eρ².** Should select the lenient point from the pool that contains it, because a point mass has within-bin variance 0.
- **Maximum own-source Φ.** Should prefer a peaked persona (strict or lenient) over the polarized persona, and that choice should lose to persona cross-entropy on RPS of the polarized law.
- **Validation RPS.** Should not select the lenient point once the quota is large.

## Cuts

Exact, from the known laws:

- Each persona mean matches its target within 1e-7.
- Pearson correlation of the binomial means with the polarized means is above 0.999, TVD is above 0.50, and the RPS gap versus the oracle is above 0.04.
- The lenient point has Eρ² = 1.
- Against the strict law, the binomial bias is +0.8, Pearson correlation is above 0.999, and TVD is above 0.40. The strict oracle has TVD 0.
- The two equal-moment panels have the same Eρ². Their average RPS differs and their TVD is above 0.25.

The first draft of the moment cut asked for an average RPS gap above 0.01. That threshold was wrong: the three-item average gap is 167/23040, about 0.00725, while the center pair alone (M2 scored on M1) is 9/512, about 0.01758. The cut was corrected to the statement above after that exact value was computed, and the Monte Carlo was re-run. The other cuts were not edited. Lean checks the center gap 9/512 and the panel coefficient 1/5.

Finite sample, 400 replications, Student-t 95% intervals. The paired contrasts are replication-level differences, not differences of the marginal intervals:

- At 256 ratings, binomial RPS minus persona cross-entropy RPS on the polarized law has lower endpoint above 0.03.
- At 256 ratings, persona cross-entropy RPS minus the polarized oracle has upper endpoint below 0.01.
- At 256 ratings, binomial Pearson correlation has lower endpoint above 0.99 and TVD has lower endpoint above 0.40.
- At 64 and at 256 ratings, maximum Eρ² selects the lenient point in at least 95% of replications.
- At 256 ratings, validation RPS selects the lenient point in at most 5% of replications, and the selected forecast's RPS minus the lenient point's RPS has upper endpoint below 0.
- At 256 ratings, maximum Φ selects strict or lenient in at least 95% of replications, and that choice's RPS minus persona cross-entropy RPS has lower endpoint above 0.
- At 256 ratings, the binomial's bias on the strict law has lower endpoint above 0.7, its Pearson correlation has lower endpoint above 0.99, its TVD has lower endpoint above 0.3, and the strict persona cross-entropy bias has absolute mean below 0.05.
- At 16 training ratings and 32 disjoint test ratings per item, the interval for sample RPS minus exact RPS of the persona cross-entropy forecast covers 0.
- The item-clustered percentile interval for one n=64 fit covers that fit's exact RPS.
- Two runs of the Monte Carlo write the same summary bytes, ignoring wall-clock time.
- The leakage checks pass: the fit does not read the test array, rater ids do not cross splits, and item ids do not cross splits.

## Seeds

`SeedSequence([20261127, scenario, n, replication, stream])`. Stream 1 is training, stream 2 is validation, stream 3 is the sample-risk check. NumPy Generator PCG64.
