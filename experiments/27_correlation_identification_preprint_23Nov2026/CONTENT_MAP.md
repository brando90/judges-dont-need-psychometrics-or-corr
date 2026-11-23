# Derivation-to-paper map — 11-23-2026

The standalone article is `PAPER_11-23-2026.tex`, which inputs one source file per section from `sections/`.
`PAPER_11-23-2026.md` contains the full matching article, including every proof and displayed derivation.
`build_markdown.py` reproduces that Markdown from the section sources using only Python's standard library.

## Source-to-paper map

| Source derivation | Final location | Preserved content and qualification |
|---|---|---|
| R1 | Proposition 1, Section 3 | Centering cancellation; probability-linear numerator; complete dependence on the vector of conditional means. |
| R2 | Propositions 1–2, Section 3 | Positive affine invariance, exact correlation-one equality condition, feasibility restriction, interior dimensions and boundary caveat. |
| R3 | Proposition 3, Section 3 | Full mean and softmax-logit gradient derivation; summed CE scaling; sign for minimized correlation loss; feasible first-order versus exact path distinction. |
| R4 | Proposition 7, Section 4.1 | Exact CE decomposition with entropy gap and signed cross term; cancellation for log-affine forecasts; negative cross-term example at unrestricted truth. |
| R5 | Proposition 8, Section 4.2 | Irreducible individual-rating RPS noise floor; normalized excess; signed CDF identity; correct `(K-1)^2` mean-error denominator; TVD mean bound. |
| R6 | Propositions 5 and 8, Sections 4.1–4.2 | Strict propriety of CE and individual-rating normalized RPS, TVD as true-distribution distance, and explicit failure of one-hot TVD as a proper score. |
| R7 | Proposition 4, Section 3 | Within-item probability-profile Pearson construction; item-specific contraction factors versus common stacked factor; uniform-profile undefined cases; distinct scope from mean correlation. |
| W1 | Proposition 2 and following paragraphs, Section 3 | Binary mean determines distribution, but feasible positive affine mean errors persist; exact excess RPS and KL formulas. |
| W2 | Proposition 2 and following paragraphs, Section 3 | `K-2` independent interior shape directions; three-category family `(s,1-2s,s)` and variance `2s`; full optimum-set dimension. |
| W3 | Proposition 6 and following derivation, Section 4.1 | Regular finite natural exponential family with positive base measure and statistic `r`; positive variance; mean-map bijection; CE moment matching, boundary limits, misspecification caveat, local `(m-t)^2/(2V(t))` expansion. |
| W4 | Section 5.1 | Exact global feasible affine-copy criterion; parameter symmetries versus predictive identification; separate local criterion and sufficient derivative-rank condition; weak-logit Taylor expansion as approximate symmetry. |
| W5 | Proposition 11 and following discussion, Section 5.2 | OLS direction and squared maximum correlation; covariance assumptions; zero feature-target covariance case; feasibility; softmax-linear CE feature-frequency gradient. |
| W6 | Proposition 3 and Section 5.3 | Nonlinear input-to-mean functions remain available; exact mean-preserving paths are invisible; parameter sharing can alter shape indirectly. |
| W7(i) | Section 6.1 | Heteroskedastic conditional mean-zero rating noise; total covariance and variance derivation; held-out population attenuation; varying rating counts; finite-sample coefficient distinction. |
| W7(ii) | Sections 6.2–6.3 | `P/(2N)` nats only as a regular, well-specified, interior iid MLE asymptotic with expectation/moment control; no universal crossover; same-unit heuristic only for a separately specified procedure with known positive limiting excess. |
| W7(iii) | Section 4.3 and Section 6.3 | Equal-mean candidates cannot be selected by the mean-correlation objective; no universal finite-data CE ordering. Archived tiny-sample numerical results omitted because not replayed. |
| Additional first-moment discussion | Proposition 9, Section 4.2 | Bounded normalized moment control by TVD and square-root KL, with a self-contained finite-alphabet proof. |
| Mean-only recovery limits | Proposition 10, Section 4.3 | Two-truth TVD, RPS, and KL lower bounds; midpoint sharpness for TVD/RPS; randomized-rule statement; noise floor distinguished from excess score. |

## Exact illustration and verification

`experiments/exact_counterexample/reproduce.py` defines the exact strictly positive three-item, three-category construction in Section 7.
It fits no model, samples no ratings, and invokes no external service.
All probability, CDF, TVD, and RPS arithmetic uses exact rational numbers.
The script verified means `(1/2,1,3/2)` for both forecasts, Pearson `1`, average TVD `1/5`, average excess RPS `3/200`, true expected RPS `49/300`, and changed-shape expected RPS `107/600`.
It evaluates CE from the exact inputs using standard-library logarithms, yielding `0.915387330` and `1.064735095` nats and an unrounded-difference excess of `0.149347765` nats to the displayed precision.
Its assertions also verify positive probabilities, total mass, and the signed CDF identity.

## Deliberate omissions and corrected scope

- Removed duplicate derivations, notes to the reader about drafting, historical bylines, planning language, and private correspondence.
- Removed source labels R1–R7 and W1–W7 from rendered headings; the table above retains traceability without exposing working-note structure in the article.
- Omitted all empirical sweep tables, measured crossover claims, historical calibration-head results, numeric checks of logistic fitting, and archived numerical examples because this preprint does not reproduce their underlying experiments.
- Replaced the archived numerical illustration with a new exact construction whose complete arithmetic is supplied and reproduced.
- Omitted any claim of full theorem-prover verification; the preprint contains detailed mathematical proofs and makes no claim about compiling formal proofs.
- Replaced the blanket finite-design logistic identification assertion with the exact feasible affine-copy criterion and its local/global qualifications.
- Restricted moment matching to the stated natural exponential family; an arbitrary one-parameter family or a beta-binomial family with fixed dispersion does not inherit the proof.
- Removed a universal positive correlation error floor and any universal CE break-even claim; optimum-set ambiguity alone does not force a fitting procedure to select an inaccurate forecast.
- Kept only the two requested scoring-rule references and made the substantive mathematics self-contained.
- Added no claims of novelty priority, peer review, trained-model superiority, or generalization from sampled human ratings.

