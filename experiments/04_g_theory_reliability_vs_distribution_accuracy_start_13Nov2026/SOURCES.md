# 04: Sources, claim attribution and reading record

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/SOURCES.md>

**TLDR:** The source is the published 1989 introductory article *Generalizability Theory*. It supports variance decomposition and evaluation-design planning; it does not claim that reliability coefficients identify a correct human-rating distribution.

## Supplied paper — identity verified from the document

Richard J. Shavelson, Noreen M. Webb and Glenn L. Rowley. **Generalizability Theory.** *American Psychologist*, volume 44, number 6, pages 922–932, June 1989.

- User-supplied source: <https://drive.google.com/file/d/1lftgBiJRKw89dl_aBzURBTof-eTKFO20/view?usp=sharing>.
- Retrieved on 11-13-2026 using the public Drive download endpoint, without authentication.
- Source receipt: [source_manifest.json](source_manifest.json), including the PDF's SHA-256 digest (Secure Hash Algorithm, 256-bit).
- Reading scope: all extracted text from all 11 PDF pages, including the references; page 925 also rendered and visually inspected to check the coefficient definition and relative/absolute distinction. The text extraction contains optical-character-recognition artifacts, especially in equations; it is not an authoritative transcription.
- Local retained copy: `sources/local/shavelson_webb_rowley_1989_generalizability_theory.pdf`; extracted text and the inspected page are alongside it. These reading copies are ignored by Git. The PDF carries an American Psychological Association copyright notice; no redistribution license was verified, so the public experiment records its citation, link and hash rather than republishing the full text.
- This is a source reading in Pi with `gpt-6-astra` (session records `xhigh` effort), not a Lean-checked proof or an executed statistical experiment. Differential testing is not applicable to this source-reading task. No experimental model calls were made.

## Page-specific claim map

| Printed page (PDF page) | Verified content | Consequence for the plan |
| --- | --- | --- |
| 922 (1) | Defines a universe of generalization and a universe score; emphasizes generalizing from sampled observations to a specified set of situations | Name the population, object and acceptable changes of conditions before interpreting a coefficient |
| 923 (2) | Multiple variance sources and interactions; distinguishes relative/absolute decisions and generalizability (G)/decision (D) studies | G and D are study types; a single variance statistic does not replace a decomposition |
| 924–925 (3–4) | Variance components; relative error from object interactions; absolute error also includes facet-level effects; ratio of universe-score variance to expected observed-score variance | Derive the correct coefficient for the exact averaging design; do not equate internal absolute reliability with external human agreement |
| 925–926 (4–5) | G study estimates components; D study specifies the universe and proposed use and considers alternative numbers of facet levels | Include an independent, cost-matched design-allocation test |
| 926 (5) | A single judge can yield high reliability for a narrow universe; multiple judges can generalize to a more interesting universe; tasks versus observers example | A stable single judge is not automatically better for the desired population; this warning is already in the source |
| 926–927 (5–6) | Steady-state assumption, systematic change and dependent successive observations; dependence can inflate estimated coefficients | Dependence and population-shift tests examine acknowledged limitations, not a newly discovered contradiction |
| 927–928 (6–7) | Unstable or negative component estimates in small samples; estimation alternatives; fixed facets, exchangeability and changing the object of measurement | Distinguish finite-sample estimation from population formulas and fixed-benchmark from new-task inference |
| 928–929 (7–8) | Statistical weighting to maximize a coefficient may conflict with conceptual justification | The paper already warns against optimizing a coefficient without regard to the target construct |
| 929–931 (8–10) | Job-rating application, rater/occasion components, multivariate covariance and alternative composites | Do not attribute a claim that marginal variance alone describes measurement error |
| 931–932 (10–11) | Summary and references | Introductory source, not evidence that any current large language model judge is valid |

## Public scope and attribution boundary

The user requested this experiment on 11-13-2026. It operationalizes a proposed critique as falsifiable claims, including counterexamples to variance matching and positive controls for the cited framework. A desired conclusion is not evidence.

This public experiment uses the published article, the repository's existing evidence and prospective synthetic constructions. It includes no unpublished slides, private correspondence, study-specific findings or claims about a contemporary researcher's implementation. Any later analysis of nonpublic work must remain separate and within its authorized audience.

The relative coefficient Eρ² and absolute coefficient Φ are explicitly defined working quantities in [PLAN.md](PLAN.md). No reported result from another study has been reproduced or refuted.

## Repository sources inspected

- [Repository thesis and limitations](../../README.md).
- [Experiment index and current path history](../README.md).
- [Matched comparison](../02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/RESULTS_comparison.md) and [checkpoint](../02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/ckpt.md): preserve the null/inconclusive comparison and prior-exposure limitations.
- [Prospective primary and secondary evaluation definitions](../02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/EVALUATION_CRITERION_11-09-2026.md): ranked probability score primary; cross-entropy and total variation distance secondary.
- Active paper abstract and problem statement: both explicitly exclude a conclusion that psychometric methods are unnecessary.

## Google companion

Pending synchronization destination: the existing [Trustworthy LLM Judges: Research Notes and Experimental Framing](https://docs.google.com/document/d/1WI-raRtTBnQETjkyMzg23PdhgDLgn33A9kQY1EU0zyE/edit), linked from this repository's earlier research synthesis. No Google write or delivery is claimed. The exact pending addition is:

> Experiment 04 — reliability versus human-rating distribution accuracy (planned, 11-13-2026). The supplied introduction by Shavelson, Webb and Rowley (1989) motivates variance decomposition and evaluation-design allocation, not distributional accuracy guarantees. We will test same-moment/different-distribution and repeatable-but-wrong counterexamples, while retaining a positive control where decision-study allocation should work. No experiment has run and no contemporary study has been refuted. Plan: https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/PLAN.md

**TLDR-end:** [judge objectives: 04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026] The full supplied article has been read and its claims mapped to tests. Its design-usefulness claim must be kept separate from stronger sufficiency claims that it does not make.
