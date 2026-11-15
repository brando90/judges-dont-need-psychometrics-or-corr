# Mega QA stage 2: experiment 04 and the reliability appendix

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/qa/mega_qa_11-14-2026/qa2_codex_vals.md>

**TLDR:** **Accept with minor open issues.** The required rerun reproduced the saved measurements: 945/945 formula checks and 134/134 recomputation checks passed, with the same three frozen prediction misses. I corrected scientific overstatements, disclosed selection of the historical comparison pair on the test results, corrected stage 1's mathematical explanations, and rebuilt the six-page paper. No frozen measurement code or numerical result changed. One unimplemented secondary analysis remains explicitly documented; no critical or major finding remains open.

## Reviewer and bounds

- Stage: `qa2_codex_vals`; date: 11-14-2026; launch: `2026-11-14T19:58:11-08:00`.
- **Actual model: `gpt-6-astra`; actual effort: `ultra`; harness: OpenAI Codex CLI v0.169.0; account profile: Vals (`~/.codex-vals`).** Verified from this stage's launch command and terminal startup receipt, which records provider `openai`, approval `never`, sandbox `danger-full-access`, and session `01a1dd5b-0931-79e3-b2b8-d17d20092a02`. The effective launch override is `ultra`, despite the profile file's `high` default. My initial chat statement that the exact settings were unverified was superseded by this receipt.
- Owned checkout: `/Users/brandomiranda/.worktrees/judges-dont-need-psychometrics-or-corr/mega-qa-04-1114`; input HEAD: `9e4b243518cb28538d4aa2284b76151f967d3555`; builder: `03ada1e`; base: `d9d1c1d`.
- Read `COMMON.md` and then `BRIEF_qa2_codex_vals.md` in full. Reviewed the builder diff and experiment independently before reading stage 1's report/marker. The timestamped independent notes are in the runtime directory.
- This is mathematical/code review plus executed Python checks, **not a Lean-checked proof**. Differential testing of theorem implementations is not applicable. No new judge calls, human ratings, agents, commits, pushes or emails. All checkout edits are uncommitted. The other checkout was not modified.
- The shared-rules refresh stopped because an unrelated local skill edit would be overwritten; I used the local index/details and preserved that edit. Shared priorities/state/log were read; no shared-state write was made because this stage's writing scope is the owned checkout and designated runtime directory.

## Reading coverage

Experiment paths below refer to `experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/`. Paper paths refer to `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/`.

| Material | Coverage |
| --- | --- |
| Briefs and earlier stage | `COMMON.md`, `BRIEF_qa2_codex_vals.md`, `qa1_carbon.md`, and `qa1_carbon.json` in full; builder diff `d9d1c1d..03ada1e`; stage 1 changes in `9e4b243`, including each finding's affected code/prose |
| Protocol and experiment prose | Full `PLAN.md` (188 lines), `protocol.json` (161), `FREEZE.json` (45), generated report (187), `README.md` (69), `HOW_TO_READ_RESULTS.md` (51 on entry), `results.md` (25), checkpoint (31 on entry), `SOURCES.md` (55), and source receipt; experiment 04 rows/context in `experiments/README.md` |
| Measurement scripts | Full `gtheory_lib.py` (292), `check_counterexamples.py` (448), `run_d_study.py` (349), `run_historical.py` (213), and `test_metrics.py` (240) |
| Verification/output scripts | Full `verify_results.py` (219), `make_figure.py` (86), `make_report.py` (322), and `freeze.py` (42) |
| Saved results | All result files compared against their reruns; all JSON parsed, all Phase B CSV rows consumed by checks. Manually inspected the reported aggregate values, prediction records, moment examples and historical intervals; did **not** manually read every repeated JSON/CSV entry. Figure inspected visually |
| Source article | Title/bibliographic material and extracted printed pages 924–928 in full; page 925 inspected visually for the coefficient equations and expected mean squares. Other portions were sampled, not a full end-to-end reading of all 11 pages |
| Paper | Full builder/current reliability appendix, main input file, bibliography, proposal section, historical-evidence section, Makefile and paper README; first 85 lines of the RPS appendix for terminology/definitions. Built PDF: visually inspected appendix pages 5–6; scanned full-document text/metadata for identity strings, without claiming a full out-of-scope paper review |
| Private data | Read only through the prescribed `run_historical.py` and `verify_results.py`; no per-item data, artifact names or rater columns were exported or printed |

## Findings

Locations identify the relevant code or current corrected text; `qa1_carbon.md` is preserved as the historical stage 1 report. Totals for **new stage 2 findings**: critical 0, major 2, minor 5, nit 1.

| ID | Severity | File:line | Evidence | Fix or open |
| --- | --- | --- | --- | --- |
| QA2-F1 | major | `results.md:19`; `scripts/make_report.py:55`; paper `9-appendix-reliability.tex:28` | “Every moment” is false: A1 changes the third raw moment by `6 * epsilon`. On six fixed categories, all moments would determine the distribution. “Every proper loss” also needs strict propriety: a constant loss is proper but cannot distinguish forecasts. | Fixed the status/checkpoint to first two conditional moments and population coefficients; appendix now says expected strictly proper probability losses. The reading guide explicitly corrects the preserved generated report (`HOW_TO_READ_RESULTS.md:56`). |
| QA2-F2 | major | `scripts/run_historical.py:168`; `HOW_TO_READ_RESULTS.md:43`; paper `9-appendix-reliability.tex:43` | `most_rel` and `best_rps` are chosen from the same historical test point estimates, then their identities are fixed for bootstrap differences. Stage 1's “zero fitting or selection” was incorrect. The interval does not account for choosing extrema from those data. | Fixed guide, status and appendix: descriptive comparison, same-test pair selection, fixed raters/calls/pair, no selection adjustment or confirmatory selection advantage. Preserved all estimates and intervals. |
| QA2-F3 | minor | paper `9-appendix-reliability.tex:18,53`; `README.md:48`; `HOW_TO_READ_RESULTS.md:57` | The synthetic study includes Monte Carlo estimates, and 35/36 in-model error checks met the tolerance, so “all synthetic numbers are exact” and “failed only” under violations were too broad. The report's simultaneous-miss probability assumes independent intervals despite shared panels; a larger diagnostic standard deviation is estimated, not known truth. | Scoped population expectations versus simulation, explicitly retained the in-model miss, softened “confirmed chance” wording in status/checkpoint, and added saved-report corrections. No failed prediction was converted to a pass. |
| QA2-F4 | minor | `HOW_TO_READ_RESULTS.md:41,42`; `scripts/gtheory_lib.py:242`; `qa1_carbon.md:47` | The RPS objective is quadratic, not linear programming. REML starts at `max(raw, 1e-3 * MS_e)`, with residual bound `1e-12`; it is not simply zero-truncated. Stage 1's likelihood term is not globally convex in variance. | Corrected objective, bounds, initialization, optimizer limits and the feasible-ANOVA global-minimum argument. The stage 1 reasoning correction appears below. |
| QA2-F5 | minor | `qa1_carbon.md:48`; `scripts/check_counterexamples.py:288`; paper `9-appendix-reliability.tex:35,36` | Stage 1 reports pooled Φ = 0.400 and Eρ² = 0.5 from components 1/2, 1/2, which are not this experiment's offset population. The actual components are `9/50`, `27/50`, `1/4`; pooled Φ = `18/97`, Eρ² = `1/4`. No claimed finite-facet variant is present in the saved output. | Independently recalculated and corrected here; guide distinguishes random rater-type sampling from a fixed two-rater panel. Appendix identifies the separate five-artifact, unclipped, separately analyzed control. Stage 1 report retained unchanged. |
| QA2-F6 | minor | `HOW_TO_READ_RESULTS.md:23,24,25,27,30` | “Φ ≤ Eρ² always” cannot be applied to raw sample estimates: saved historical H0_opus has negative estimated call variance and Φ > Eρ². Signed bias also ranges from −5 to 5, not 0 to 5. | Corrected population qualifications, raw-estimator caveat, bias units and association/absolute-agreement terminology. |
| QA2-F7 | minor | `protocol.json:127,132`; `scripts/run_historical.py:94,164`; `HOW_TO_READ_RESULTS.md:60` | The frozen protocol names a secondary three-expert generalizability analysis, but only the two-expert analysis is implemented. The secondary all-75 bootstrap uses 2,000 draws rather than the protocol's general 10,000-draw setting; the primary test analysis uses 10,000. | **Open secondary-analysis omission**, now explicitly marked not measured. Documented both discrepancies without changing the frozen protocol/results. Neither affects the primary test numbers or synthetic counterexamples. |
| QA2-F8 | nit | paper `9-appendix-reliability.tex:55` | “Correlated raters understated deployment error” assigns the error to raters rather than the prediction. | Changed to “the independence-based prediction understated deployment error for correlated raters.” |

The generated report and its presentation generator retain their historical wording under `COMMON.md`'s restoration rule; their corrections are centralized in the linked reading guide. This is deliberate preservation, not an assertion that the old wording was correct.

## Independent mathematical assessment

**Expected mean squares and error variances.** The component order is `(u,a,b,ua,ub,ab,e)`. A main-effect stratum includes its own component multiplied by the product of the other facet sizes, its containing two-way interactions multiplied by the remaining facet size, and residual variance. Interaction strata have the appropriate remaining-size multiplier; residual has multiplier one. Degrees of freedom are products of the relevant `(n-1)` factors. Orthogonal projectors independently check these quantities in `test_metrics.py`.

Averaging over `n_a` raters and `n_b` occasions gives relative error variance `v_ua/n_a + v_ub/n_b + v_e/(n_a*n_b)`; absolute error additionally contains `v_a/n_a + v_b/n_b + v_ab/(n_a*n_b)`. These match source page 925's relative/absolute distinction and direct covariance calculations. For configuration C1, absolute error at `8×2`, `4×4`, `2×8` is respectively `23/80`, `31/80`, `53/80`; configuration C2 swaps the first and last. All are population squared-error expectations, lower better; `4×4` is the equal-allocation reference. There is no uncertainty interval for these analytic values.

**REML.** For stratum mean square `m` and variance eigenvalue `lambda`, the negative log-likelihood contribution, apart from its positive degree-of-freedom factor, is `log(lambda) + m/lambda`. Its derivative is `(lambda-m)/lambda²`, so its global minimum is at `lambda=m`. Its second derivative is `(2*m-lambda)/lambda³`, which becomes negative: stage 1's convexity claim is false. When the ANOVA solution is feasible, it attains every stratum's minimum simultaneously, proving global optimality without convexity. All 512 base fits have strictly positive raw components and match REML exactly. For the heterogeneity diagnostic, replication 17 has a negative raw occasion component and a constrained zero estimate; no generic global-optimum claim is made for this boundary fit.

**Proper losses and moment construction.** With cumulative distributions `P,Q`, expected excess normalized ranked probability score (RPS) is `sum_k (Q_k-P_k)²/5`, on a 0–1 loss scale, lower better. Expected excess cross-entropy is Kullback–Leibler divergence in nats; Brier excess is `sum_r (q_r-p_r)²`; total variation distance (TVD) is `sum_r abs(q_r-p_r)/2`, range 0–1. The reference is the true conditional distribution, with zero excess but generally positive expected loss.

For A1, the perturbation `(-1,3,-3,1,0,0)` has zero zeroth, first and second moments; its third moment is 6. Its cumulative perturbation is `(-1,2,-1,0,0)`, giving RPS excess `6*epsilon²/5`, Brier excess `20*epsilon²` and TVD `4*epsilon`. For the central artifact, the saved exact probabilities and `epsilon=43/5760` give RPS excess `1849/27648000`. The A1_max feasible set has three independent equalities; every vertex after subtracting the probability floor has at most three nonzero coordinates. Enumerating all 20 triples is sufficient, including degenerate vertices. Convexity of RPS ensures that a maximum occurs at a vertex; enumeration is exact and globally exhaustive, not gradient optimization.

**Population versus sample panels.** Conditional independence makes the covariance of two ratings of the same artifact equal to the across-artifact variance of the conditional mean. Cross-artifact residual covariance is zero; the residual component is average conditional variance. Equal conditional means and variances therefore give equal population components for these panels. Higher moments still affect finite-sample estimator variability. For the offset control, a shared random rater type with probability one half contributes `1/4` to rater variance. With the actual `v_u=9/50` and `v_e=27/50`, pooled single-rating Φ is `(9/50)/(9/50+27/50+1/4)=18/97`, and relative Eρ² remains `1/4`. This corrects stage 1's unrelated numerical example.

**Ties, random streams and retrospective data.** The temperature-zero limit splits mass over equal modes; five of the eleven binomial distributions have ties. The retained prediction that its coefficient would be exactly one is false, while the implementation's limit is correct. Phase B separates conditions/stages using `SeedSequence([rep, condition, stage])`; each evaluation design draws fresh objects and facets. Pairing by replication is a valid difference of independent design evaluations, not a claim that they share evaluation artifacts. Phase C averages losses within each artifact and then equally over artifacts, resamples whole tasks, uses the declared probability-floor transform, retains the flagged-artifact sensitivity and labels prior exposure. The historical statistic scores a three-call histogram, not an exactly observed latent judge distribution.

## Reconciliation of every stage 1 finding

| Stage 1 ID | Assessment after independent review |
| --- | --- |
| QA1-F1 | **Agree and extend.** Only occasion-instability shift reverses the selected optimum. Fixed-design prediction and subgroup limitations are distinct. Stage 1's residual “failed only” wording still concealed the in-model tolerance miss; corrected in this stage. |
| QA1-F2 | **Agree.** The proposal sentence split is correct; all in-scope LaTeX files pass the sentence-layout check. |
| QA1-F3 | **Agree.** Expanded RPS labels, task-bootstrap draw count and caption selection context are present. Figure regeneration is byte-identical to input HEAD and visual labels/gridlines are legible. |
| QA1-F4 | **Agree with the clarity goal, correct details.** Squared error and subtraction direction are correct. The new guide had inaccurate linear-programming, optimizer-start and “zero selection” descriptions; corrected under QA2-F2/F4. The base-study ANOVA certificate remains valid. |
| QA1-F5 | **Agree and verified.** The amendment comparator now uses the latest receipt. An isolated three-step test changed file A, then file B, then nothing; the latter two amendments correctly listed only B and no files. Historical receipts remain untouched. |
| QA1-F6 | **Agree.** Appendix integration and commit references match the input history. Optional main-body linkage remains optional; I updated only stage 2's checkpoint entry and scientific wording. |
| QA1-F7 | **Historical receipt, not a current-status defect.** `source_manifest.json` describes the retrieval-time state. Preserve it; the current README/status already describes execution. |
| QA1-F8 | **Carry as an out-of-scope nit.** The paper README still describes an older PDF and the RPS appendix has pre-existing sentence-layout issues. Neither file was edited. A main-body pointer remains an author choice, not an acceptance requirement. |

Also corrected two errors in stage 1's mathematical assessment (the pooled-offset numbers and variance-scale convexity), as derived above. Stage 1's blanket claim of no identifying strings anywhere in the PDF should be narrowed: the new reliability appendix pages contain none, but existing bibliographic text elsewhere contains the companion-project name and a cited author's surname. This stage checks the requested appendix's anonymity; it does not certify complete submission readiness of the out-of-scope main paper.

## Files changed

1. `HOW_TO_READ_RESULTS.md`: method definitions, bounds, selection/uncertainty qualifications, and corrections to the preserved report.
2. `README.md`: accurate in-model result and historical-evidence scope; link to corrections.
3. `results.md`: first-two-moment wording, descriptive pair interval, and retained tolerance miss.
4. `CKPT_g_theory_plan.md`: scoped claims and stage 2 log entry.
5. Paper `9-appendix-reliability.tex`: corrected claims, metric scales, offset-control identity, pair-selection disclosure and retained miss.
6. Paper `00_main_aaai_2027.pdf`: rebuilt from the corrected source.
7. This report, `qa/mega_qa_11-14-2026/qa2_codex_vals.md`.

All frozen hashes still match Amendment 2. No amendment was required. The generated results/report were restored to input HEAD after confirming equality apart from runtime fields and their dependent hashes. No change was needed to the experiment-index rows, source citation entry, main input line, or stage 1's proposal fix.

## Deterministic verification and observed output

Experiment commands ran from the experiment folder. Full command outputs are in runtime `qa2_codex_vals.verification.json`.

| Command | Observed result |
| --- | --- |
| `python3 scripts/test_metrics.py` | Exit 0; `n_checks=945`, `n_fail=0`; output byte-identical |
| `python3 scripts/check_counterexamples.py` | Exit 1 as expected; 16 predictions, failures `A1.finite_sample`, `A3.sampled_reliability`; runtime 2.47 seconds; all other fields identical |
| `python3 scripts/run_d_study.py` | Exit 1 as expected; 6 predictions, failure `B.formula`; runtime 2.06 seconds; all other JSON fields and all CSV bytes identical |
| `python3 scripts/run_historical.py` | Exit 0; 1 prediction, no failures; runtime 41.14 seconds; all other fields identical |
| `python3 scripts/verify_results.py` | Exit 0; `n_checks=134`, `n_fail=0`; verification JSON byte-identical, including labeled post-hoc diagnostics |
| `python3 scripts/make_figure.py` | Exit 0; PNG byte-identical to stage 1; inspected visually |
| `python3 scripts/make_report.py` | Exit 0; only runtime text and hashes dependent on those runtime fields changed; restored report/results as instructed |
| SHA-256 integrity check against latest `FREEZE.json` amendment | 6/6 files match; all scientific measurement sources preserved |
| `python3 /Users/brandomiranda/.local/state/judges/mega-qa-04-1114/qa2_codex_vals.independent_checks.py` | Final exit 0; 951/951 supplemental assertions. Recomputed Phase B formula intervals and principal selection-summary means/intervals from CSV, checked base-fit certificates, exact pooled offset and stage 1's freeze fix; all V2/V4 fits converged on their original seeds |
| `make` in the paper folder | Exit 0; six pages, 297,877 bytes; no `!` errors, undefined references/citations, overfull boxes or BibTeX warnings |
| `python3 ~/agents-config/scripts/latex_sentence_lines.py --check <paper>/00_main_aaai_2027.tex <paper>/4-proposal.tex <paper>/9-appendix-reliability.tex` | Exit 0; no output |
| `git diff --check` | Exit 0; no whitespace errors |

The supplemental audit's **initial** version had 949/951 assertions pass: it incorrectly hypothesized that every V2 raw ANOVA solution would be feasible and equal the constrained fit. Replication 17 disproved both hypotheses. Both initial script and output are retained as `qa2_codex_vals.independent_checks.initial.{py,json}`. The final audit checks the actual constrained bounds and ANOVA equality only when ANOVA is feasible; it does not convert that boundary case into a global-optimality claim or change any experiment result.

The figure, appendix pages, new prose and findings were manually proofread, including spelling and mathematical terminology. No standalone spelling checker was available. Runtime evidence also includes `qa2_codex_vals.output_comparison.json`, `qa2_codex_vals.paper_build.log`, `qa2_codex_vals.paper_checks.json`, the independent-check scripts/results and rendered appendix page images.

## Open issues and verdict

- **Open critical: 0. Open major: 0.**
- **Open minor: QA2-F7.** The secondary three-expert analysis remains unimplemented. The primary analysis and numerical claims are verified; the omission and secondary bootstrap setting are now explicit. Completing that analysis would be a separately recorded amendment, not a silent rewrite of frozen results.
- **Inherited out-of-scope nit: QA1-F8.** Paper README and RPS-appendix source layout are outside this stage's editable scope. No full-paper submission/anonymization approval is implied.
- The saved report/generator's old wording is preserved per the brief and corrected by `HOW_TO_READ_RESULTS.md`. The source receipt and stage 1 report remain historical evidence.
- No required stage 2 review or verification step was left unfinished. Marker writing is the last stage action. The next authorized reviewer remains the fresh Antigravity CLI / `gemini-4-carbon-high` / high stage specified by `COMMON.md`; this reviewer did not dispatch it.

**Verdict: accept with minor open issues.** The supported result remains limited to constructed insufficiency examples, a successful base-model design-allocation demonstration, and a descriptive historical comparison. None establishes deployment prevalence, universal superiority, or that psychometrics is unnecessary.

**TLDR-end:** [judge objectives: experiment 04 Mega QA stage 2] Completed independent review and stage 1 reconciliation, fixed claims and interpretation, reproduced measurements and rebuilt the appendix. Accept with the explicitly documented secondary-analysis omission and inherited out-of-scope nit; no critical or major finding remains open.
