# Experiments 09 and 10: full-base correctness and structure review

Review date: 11-16-2026. Base: `233d5047d20da8591b4609ab52242ceed8717152`.

Scope: the complete changes from the base to the working tree in both experiment folders and `experiments/README.md`, including committed changes and new files. Root `AGENTS.md` was read. The pre-existing empty `qa/mega_qa_11-16-2026/qa1_codex_vals.md` was left unchanged.

This was a Codex model review plus local deterministic verification, with native subagents dividing the analysis, execution, and reporting files. No paid model calls, remote runs, commits, pushes, or external messages were performed. No raw individual human ratings were added to public artifacts.

## Findings and minimal fixes

Counts group related manifestations by the underlying correctness requirement: 0 critical issues, 13 major issue groups, 13 fixes.

| # | Major issue | Fix |
|---|---|---|
| 1 | Experiment 09 maximum-entropy objective omitted the removed logit maximum, so the fitted reference did not preserve the target mean. | Use stable log-sum-exp, its analytic derivative, and a mean-matching assertion; regenerate local analysis and reports. |
| 2 | Experiment 10 resampled overlapping split summaries but interpreted their narrow intervals as population uncertainty and a rating-count threshold. | Preserve point estimates and interval endpoints, label conditional split Monte Carlo variability, remove population p-values and established-threshold claims, and retain descriptive tail fractions separately. |
| 3 | Reports inferred objective-only causation despite unsuccessful non-convex optimization, and isolated representation effects from unmatched model/prompt/ensemble comparisons. | Bound conclusions to the recorded optimizer and budget; disclose changed pool composition and the five-forecast versus three-vote ensemble comparison. |
| 4 | Reported split counts, validation sizes, tuning grids, model/evaluator identities, probability completeness, and validation scope disagreed with executable evidence. | Correct current guides/builders; disclose truncated or imputed digit probabilities and the limited recorded gradient/epoch-zero checks. Preserve original pre-run source and plans where no execution fix is needed. |
| 5 | Missing Experiment 09 sanity controls were rendered as a fabricated score of −1 on a 0–5 scale. | Render missing controls as `n/a`. |
| 6 | Experiment 10 marked observations outside the preregistered sanity prediction ranges as matches. | Correct the displayed match verdicts without changing observations or the original prediction ranges. |
| 7 | Tables/figures omitted required metric expansions, selection details, or uncertainty descriptions. | Add metric definitions, accurate split/optimizer captions, and expanded axis labels; regenerate figures. |
| 8 | Input-parity descriptions obscured 60/75 reference hashes that differ from annotation metadata. | Disclose the mismatch in both current reports/guides. Saved evidence cannot distinguish stale metadata from changed reference bytes; identity with human-delivered references remains unverified. |
| 9 | Experiment 09 called all fixed prompts zero-exposure despite P2 including historical expert-mean and leniency summaries. | Identify P2-containing pools as retrospectively informed; distinguish family-separated fitting/selection from earlier aggregate exposure. |
| 10 | Experiment 10 silently substituted uniform forecasts after parsing failures and omitted failure provenance. | Recover status metadata from local cached responses, retain saved forecasts and numerical results, preserve failure flags in future exporters, and disclose the affected denominators. |
| 11 | Experiment 10 spending safeguards reused ledger IDs, reserved an average rather than conservative input-token bound, and the report builder checked the wrong ledger filename. | Give each invocation unique reservation IDs, reserve against UTF-8 byte count plus framing, and read the canonical ledger path. The existing ledger has no overwritten positive charges. |
| 12 | Experiment 10 verification did not enforce frozen-file digests; existing post-freeze changes lacked effective hashes. | Preserve original freeze dictionaries, separately record pre-review discrepancies and post-run amendments, and verify effective hashes in both experiments. |
| 13 | Verification copied certificate totals without asserting acceptance and lacked finite/nonnegative/shape checks; constant forecasts could divide by zero in the independent mean squared error identity. | Enforce nonempty accepted certificate totals, validate probability arrays, and use the covariance form of the identity at zero variance. |

## Validation and evidence

- Reviewed all 17 new Python scripts; large orchestration functions did not justify a cosmetic refactor. Structural changes repair verification contracts, failure propagation, and the ledger-source branch.
- Recomputed Experiment 09 locally from saved forecasts. Only three analysis-summary leaves changed: the information-gap correlations. Primary loss results and saved model forecasts were unchanged.
- Checked maximum-entropy mean matching over all 77 possible two-/three-rating multisets on the six-category scale, plus synthetic mean targets. No individual human ratings were printed.
- Checked correlation gradients, synthetic convex/cone certificates, saved subtraction order, and exactly-once out-of-fold coverage. The minor clipped-loss gradient concern did not change predictions in the sampled actual fit and was not included as a major issue.
- Mocked repeated identical requests entirely offline: distinct cost IDs were preserved; conservative reservation and zero-variance/negative-probability checks passed.
- Public-file scans include all tracked and visible new files, rather than only changes against `HEAD`; no private paths, secret patterns, or forbidden human-rating fields were found.
- Both `scripts/verify_results.py` commands passed: Experiment 09 reports 394 deterministic checks; Experiment 10 verifies 14 effective frozen files, 28 forecast/status matrices, and all 75/75 saved certificates for each of the three certified objectives. Its independent metric discrepancy is below its prescribed tolerance. Latest receipts are in each experiment's `results/verification.json`.
- Synthetic digest corruption and an unsuccessful certificate were both rejected by the strengthened verifier. Original freeze dictionaries and all saved Phase 3/4 forecast arrays compare equal to their pre-review values.
- `git diff --check` passed both for the current edits and for the full authorized diff from the requested base. All changed/new files are within the three authorized paths.

Original model outputs are preserved. Amended sources describe post-run fixes and do not claim that the original runs executed those sources. Reference-byte identity, population inference from split variability, and unmeasured global optimizer guarantees remain explicit limitations of the historical evidence.
