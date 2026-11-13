# Fix Pearson fits and rerun experiment 06 with matched budgets

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/ckpt.md>

**TLDR:** The matched cross-entropy versus Pearson comparison is complete; no scientific rerun remains under that request. This repository now also holds the full historical experiment package and private reconstruction inputs; see CKPT_migration_11-09-2026.md for the completed relocation.

Created: 11-09-2026 10:25 PST
Last updated: 11-09-2026 14:34 PST

## Goal
- Fairly compare cross-entropy (CE) and Pearson objectives on held-out human-rating metrics while preserving historical evidence.
## What ran
- Prospective TVD evaluator published in trustworthy-theorem-judges at 27237a2; matching panel published here at 7e87102. Paper-coordination packet below; no new experiment.
- Historical failure reproduced with the original configured condition; longer-budget and no-penalty diagnostic ablations were retained. Five synthetic test groups passed.
- Frozen matched-simplex comparison completed locally under Brando’s standing fallback authorization; 50 primary fits and 49,204 bootstrap fits attempted. No new model calls.
## Results
- All 50 primary fits passed. CE minus Pearson held-out log loss: -0.260401 [-1.457729, 0.198472], 95% family-bootstrap interval with refitting, 998/1,000 valid draws, p-val=n/a. CE superiority is not established.
- Two CE bootstrap certificates failed and remain in the ledger; no replacements. Independent reconstruction of 75 models / 1,125 primary forecasts and the full metric panel passed.
- Both mixture methods had worse observed log loss than the training-human-marginal baseline. See RESULTS_comparison.md and its Google companion for full statistics.
## Stuck / blocked
- No execution blocker; the requested comparison is complete. The old nonlinear softmax optimization problem remains unresolved; the new condition avoids that confound rather than relabeling old fits.
- Native chat indexes remain separate. Shared project/checkpoint reads are verified across all three profiles; native cross-account sidebar/resume behavior is not certified.
## Next steps
- [p 8/10] VeriBench paper owner reads the coordination packet below and integrates only condition-matched public evidence; practical judge development remains with the sole human-rating owner.
## Brando to-do
- None for the completed comparison. No background experiment, reconnect loop or temporary checkout remains.

## Log
11-09-2026 12:16 PST -- Completed this chat's historical-package relocation: historical_exp06/ holds the full original public study, distance follow-up and weighted-PMF training; private/historical_exp06/ is the single canonical private store. All 83 public and 1,521 private files verified, original freezes preserved, no scientific rerun. See CKPT_migration_11-09-2026.md and MIGRATION_11-09-2026.md; the earlier instruction to switch the working directory is obsolete.
11-09-2026 -- Completed the frozen local comparison in 63.08 seconds; all primary fits passed, two CE bootstrap certificates failed and were retained. Independent verification passed. Published findings companion and verified public viewer-link access. Rule 55 preference update pushed as agents-config 9aee76b.
11-09-2026 -- Brando authorized remote-first/local-CPU fallback without future confirmation. Original-settings diagnostic failed at 300 iterations, the 3,000-iteration ablation also failed, and the no-penalty ablation failed at 300. Reviewed gradient/variance controls and froze the matched-mixture condition before new held-out measurement.
11-09-2026 -- Receiving chat verified the supplied checkpoint SHA-256, read original/follow-up source, prepared training-only diagnosis and candidate CE/Pearson mixture implementation, passed five synthetic controls, and verified three shared project pointers. SNAP guard has an active cooldown; local fallback question is pending. No empirical rerun launched.
11-09-2026 10:25 PST -- Created the handoff for the user-selected “Fix Pearson Fits and Rerun Exp 06” chat; the originating chat will stop experiment execution when the handoff is delivered.

11-09-2026 14:34 PST -- Recorded the user-authorized public-paper coordination packet: pinned evaluator definitions/receipts, separate historical evidence, practical sole owner, private-results boundary and immutable without-DT baseline; no new fitting or inference.

## Public-paper coordination packet — 11-09-2026

**Request and ownership.** Verified Brando's direct instruction in the “Check DT status in VeriBench” chat (01a1a7da-397d-7c53-b93a-3f5d3bf1886e): coordinate the public differential-testing (DT) and judge evidence for the Transactions on Machine Learning Research (TMLR) and AAAI drafts, keep private benchmark results in the private report/blog, and preserve his photographed without-DT baseline unchanged. That chat owns VeriBench manuscript integration and checks. This chat retains the evaluator assignment and completed objective-comparison evidence. “Prepare human grading task — sole owner” (01a1a8f9-dcb7-7381-9368-91d6e51faffd) owns collection and practical judge development; see its [coordination record](https://github.com/brando90/trustworthy-theorem-judges/blob/b3ce98e0034972ee5d7423a1ff54f4609ab3ea11/experiments/09_human_rating_collection/COORDINATION_11-09-2026.md) and [handoff](https://github.com/brando90/trustworthy-theorem-judges/blob/b3ce98e0034972ee5d7423a1ff54f4609ab3ea11/experiments/02_probabilistic_ordinal_judge/CKPT_HANDOFF_11-09-2026.md). The TVD worktree mentioned historically in that handoff has been removed after verified publication and local synchronization.

**Canonical evaluator and publication.** The [applied evaluator commit](https://github.com/brando90/trustworthy-theorem-judges/commit/27237a27d49b83cfdb910153b59af32e15198325) is `27237a27d49b83cfdb910153b59af32e15198325`; its relevant files remain byte-identical on inspected main `b3ce98e0034972ee5d7423a1ff54f4609ab3ea11`. Manifest version: `native_probability_offline_v1`; metrics version: `native_probability_metrics_v2`. This repository's [prospective-panel commit](https://github.com/brando90/judges-dont-need-psychometrics-or-corr/commit/7e87102867c07795c7cd466da14e14cca9aead7f) is `7e87102867c07795c7cd466da14e14cca9aead7f`. Both were pushed to main and synchronized locally; temporary publication worktrees were removed.

For an artifact's forecast q over ratings 0–5, let Q(k) be its cumulative probability and p_hat its empirical human rating frequency. Normalized ranked probability score (RPS) is primary and the common validation criterion: average `(1/5) sum_{k=0}^4 (Q(k)-1{y<=k})^2` over observed human ratings. Cross-entropy (CE), the average `-ln(q(y))`, and total variation distance (TVD), `0.5 sum_r |q(r)-p_hat(r)|`, are secondary. Average each artifact's metric equally across the same eligible artifacts; preserve the full scheduled denominator and failed/missing forecasts. The evaluator's CE has no probability floor and permits infinity for zero observed-rating probability. TVD uses the empirical histogram once, not distances to individual one-hot ratings; its full-denominator bounds use 0–1 and are not confidence intervals. TVD mutual information is a different target and is not part of this extension. The complete definitions and historical distinctions remain in [EVALUATION_CRITERION_11-09-2026.md](EVALUATION_CRITERION_11-09-2026.md).

| Applied-study artifact under experiments/02_probabilistic_ordinal_judge/ | SHA-256 at the evaluator publication |
| --- | --- |
| native_probability/scaffold.py | 5c5b5520040dd4c6320eb01f24c9eb7af52bb4862f50da8f6f24f83ca1825a82 |
| config.json | 57501935e8f169bb7f442288ef238416208af6c63a1eb97b2dd5ca01318e9bd8 |
| PRIORITY_2_VS_4_11-09-2026.md | c39a649ced5c8d25e203e444bbf6d7cd7806aadbb9568570359969db02b2259d |

**Which evidence supports which claim.**

- **Evaluator extension:** 17/17 deterministic synthetic evaluator tests passed at publication. This supports the documented computation, parsing, hashing, weighting and missingness behavior. It supplies no new model forecasts, fitted judge, human-agreement measurement, deployment result or public DT outcome.
- **Historical VeriBench judge component:** use the exact condition-matched original receipt, supported by the [historical aggregate archive and limitations](https://github.com/brando90/trustworthy-theorem-judges/blob/b3ce98e0034972ee5d7423a1ff54f4609ab3ea11/experiments/01_legacy_veribench_judge/README.md) and its [source manifest](https://github.com/brando90/trustworthy-theorem-judges/blob/b3ce98e0034972ee5d7423a1ff54f4609ab3ea11/experiments/01_legacy_veribench_judge/SOURCE_MANIFEST.json). The original score-mapped judge, selected two-model judge and cross-validated recipe are distinct conditions; none becomes a new RPS-primary/TVD-evaluated judge by relabeling. Preserve split/exposure, reference-version, solver-population, absolute-bias and repeatability limitations. These are human-rating agreement records, not formal-prover acceptance evidence.
- **Separate objective-comparison study:** this repository's [RESULTS_comparison.md](RESULTS_comparison.md) compares CE versus Pearson fitting of mixtures over five cached scalar-judge histograms. It uses 75 artifacts, 25 tasks, 19 task families and 195 expert ratings, with prior test exposure. Its frozen primary endpoint remains held-out CE. It is eligible as explicitly labeled retrospective objective-comparison evidence, not a validation of a deployed VeriBench judge, fresh probability-output judge, DT intervention or frontier-agent transfer.

The objective-comparison report's paired CE-minus-Pearson contrasts are CE `-0.260401 [-1.457729, 0.198472]`, RPS `-0.004555 [-0.035602, 0.009965]`, and TVD `-0.005688 [-0.067752, 0.027772]`. These are 95% task-family percentile intervals with both methods refitted; 998/1,000 draws passed certificates, two failed draws were retained and excluded from intervals; p-val=n/a. Every interval includes zero. The full report gives the other diagnostics and limitations. RPS and TVD remain secondary for that completed comparison; the prospective primary criterion does not rewrite it.

**Publication handoff.** Both public drafts must identify the same exact judge condition and evidence revision whenever they make the same claim. Any replacement practical-judge result requires the sole owner's packet: actual model/version, harness/effort, prompts/rubric and input hashes, evaluator version, rating scale, probability policy, train/validation/test membership and exposure, observed/missing denominators, paired uncertainty, frozen selected parameters/forecasts and aggregate result hashes. No replacement human-validation result is supplied by this evaluator assignment; retain the historical condition and label a proposed replacement pending until that packet exists. No extra experiment is needed merely to coordinate.

**Boundaries.** The DT paper owner maintains the public DT receipts separately; this packet does not revise or merge them with the cached judge comparison. Private benchmark results remain only in the private report/blog and are not copied into this checkpoint or either public draft. Brando's photographed without-DT baseline is immutable: no re-estimation, rescaling, denominator substitution or replacement by another historical row. Preserve every completed study's frozen protocols, metrics and result hashes. No fitting, inference or paper edit was launched by this coordination update.

## User authorization and ownership

### Receiving-chat completion and reproduction

The receiving chat completed execution and is the sole notifier. The frozen sources and original results are retained. Read RESULTS_comparison.md and results/verification.json; private forecasts and bootstrap draws are in private/run_v1. Run `python build_report.py` to rebuild the report from committed aggregate evidence. The complete public numerical bootstrap-fit ledger is losslessly compressed; decompress it before running the frozen verifier. A fresh scientific rerun must use new output paths and the same private dataset hash, never overwrite completed evidence.

Google companion: https://docs.google.com/document/d/1f6AtTPIoHx-B3jaOy4ot8TNH__t_Kbiut5KwO0kKuX0/edit . Native readback verified its main results, and browser sharing showed “Anyone on the internet with the link can view” with “Access updated”. Shared profile pointers remain verified; authentication and live session databases were not merged.

The user explicitly requested: “Investigate why the Pearson fits failed to converge, fix it, and rerun experiment 06 with matched budgets. I want a fair comparison of CE vs correlation objectives.” They then specified that this work belongs in `https://github.com/brando90/judges-dont-need-psychometrics-or-corr`, requested its Codex Desktop project and cross-account session access, and asked the originating chat to save `ckpt.md` and transfer all remaining work to **Fix Pearson Fits and Rerun Exp 06**.

The receiving chat becomes the sole execution owner after delivery. The originating chat has launched no new fit, background job or subagent for this repair. Keep current recipient model settings unless the user changes them; verify effective full access and useful execution. You are acting as Brando's assistant, also called ub9sc.

Read this repository's `AGENTS.md` and the current shared instruction index before work. Research changes go directly to `main`; preserve concurrent work. The immediate comparison uses already cached ratings and requires no new paid inference. Existing funding pauses and private/sealed-data boundaries remain in force. Existing authorized compute can be used after its required checks.

## Project placement and historical lineage

- Migration completed 11-09-2026: the full historical package is now in `historical_exp06/`, with private inputs in `private/historical_exp06/`. [Migration checkpoint](CKPT_migration_11-09-2026.md) records current ownership, verification and compatibility paths. The source locations below describe historical provenance.
- New canonical working repository: `/Users/brandomiranda/judges-dont-need-psychometrics-or-corr`.
- New canonical experiment: `experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/` in that repository. “Experiment 06” remains the historical sister-repository identifier; local number 03 avoids collisions with existing experiments 01 and 02.
- Historical repository: `/Users/brandomiranda/trustworthy-theorem-judges`.
- Historical experiment root: `experiments/06_information_loss_vs_correlation_judge_selection/` inside that repository.
- Preserve historical code, results, and their limitations. Copy only required code with source revisions/hashes or reference the historical files; do not overwrite the old condition as if it had converged.
- This repository already has a completed synthetic conditional-distribution study at `experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/`. Read its report and current paper claims before adding conclusions. It is not the cached human-rating experiment.
- The main paper is `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/`; if results justify changing its text, rebuild the PDF using the current Makefile and preserve separate historical conditions.

## Read these source files first

All paths in this list are relative to the historical experiment root above:

1. `RESULTS.md`, `HEAD_PROTOCOL.md`, `head_protocol.json`, `protocol.json`, `source_audit.json`.
2. `scripts/fit_heads.py`, `scripts/metrics.py`, `scripts/run_cached.py`, `scripts/verify_heads.py`.
3. `results/head_results.json`, `results/head_verification.json`, `results/head_launch.json`, `results/head_terminal.json`. These JSON files are large: extract relevant fields rather than printing them in full.
4. `training_followup_11-08-2026/RESULTS_training.md`, `MATH_baseline.md`, `PROTOCOL.md`, `protocol.json`, `scripts/mixture_models.py`, `scripts/check_mixtures.py`, `scripts/run_training.py`.
5. `followup_11-07-2026/RESULTS.md` for the intervening distance-metric comparison.

The November 8 work is especially important: it already supplies a stable, matched **weighted probability mass function (PMF) loss versus Pearson mixture** comparison and numerical optimality checks. It does **not** repair the old Pearson softmax fitting, and its softmax comparison excludes Pearson. Reuse its relevant mathematics and controls; do not repeat it under a new name or mistake it for the requested CE-versus-Pearson comparison.

## Private inputs, verified on 11-09-2026

Private root: `/Users/brandomiranda/trustworthy-theorem-judges/private/06_information_loss_vs_correlation_judge_selection/`.

| File under private root | SHA-256 |
| --- | --- |
| `cached_v1/dataset.json` | `b3a05699a26d3ffd9f28e793178811219885df78319b3aa57f9d393c68724ba0` |
| `head_forecasts.json` | `b2de810109e5c2f22108cc09354436194353aaadc23354a19a2fb5133431b87d` |
| `bootstrap.json` | `d06ad11146c0f5bd8bcf9faa80a34375326769b03308432041b868c5f5436e21` |
| `inputs/original_note_11-06-2026.png` | `c76657091ffb2e6515ad6bb4565dacc66e7fd8368ad2681a058d399c6bc21443` |
| `inputs/USER_REQUEST.txt` | `9354519c0c3b84926d8ed3ac49720798605f31914db9a4832dc433e2650a7752` |

These files exist locally. The original image and user prompt are already preserved. Keep raw human ratings, private forecasts, transcripts and operational account information out of this public repository. If private copies are needed here, establish and verify ignore rules first.

The dataset contains 75 artifacts, 25 Python tasks, 19 task families and 195 expert ratings. Each artifact has two or three integer ratings on 0–5. There are five cached judge candidates, each with three scalar ratings per artifact. These are not native probability predictions. The original 10/5/10-task train/validation/test split and later family folds have prior exposure; any rerun on this cache remains retrospective.

The saved candidate order and exact model/rubric provenance are in the source configuration. Saved selectors include `openai/gpt-6-sol-high` with high effort and `anthropic/claude-opus-5-5-no-fallback` with max compute effort. Do not infer a new model identity from these historical aliases.

## Old softmax condition and the failure to investigate

The old condition used the mean normalized rating of each of the five candidates as features, training-only standardization, an intercept and a six-category affine softmax: 36 stored coefficients. Every fitting objective used matching initialization seeds, L2 strengths 0.001 and 0.01, and four inner task-family folds selecting regularization by normalized ranked probability score (RPS). Five partitions, five outer folds and four objectives yielded 900 fits: 225 per objective, including 25 final outer fits per objective.

Optimizer: SciPy L-BFGS-B, analytic gradient, `maxiter=300`, `maxfun=600`, `ftol=1e-10`, `gtol=1e-6`. L2 penalized every coefficient including the intercept. Fits used unsmoothed softmax probabilities; scoring used `epsilon=1e-6`. Inspect code for the exact smoothing convention. Gradient controls passed at the sampled control points; this does not prove numerical stability near collapsed predictions.

For Pearson, with expected rating mu, centered predictions u and centered human means v, the implementation minimized

    1 - (u @ v) / sqrt((u @ u + 1e-12) * (v @ v + 1e-12))
      + lambda * sum(W**2) / 2.

Measured original diagnostics:

- Pearson: 215/225 nonconverged fits, including 24/25 final outer fits. All 675 CE/Brier/RPS fits converged.
- Mean training Pearson across final Pearson heads: 0.942168.
- Mean within-test-fold Pearson across those heads: 0.548347.
- Mean within-test-fold standard deviation of expected ratings: 0.010283 on the normalized 0–1 scale.
- Pooled out-of-fold Pearson, averaged across partitions: 0.056962. This is a different statistic from the average within-fold correlation.
- CE-fitted test CE: 1.527141; Pearson-fitted test CE: 1.897422.
- CE minus Pearson test CE: -0.370 [-0.690, -0.080], 95% paired task-family bootstrap interval conditional on fitted predictions; p-val=n/a. It excludes refitting and regularization-selection uncertainty.

These measurements motivate, but do not by themselves establish, the mechanism. Investigate correlation's location/scale invariance, the L2 shrinkage pressure, the denominator stabilizer near constant predictions, coefficient redundancy, stopping diagnostics and pooled-fold scale/offset differences. Test candidate explanations using synthetic controls and training-side traces. Do not declare an analytic-gradient bug without evidence or treat an iteration-limit increase alone as a fair scientific repair.

## What the November 8 follow-up already established

Both primary methods fit the same global nonnegative mixture of five cached probability vectors, with weights summing to one. Pearson is optimized through nonnegative cone projection with explicit handling of nonpositive correlations, constants and ties; the weighted objective is a convex quadratic. The report records all 50 primary fits passing optimality checks. Its 900 secondary softmax fits all converged, but their objectives were weighted PMF, CE, Brier and RPS, **not Pearson**.

The primary weighted-minus-Pearson weighted-PMF difference was -0.010409 [-0.061911, +0.005933], with a 95% family-bootstrap interval including mixture refitting, 998/1,000 valid draws, p-val=n/a. This does not resolve an advantage. Two numerical certificate failures were retained; do not silently replace them. See the complete report for denominators, model class, tie conventions and all metrics.

This provides a possible well-defined common model class for a new CE-versus-Pearson comparison. It must remain distinct from repairing the historical softmax implementation. Comparing CE softmax with Pearson mixture would confound objective and model class.

## Required continuation and fairness criteria

1. Reproduce at least one saved failing Pearson fit with its exact training subset, initialization and settings. Record termination reason, gradient norm, objective components and prediction variance. Confirm the old convergence counts from saved receipts.
2. Diagnose the mechanism with numerical and analytic controls, including near-constant predictions and gradient checks there. Separate optimizer convergence, statistical identification and fold aggregation.
3. Choose the corrected design using those controls. Record the intended comparison, model family, constraints, tie rule, smoothing, objective definitions, validation criterion, seeds, tolerances and budget **before** measuring the rerun. If changing model family, label it as a new condition. Prior test exposure remains disclosed.
4. Match inputs, features, splits, capacity and search opportunities. Give all objectives the same prespecified resource ceilings and account for iterations, function/gradient evaluations, starts, tuning candidates, time and failures. If using exact or convex solvers, report objective-specific optimality checks as well as resource ceilings; a common iteration count is not proof of comparable optimization quality.
5. Compare CE and Pearson within the same probability family. Retain Brier/RPS and rank-correlation diagnostics where useful, but do not describe a Pearson-only fitting result as testing all possible correlation objectives. Use the same held-out metric panel: CE/forward Kullback–Leibler (KL) divergence, Brier, RPS, weighted PMF loss, total variation, Pearson, Spearman, Kendall, normalized mean squared/absolute error and signed bias with both means. Distinguish population claims from empirical human histograms.
6. Report pooled out-of-fold correlation and within-fold diagnostics separately. Predefine any train-only output mapping; it is an additional procedure, not an invisible correction using test ratings.
7. Preserve every attempted fit and failure. Produce paired family-resampling intervals; include refitting/selection where feasible or label fixed-prediction intervals accurately. Never count repeated partitions as independent new artifacts. Keep historical exposure, tiny rater panel and lack of new-agent/rater validation visible.
8. Independently reconstruct saved predictions, chosen settings and aggregate metrics, run focused deterministic tests, save logs and hashes, update results/checkpoint/index and push verified changes to `origin/main`. Report the actual result, including a tie or failure; no objective is required to win.

The original fixed-candidate comparison is also relevant context: CE-selection minus Spearman-selection test CE was +0.770 [-0.895, 1.317], using 1,000 family resamples with candidate reselection inside fixed partitions, p-val=n/a. It did not establish CE superiority.

## Compute and execution state

The retained old SNAP (Stanford Network Analysis Project) job directory is `/lfs/skampere3/0/brando9/ttj06_cached_objectives_11-06-2026`. Its old jobs completed; this is evidence storage, not a running repair. Historical environment: Python 3.12.3, NumPy 2.2.6, SciPy 1.17.1; reverify the environment before new work. The old fits are small CPU jobs; do not assume GPUs or a full overnight run are necessary.

On 11-09-2026, local inspection found the shared SNAP reconnect guard in cooldown after four failures. Do not bypass or flood it. Read the current gateway recovery instructions, check `python3 /Users/brandomiranda/agents-config/scripts/snap_reconnect.py --status`, and reuse available existing connections. The old coordinator reported a local NumPy architecture mismatch; treat this as a historical environment issue to recheck, not proof that all local Python environments are unusable.

## Codex Desktop and account-access work still required

Verified now:

- `list_projects` returns this repository as an existing local Git project with the correct absolute path.
- The originating chat was moved successfully from its old project into this one via the supported project tool.
- The receiving chat has cwd `/Users/brandomiranda/judges-dont-need-psychometrics-or-corr` and was idle before this handoff.
- `/Users/brandomiranda/.codex`, `/Users/brandomiranda/.codex-vals` and `/Users/brandomiranda/.codex-su` all exist and have `sessions/` directories.

Still to do:

- Inspect where the relevant native session is actually stored and how the installed Desktop/profile configuration uses it. Directory existence alone does not prove cross-account chat visibility.
- Make the project, checkpoint and needed continuation context accessible from all three existing owned profiles; verify actual read/access paths. Use supported native behavior where possible and document limitations precisely.
- Read `/Users/brandomiranda/agents-config/workflows/multi-account-agent-clis.md` and `workflows/cross-machine-projects.md`. Keep credentials and live account/session databases separate; do not solve history sharing by merging or concurrently sharing writable SQLite databases. A shared project checkpoint/catalog can supply durable continuation, but it is not proof that native chats sync across accounts.
- Save private session identifiers and account mappings in private storage. Do not put authentication files or raw account/session transcripts into this public repository. No account credentials were inspected or changed by this handoff.

## Scientific context to preserve

The judge predicts a rating conditional on the Python input, candidate formal specifications and gold reference: `p(rating | x_input, y_pred, y_gold)`. Formal specifications mean unproved theorem statements. This experiment predicts human ratings; the separate AI-assisted formal-prover target study belongs in the sister repository. Matching a prover's budgeted proof-success outcomes is a different target from predicting expert rating distributions.

The user wants a fair test of information-based objectives against correlation-based optimization, not an assumed CE win or a psychometric framework. High correlation does not identify the full conditional distribution. Proper losses identify that target under their assumptions, but they do not guarantee better finite-sample performance on every held-out metric. Preserve this distinction in the paper and final report.

**TLDR-end:** [judge objectives: experiment 06 handoff] Diagnose and repair Pearson fitting, run a matched CE-versus-Pearson comparison in this repository, and finish cross-account continuation access. Preserve historical evidence and report what the corrected experiment actually establishes.
