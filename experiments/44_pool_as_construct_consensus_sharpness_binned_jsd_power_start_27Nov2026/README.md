# Experiment 44: the rater pool is the construct, consensus items, binned JSD, and power at one rating per item

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/44_pool_as_construct_consensus_sharpness_binned_jsd_power_start_27Nov2026/README.md>

**TLDR:** Pre-registered, CPU-only diagnostics of five frozen frontier judges on public multi-annotator sets: whether the choice of rater pool that defines p* changes scores or judge rankings, whether proper scores reward a peaked forecast on consensus items, what binned JSD can and cannot see, and how many held-out items a paired comparison needs at one rating per item. The protocol is frozen in `PROTOCOL.md` (amended in `AMENDMENT_A1.md` before any run); results are in `RESULTS.md`: pool changes did not detectably move distributions, scores or the judge ranking (low power), sharpening is cheaper on consensus than on contested items in 25 of 25 cells, binned JSD is exactly blind to within-bin forecast shuffles that proper scores penalize, and at one rating per item and 400 items 0 to 6 of the 10 judge pairs per set are resolvable.

**As of:** 11-27-2026 (Friday, Pacific time). **Status:** done; results in `RESULTS.md` (11-27-2026). Number 44 is the one the root README reserved for base-rate skill scores, human-agreement stratification and evaluation-set power. A parallel session froze Experiments 46 (rank reversal and skill score), 50 (strata by human entropy) and 52 (skill-score estimator simulation) for those parts, and Experiment 36 holds power curves for trained heads, so this folder does not repeat their tests; it adds the four analyses below.

## Construct and positioning (read first)

**Target.** What a judge is validated against is a conditional distribution, p*(y | c, R, rho): the distribution of the rating y that a rater drawn from the pool R gives item c under the rubric rho. The pool has three parts: who is eligible, how raters are weighted, and which question and scale they answer. Changing the pool or the rubric changes p*, so choosing R and rho is the construct and is done before any metric is computed. The estimate used here is the per-item rating histogram of the stated pool with uniform weights.

**Credit.** Pavlick and Kwiatkowski (TACL 2019) and Plank (EMNLP 2022) established that labels are distributions and argued for evaluating against them; Nie et al. (ChaosNLI, 2020), Uma et al. (2020) and Kurniawan et al. (2025) proposed soft metrics for NLP models. Nothing here claims the target, and "use a proper score" is not claimed either (Lyu, Do and Pergola, arXiv 2610.37868; Li et al., Findings of EMNLP 2026).

**What is claimed instead: the ruler, for LLM-judge validation.** (1) Identification: correlation and mean squared error see a judge only through its scalar score, so judges with the same conditional mean tie on them while a strictly proper score, which is uniquely minimized by the true distribution, separates the true distribution from any same-mean judge with a different one (Lean 4 and numerics, Experiments 35 and 36). Agreement with the majority label sees only the mode, so it ties judges with the same mode; that statement is Lean-checked in core Lean for any number of categories (`Point1.agrees_iff_mode` in `docs/notes/point1_support_27Nov2026/Point1Checks.lean`), while accuracy against the majority label being proper but not strictly proper stays argued by hand. (2) A protocol: a strictly proper score matched to the label encoding (one-hot Brier or cross-entropy, cumulative RPS for ordinal), a base-rate skill score built from the unbiased per-sample risk, and reporting by agreement stratum. (3) Evidence on practice and on real judges: the 49-paper check (Experiment 43) and the pre-declared census (Experiment 45); the calibration gap (Experiment 41); and the real-judge rank-reversal tests (Experiment 46: protocol frozen; results were pending when this was written and are now interim, see its README).

**Consensus items.** Where raters agree p* is peaked, and a good judge is peaked too; a proper score rewards that and punishes flattening there. So the statement is not that correlation is always wrong. It is that correlation does not identify the conditional distribution in general, and that scores should be read by agreement stratum. Arm S measures the peaked-versus-hedged trade directly.

## What each arm tests

| Arm | Question | Sets | Registered tests |
|---|---|---|---|
| P | Does the pool that defines p* change the judges' absolute scores or their ranking beyond sampling noise? | DICES-350, DICES-990 | rater-level permutation test on per-item pool distance; two-way (items by raters) bootstrap on score differences and on pair-ordering interactions (Amendment A1); Holm |
| S | Do proper scores reward sharpening on consensus items and penalize it on contested items? | all five | sharpening gain by stratum on an independent rater half; item bootstrap |
| J | Is binned JSD (Elangovan et al.) invariant to within-bin shuffles of a judge's forecasts, and how does its ranking relate to proper scores? | all five | exact invariance check; paired permutation risk; descriptive Kendall tau |
| M | How many items does a paired judge comparison need at one rating per item? | all five | analytic MDD from variance components; subsampling check |

## Pools and rubrics of the five sets (the construct, as documented in this repository)

| Set | Pool R | Rubric rho and scale | Items used |
|---|---|---|---|
| ChaosNLI SNLI | 100 crowd annotators per item, no rater ids, uniform weights over the 100 labels | NLI label: entailment / neutral / contradiction (nominal) | 600 of 1,514 items, all with 3-of-5 agreement among the original five labels, so chosen toward disagreement |
| ChaosNLI MNLI-m | same | same | 600 of 1,599, same selection |
| ChaosNLI alphaNLI | 100 annotators per item | which of two hypotheses is the more plausible explanation (binary) | 400 of 1,532 (full dev set) |
| DICES-350 | the same 123 raters rate every item; rater ids with gender, race, age and education; uniform weights | Q_overall: is the last chatbot response unsafe, Yes / Unsure / No, coded No = 0, Unsure = 1, Yes = 2 (ordinal) | 350, stratified toward clear safe or unsafe gold labels |
| DICES-990 | 172 raters appear in the file, 69 to 76 per item; same demographics | same | 400 of 990 (Experiment 28's subsample) |

The frontier judges of Experiment 28 were told to forecast the share of a whole crowd ("a crowd of human annotators", "a diverse crowd of human raters"). Arm P scores them against sub-pools on purpose, to measure how far conclusions move when the target pool changes; it does not claim a judge is wrong for a sub-pool.

## Files

| Path | Role |
|---|---|
| `PROTOCOL.md` | frozen pre-registration (hypotheses, definitions, tests, what the analyst had seen) |
| `README.md` | this page |
| `AMENDMENT_A1.md` | three statistical corrections plus clarifications, written before any run |
| `src/` | `lib44.py` (scores, binned JSD, power, pool helpers), `run_all.py` (the four arms), `verify.py` (independent per-rating loops), `summarize.py` (prints every number in `RESULTS.md`), `make_figures.py` |
| `tests/test_lib44.py` | 17 deterministic checks (score values and strict propriety, exact unbiasedness at one rating per item, temperature leaves the argmax, binned-JSD permutation invariance, power formulas by enumeration, pool bootstrap, determinism) |
| `results/` | nine CSVs, `summary.json`, `key_numbers.md`, and `sensitivity_drop_invalid/` |
| `figures/` | four figures, `ax.grid(True)` on every axis |
| `RESULTS.md`, `HOW_TO_READ_RESULTS.md` | results against every registered hypothesis, and which data, metric, subtraction, uncertainty and judge choice each number has |

## Reproduce

```bash
python3 experiments/39_proper_score_vs_d_study_budget_allocation_start_26Nov2026/src/fetch_raw_data.py   # public data, SHA-256 checked
python3 experiments/44_pool_as_construct_consensus_sharpness_binned_jsd_power_start_27Nov2026/src/run_all.py                      # about 20 seconds
python3 experiments/44_pool_as_construct_consensus_sharpness_binned_jsd_power_start_27Nov2026/src/run_all.py --out results/sensitivity_drop_invalid --drop-invalid
python3 experiments/44_pool_as_construct_consensus_sharpness_binned_jsd_power_start_27Nov2026/src/verify.py
python3 experiments/44_pool_as_construct_consensus_sharpness_binned_jsd_power_start_27Nov2026/src/summarize.py
python3 experiments/44_pool_as_construct_consensus_sharpness_binned_jsd_power_start_27Nov2026/src/make_figures.py
python3 -m pytest experiments/44_pool_as_construct_consensus_sharpness_binned_jsd_power_start_27Nov2026/tests -q
```

CPU only, $0, no API or GPU job. Amendments after the freeze go in `AMENDMENT_A1.md` with a date.

## Relation to other experiments

Experiments 43 and 45 (practice), 41 and 47 to 49 and 51 (calibration), 46 (rank reversal and skill), 50 (strata), 52 (skill estimator), 36 (power for trained heads), 35 and 33 (identification, Lean), 39 (item-versus-rater budget for selection).
