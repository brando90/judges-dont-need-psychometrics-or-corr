# How are LLM judges validated against humans in practice? A pre-declared literature census (Experiment 45, 27 Nov 2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/45_how_llm_judges_are_validated_in_practice_literature_census_start_27Nov2026/README.md>

**Google Doc copy (private to Brando):** <https://docs.google.com/document/d/1X8FiUNw8gDDFGC0rn4CZK_o950ojzMm9VJ83khUxukI/edit>

**Status:** complete, Fri 11-27-2026. No model training and no judge API calls; the reads used the session's web fetch tool.

## 1. Question

Brando (11-27-2026): the correlation-versus-cross-entropy paper breaks trust in a ruler. Is that ruler, correlation with human ratings, really how LLM judges are validated in ML and AI? Otherwise the paper breaks a ruler people are not using. This is also the open item in his handwritten note on formulation ① ("this is what people are doing", check by deep research).

## 2. Answer in brief

Partly yes. In 40 human-validated sources, a correlation is the headline in 16 (40%; Wilson 95% 26% to 55%): 12 of 14 graded-score judges and metrics, 4 of 6 leaderboard-ranking validations, and none of the 20 pairwise-preference, frontier-lab and retrieval or annotation sources, which lead with accuracy, agreement, F1 or kappa. A correlation is reported somewhere in 24 of 40 (60%). 25 of the 40 select, tune or train a judge on human data; 9 of those 25 (36%) say they maximized agreement or correlation with human labels and 4 (16%) correlation alone. 1 of 40 scores the human rating distribution with a proper rule, at the system level. The full answer, with the consequence for the paper, is in [`REPORT_11-27-2026.md`](REPORT_11-27-2026.md).

## 3. Files

1. [`REPORT_11-27-2026.md`](REPORT_11-27-2026.md) and `REPORT_11-27-2026.docx`: the answer, the results, the decision rules, the cross-check, the limits and the numbered sources.
2. [`PROTOCOL_AND_PREDICTIONS_11-27-2026.md`](PROTOCOL_AND_PREDICTIONS_11-27-2026.md): the protocol, predictions and decision rules, committed before any census row existed.
3. `census.csv`: 53 rows by 25 columns, final after the corrections below. `census_edits_log.jsonl`: the 34 field-level edits with old value, new value and reason.
4. `verification.csv`: the second read (20 rows: 12 seeded, 4 targeted extras, 4 added selection checks).
5. `tally.py`, `tally_output.txt`, `fig_headline_share_by_lineage.png`: every number in the report and the figure.
6. `compare_with_exp43.py`, `compare_with_exp43_output.txt`: the cross-check against the parallel Experiment 43.
7. `make_sources.py`: prints the numbered Sources list of the report from `census.csv`.
8. [`HOW_TO_READ_RESULTS.md`](HOW_TO_READ_RESULTS.md): data, metric, comparisons and uncertainty.
9. `GOOGLE_DOCS_SUMMARY_11-27-2026.md`: the paste-ready summary and changelog for Google Docs.

## 4. Reproduce

From this folder, with Python 3.10+, numpy and matplotlib:

1. `python3 tally.py --figure` prints the full tally (identical on a second run) and rewrites the figure.
2. `python3 tally.py --verify-sample` prints the seeded verification sample.
3. `python3 compare_with_exp43.py` prints the cross-check (it reads `../43_llm_judge_validation_practice_survey_start_27Nov2026/papers.py` read-only).
4. `python3 make_sources.py` prints the Sources section.

Re-reading the sources is not deterministic (the web pages and the reading model can change); the census rows and `census_edits_log.jsonl` are the record of what was read on 11-27-2026.

## 5. Method in brief

The protocol fixes the design; this is only a summary. 53 sources in seven groups were chosen on purpose, not at random (graded-score judges 14, pairwise-preference judges 10, benchmark and leaderboard validation 6, frontier-lab graders 6, retrieval and annotation replacement 6, critiques 8, pre-LLM context 3). The base-rate denominator is the 40 sources of the first five groups that validate an LLM judge against human labels and were read in full (26) or in part (14). Each source was read with a fixed extraction prompt into a closed 25-column vocabulary. The headline statistic is the family of the first agreement number in the abstract, else in the first main results table. Four decision rules (V1 to V4) and the predictions were declared before any data.

## 6. Deviations from the protocol

1. **Vocabulary additions.** Two coded fields and two values were added during coding: `selection_loose` (human agreement informed the design without being stated as maximized), `selection_metric` (CORR, AGREEMENT, BOTH or NA), and the values `CORR_UNSTATED` (a correlation headline whose unit is not stated) and `UNSTATED` (a correlation whose level is not stated). The vocabulary stayed closed and is enforced by the add and update scripts.
2. **Extraction prompt.** A one-sentence "Scope:" prefix was added to the extraction prompt for the L3, L4 and some L5 reads, saying what kind of validation to look for.
3. **Verification prompt.** The second-read prompt was prefixed with "First line: exact title." so that each answer names the paper it read.
4. **Dropped and backfilled anchors.** The original Prometheus paper (L1-08) was dropped because its arXiv pages were refused by the fetch tool and not retried; JUDGE-BENCH (L1-15) was the backfill and Prometheus 2 is L1-09. BARTScore (P-03) was dropped for lack of a readable source. L6-07 was dropped once and then recovered.
5. **Coverage upgrades.** L3-01, L3-02, L2-10 and L1-14 had their coverage upgraded by targeted fetches after the first read. At the upgrade L1-14 was recoded from system-ranking to graded and its `corr_any` from N to Y.
6. **Counting rule for the minimum.** The 40-source minimum for a verdict counts L1 to L5 sources read in full or in part regardless of `human_validation`; 42 were read, so it is met. The denominator itself (40) requires `human_validation=Y`.
7. **Interim test run.** `tally.py` was run once on 49 interim rows to test it. No code was changed in response to that run.
8. **Sensitivities.** The protocol declares two (abstract-only rows added; kappa, alpha and ICC counted as correlation-like). The report adds a co-headline reading (C) and reports any correlation anywhere (D), which the protocol used only as a prediction target.
9. **Added verification checks.** After the first pass showed the selection field was the least reliable, four selection-only re-reads (L4-02, L4-06, L5-02, L5-06) were added. They were decided after seeing the pattern, so they are outside the declared sample; all four confirmed the coding.
10. **Void re-read.** The second read of L2-03 (JudgeLM) returned a paper summary instead of the requested fields in two attempts and is excluded from the agreement rates. The first-pass row stands.
11. **Refused fetches.** Three fetches were refused for lack of approval and not retried: the arXiv HTML page for 2306.05685 (MT-Bench) and the arXiv abstract pages for 2303.16634 (G-Eval) and 2310.08491 (the original Prometheus). MT-Bench was read from a version found by search and G-Eval from a different host.
12. **Experiment number.** The protocol was committed under the number 43. Experiment 43 was then taken on `main` by a parallel literature check, and the root README reserves 44 for the base-rate skill-score study, so this folder is 45.

## 7. Corrections after the second read

Three census rows were changed after a narrowed re-read, all logged in `census_edits_log.jsonl`: L1-15 (correlation level from unstated to instance, selection from unclear to no), L2-01 (selection from no to unclear, loose selection yes) and L1-05 (selection from yes to unclear, which had over-coded it as selection by correlation). L3-06 and L5-05 disagreed on the second read and were kept as coded after a narrowed re-read. Field-level agreement before resolution was 98 of 105 (93.3%) over 15 usable rows; selection was the weakest field (11 of 15).

## 8. Related work in this repository

1. **Experiment 43** ([`../43_llm_judge_validation_practice_survey_start_27Nov2026/`](../43_llm_judge_validation_practice_survey_start_27Nov2026/README.md)) is a parallel 49-paper, single-coder check of the same question that was already on `main`. The two agree on direction (section 7 of the report). Neither folder was edited by the other; this one reads that one's `papers.py` read-only.
2. **The 11-09-2026 literature check** (about 90 papers by abstract and key passages) found no paper stating maximum correlation as its principle for choosing a judge. This census finds four sources that pick a configuration by correlation with human labels as a design step, and nine that pick by agreement or correlation.
3. **The root README** still carries its own practice sentence based on Experiment 43. It was not edited. If it should add that correlation is the headline for graded and ranking validation and not for pairwise, frontier-lab or retrieval validation, that is a one-sentence change for Brando to approve.

## 9. Limits

See section 9 of the report: reports not practice, a purposive sample, borderline verdicts (V1 turns on one row, V2 sits on its boundary, V3 on its numerator and denominator), coding choices, partial reads, a single coder plus a machine second read, and three refused fetches.

**TLDR-end:** [literature census, Experiment 45] Correlation with humans is the headline for graded-score judges (12 of 14) and ranking validation (4 of 6) but not for pairwise, frontier-lab or retrieval judges (0 of 20), so the paper should target scalar agreement with one aggregated human label and name correlation as its most common form for graded ratings. Judges are often chosen by agreement with humans (9 of 25 selecting sources), and only 1 of 40 sources scores the human rating distribution.
