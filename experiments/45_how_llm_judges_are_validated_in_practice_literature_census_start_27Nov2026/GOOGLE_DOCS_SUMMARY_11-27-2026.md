# Is correlation really how LLM judges are validated? Literature census summary and changelog (Fri 11-27-2026)

## Answer

Partly yes. Correlation with human ratings is the usual headline for graded-score judges and for validating leaderboard rankings, but not for pairwise, categorical, retrieval or frontier-lab judges, which lead with accuracy, agreement, F1 or kappa. What every lineage shares is that the judge is scored against one human label or score per item with a scalar statistic, almost never against the human rating distribution.

1. Headline statistic, 40 human-validated sources: a correlation in 16 (40%; Wilson 95% interval 26% to 55%). Accuracy or agreement also leads in 16.
2. By type: graded-score judges and metrics 12 of 14, leaderboard ranking validation 4 of 6, pairwise-preference judges 0 of 8, frontier-lab graders 0 of 6, retrieval and annotation replacement 0 of 6.
3. A correlation is reported somewhere in 24 of 40 (60%), including all 20 graded-score and ranking sources.
4. Selection: 25 of the 40 select, tune or train a judge on human data. 9 of those 25 (36%) chose by maximizing agreement or correlation with human labels; 4 (16%) by correlation alone, a fragile number.
5. Proper scoring against the human rating distribution: 1 of 40, and that one is a system-level Brier score.
6. The pre-declared thresholds were missed: the graded-score threshold by one row (12 of 14 has a Wilson lower bound of 60% against the 65% required; 13 of 14 would meet it) and the general threshold by a wide margin (40% against 70%).
7. A parallel 49-paper check (Experiment 43) agrees on direction; on the 20 shared papers the two codings agree on correlation in 17 and on every other field in 20.

## Wording for the paper

1. Write the target as scalar agreement with one aggregated human label, and name correlation as its most common form for graded and ranking tasks (7 of 8 graded-score sources, 4 of 5 ranking sources).
2. Do not write that correlation is the ruler for LLM judging in general: it is the headline in 0 of 20 pairwise, frontier-lab and retrieval sources.
3. For formulation ①, write that choosing a judge configuration by agreement with human labels, often on the same data, is common (9 of 25). Do not write that training on correlation is what people do (4 of 25, borderline).

## What changed in the repository

1. Added the experiment folder `experiments/45_how_llm_judges_are_validated_in_practice_literature_census_start_27Nov2026/` with the report (Markdown and Word), README, protocol and predictions (committed before any row), the 53-row census, the second-read verification, the tally script and its output, the figure, the cross-check against Experiment 43, a how-to-read note and this summary.
2. Added one row for Experiment 45 to `experiments/README.md`.
3. Numbering: this study is 45 because 43 is the parallel check already on `main` and the root README reserves 44.
4. Three census rows were corrected after the second read (L1-15, L2-01, L1-05); all 34 field edits are logged.
5. Not changed: the root README, agents-config, the sister repository and the live blog.

## Decisions for you

1. Approve or edit the wording for the paper above.
2. Say "update the README" if the root goal block should add that correlation is the headline for graded and ranking validation but not for pairwise, frontier-lab or retrieval validation.
3. Say "retry" to try the three refused fetches again (the original Prometheus paper, the G-Eval abstract page, the MT-Bench HTML page). The census does not depend on them.
4. Confirm number 45, or give another.

## Limits

1. The census measures what published sources report, not internal practice, and it is a purposive sample, so every share describes this sample and not the field.
2. One verdict is borderline on a single row, another sits on its boundary, and the selection field is the least reliable (11 of 15 answers agreed on the second read).
3. The second read used the same kind of reader, so it is a consistency check and not human verification.

## Links

Full report: [Google Doc](https://docs.google.com/document/d/1X8FiUNw8gDDFGC0rn4CZK_o950ojzMm9VJ83khUxukI/edit). Repository copy: [REPORT_11-27-2026.md](https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/45_how_llm_judges_are_validated_in_practice_literature_census_start_27Nov2026/REPORT_11-27-2026.md).

**TL;DR.** Correlation is the headline for graded-score and ranking validation (16 of 40 overall) but not for pairwise, frontier-lab or retrieval judges, so scope the paper's target to scalar agreement with one aggregated human label. Judges are often chosen by agreement with humans, and almost nothing scores the human rating distribution.
