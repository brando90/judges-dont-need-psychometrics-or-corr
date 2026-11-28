# How to read this folder

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/43_llm_judge_validation_practice_survey_start_27Nov2026/HOW_TO_READ_RESULTS.md>

This is a literature check, not an experiment. It has no train, validation or test split and no judge results.

1. **Data:** 37 published papers that validate an LLM judge, an automatic metric or a judge-based benchmark against humans (rows in `papers.py`), plus background and survey papers cited in `README.md`. Purposive convenience sample, not a systematic review.
2. **Metric:** for each paper, which families of validation statistic it reports: C correlation (Pearson, Spearman, Kendall), A raw agreement or accuracy, K chance-corrected agreement (kappa, Scott's pi, Krippendorff alpha), P a proper score or divergence against item-level human rating distributions, Ps a proper score at the system level only. A paper can have several codes.
3. **Numbers:** every count and percentage in `README.md` Section 3 is printed by `python3 tally.py`; a percentage is (papers with the code) / 37 and describes the sample, not the field.
4. **Uncertainty:** none quantified. Coding was done by one agent from targeted full-text search; there is no second coder.
