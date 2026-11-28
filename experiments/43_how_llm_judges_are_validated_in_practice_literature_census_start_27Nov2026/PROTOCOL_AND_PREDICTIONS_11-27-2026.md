# Experiment 43: how LLM judges are validated in practice. Protocol and predictions

**Written:** Fri 11-27-2026, 17:27 PST, before any census row was collected. Repository state when written: `main` at `0897ad5`. This file is committed before any file that holds census rows, so the commit order shows that the protocol came first.

**Why this exists.** Brando (11-27-2026, 17:16 PST): the correlation-versus-cross-entropy paper rests on "breaking trust in the ruler", which only matters if correlation with human ratings really is the ruler people use to validate LLM judges. His handwritten note of 11-07-2026 (formulation ①) says "this is what people are doing" and lists "deep research to justify/cite if this is true" as a TODO. This experiment is that research. The decision rules and predictions below are fixed in advance so the result is allowed to contradict the paper's framing.

**Prior information I had before starting (declared so it cannot leak into the coding):**

1. The project's 11-09-2026 literature check (about 90 papers read by abstract and key passages, none end to end) found no paper that states maximum correlation as its principle for choosing a judge, and none among 21 Chinese-company reports that chose a judge by maximum correlation. It reported that two papers (arXiv 2510.09738 and arXiv 2503.05965) say correlation alone selects judges poorly. I have not re-read those two papers in this session.
2. The 11-07-2026 note tabulates nine sources by how the judge was developed and what evidence was used. I recall that list. I do not use its content as coded data; every source is re-read here.
3. From this project's own experiments: system-level Spearman across 16 summarization systems is 0.877 for a raw Pearson-trained fit and 0.835 after calibration, so correlation behaves well for ranking systems. This is why the census records the level at which a correlation is computed.

## 1. Questions

1. **Q1 (headline metric).** Among sources that validate an automatic judge, metric or grader against human judgments, what share uses a correlation (Pearson, Spearman or Kendall) as its headline agreement number?
2. **Q2 (label type).** Does the answer depend on the human label type: graded scores, pairwise preferences, categorical labels, or system rankings only?
3. **Q3 (distribution).** What share reports any proper scoring rule or any comparison between the judge's output distribution and the human rating distribution?
4. **Q4 (selection).** What share chooses, tunes or trains the judge (prompt, rubric, model, hyperparameters) by maximizing agreement or correlation with human ratings? These are Brando's formulations ① and ②.
5. **Q5 (level).** Is the correlation computed per item (instance, segment, summary) or per system (leaderboard)?

## 2. Frame and sampling

The sample is **purposive, not random**. Every proportion below describes this sample only. It is not an estimate for the field, and the report says so.

1. **Unit.** One row per source (a paper, a technical report or a benchmark release). If a source validates several judges or datasets, the row codes its headline validation (the one in the abstract, else the first main results table).
2. **Strata ("lineages").** Correlation is the natural ruler for some label types and not for others, so the census is stratified by what the human label is:
   - L1: graded-score judges and metrics (Likert, continuous or rubric scores). Quota 14.
   - L2: pairwise-preference judges and reward models. Quota 10.
   - L3: benchmark-level and system-ranking validation (leaderboards, Chatbot Arena agreement). Quota 6.
   - L4: industry and frontier-lab graders and system cards. Quota 6.
   - L5: information-retrieval relevance judgments and annotation replacement (categorical labels). Quota 6.
   - L6: distributional, calibration and critique papers. Quota 8. **L6 is selected for the property under study, so it is excluded from every base-rate denominator.** It is used to describe alternatives, not to estimate rates.
3. **Named anchors, declared before any search.** Names are from memory. Title, authors and year are taken from the fetched page and recorded in `census.csv`. Sources are processed **in the order listed**, and processing stops when the lineage quota is filled. An anchor that cannot be located or read is dropped and listed with its reason. Dropping is decided by availability only, never by what the source reports.
   - L1 (LLM era): L1-01 G-Eval; L1-02 GPTScore; L1-03 "Is ChatGPT a Good NLG Evaluator?"; L1-04 Chiang and Lee, "Can Large Language Models Be an Alternative to Human Evaluations?"; L1-05 Shen et al., "Large Language Models are Not Yet Human-Level Evaluators for Abstractive Summarization"; L1-06 Gao et al., "Human-like Summarization Evaluation with ChatGPT"; L1-07 LLM-Eval; L1-08 Prometheus; L1-09 Prometheus 2; L1-10 FLASK; L1-11 TIGERScore; L1-12 InstructScore; L1-13 AutoCalibrate ("Calibrating LLM-Based Evaluator"); L1-14 GEMBA (Kocmi and Federmann); backfill L1-15 JUDGE-BENCH (Bavaresco et al.); L1-16 Themis.
   - Pre-LLM context rows, not in any quota and tallied separately as "origin of the habit": P-01 SummEval; P-02 BERTScore; P-03 BARTScore; P-04 UniEval.
   - L2: L2-01 MT-Bench and Chatbot Arena judge paper (Zheng et al.); L2-02 PandaLM; L2-03 JudgeLM; L2-04 Auto-J; L2-05 "Large Language Models are not Fair Evaluators"; L2-06 RewardBench; L2-07 JudgeBench; L2-08 "Judging the Judges" (Thakur et al.); L2-09 Self-Taught Evaluators; L2-10 "Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge"; backfill L2-11 "Humans or LLMs as the Judge?"; L2-12 RM-Bench; L2-13 "Aligning with Human Judgement: The Role of Pairwise Preference in LLM Evaluators".
   - L3: L3-01 AlpacaFarm; L3-02 length-controlled AlpacaEval; L3-03 Arena-Hard ("From Crowdsourced Data to High-Quality Benchmarks"); L3-04 WildBench; L3-05 MixEval; L3-06 JuStRank; backfill L3-07 "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference".
   - L4: L4-01 HealthBench; L4-02 FACTS Grounding; L4-03 PaperBench; L4-04 TruthfulQA (GPT-judge); L4-05 "The AI Scientist" automated reviewer; L4-06 "RLAIF vs. RLHF"; backfill L4-07 Bloom (Anthropic); L4-08 SimpleQA.
   - L5: L5-01 Faggioli et al., "Perspectives on Large Language Models for Relevance Judgment"; L5-02 Thomas et al., "Large language models can accurately predict searcher preferences"; L5-03 Gilardi et al., "ChatGPT outperforms crowd-workers for text-annotation tasks"; L5-04 RAGAS; L5-05 ARES; L5-06 "Who Validates the Validators?"; backfill L5-07 LLMJudge challenge overview.
   - L6: L6-01 "Beyond Correlation: The Impact of Human Uncertainty in Measuring the Effectiveness of Automatic Evaluation and LLM-as-a-Judge"; L6-02 "Judging with Confidence: Calibrating Autoraters to Preference Distributions"; L6-03 arXiv 2510.09738; L6-04 arXiv 2503.05965; L6-05 "AutoEval Done Right"; L6-06 Soboroff, "Don't Use LLMs to Make Relevance Judgments"; L6-07 Stureborg et al., "Large Language Models are Inconsistent and Biased Evaluators"; L6-08 Meister et al., "Benchmarking Distributional Alignment of Large Language Models"; backfill L6-09 "Rating Roulette".
4. **Backfill queries, declared now.** One per lineage, used only to replace dropped anchors and only up to the quota. L1: "LLM evaluator correlation with human judgments summarization dialogue". L2: "LLM judge pairwise preference agreement with human annotators". L3: "benchmark correlation with Chatbot Arena ranking judge". L4: "model grader validated against expert human labels". L5: "LLM relevance judgments agreement with human assessors". L6: "calibration of LLM judge human rating distribution proper scoring rule". The first result that is a research paper or technical report and is not already in the census is taken.
5. **Minimum for a verdict.** At least 40 sources coded from a full or partial read. With fewer, the report says "underpowered" and gives no verdict.

## 3. Coding scheme (closed vocabularies)

1. `human_validation`: Y if the source compares the automatic judge, metric or grader with human judgments in its own experiments; N otherwise (such rows are listed and excluded from denominators).
2. `era`: `llm` or `pre_llm`.
3. `label_type`: `GRADED`, `PAIRWISE`, `CATEGORICAL`, `SYSTEM_RANK`, `MIXED`.
4. `headline_family`, the family of the first agreement number in the abstract (else the first main results table): `CORR_INST`, `CORR_SYS`, `ACC_AGREE`, `KAPPA_ALPHA_ICC`, `F1_PRF`, `DIST`, `OTHER`.
   - Correlation family means Pearson r, Spearman rho or Kendall tau (any variant, including "tau-like"), between automatic scores and human scores, or between automatic and human system rankings.
   - Accuracy, agreement rate and pairwise accuracy are `ACC_AGREE`. Cohen's, Fleiss and Krippendorff coefficients, ICC and Lin's concordance coefficient are `KAPPA_ALPHA_ICC` (a sensitivity analysis counts ICC and concordance as correlation-like). Precision, recall and F1 are `F1_PRF`.
   - `DIST` means a proper scoring rule (log loss, cross-entropy, negative log-likelihood, Brier, ranked probability score, CRPS) or a distance between the judge's output distribution and a human rating distribution (KL, JS, Wasserstein, total variation) or a calibration error (ECE).
5. `corr_any`: Y if any Pearson, Spearman or Kendall number against human judgments is reported anywhere in the read text.
6. `corr_level`: `INSTANCE` (per item, segment, summary or turn), `SYSTEM` (per system or model, including leaderboard rank correlation), `BOTH`, `NA`.
7. `acc_any`, `kappa_any`, `f1_any`, `dist_any`: Y or N.
8. `selection_by_corr`: Y only if the source states that the judge, prompt, rubric or hyperparameters were chosen or tuned to maximize agreement or correlation with human ratings on development data; N if it states another selection or none; UNCLEAR if the read text does not say.
9. `trained_on_human_loss`: Y if the judge is fine-tuned or trained with a loss on human labels (supervised, regression or preference loss); N; UNCLEAR.
10. `coverage`: `full`, `partial` (the evaluation section was not clearly covered by the read), `abstract_only`.
11. `verified`: Y if the row was re-read by the verification procedure in section 5.
12. Free fields: `id`, `citation` (authors, year, title from the page), `year`, `venue_or_type`, `task`, `n_raters_per_item` (if stated), `source_url`, `notes` (paraphrase only; no quotation longer than ten words).

## 4. How each source is read

The reading engine is the `WebFetch` tool, where a small model reads the page and answers a fixed prompt. Its errors are measured by the verification re-read. Sources are located by `WebSearch` first, because `WebFetch` is only used on URLs that a search returned. Search results that name the wrong paper are discarded. The first line of every record must repeat the paper title; a mismatch voids the read.

**PROMPT-A (first read, used for every source):**

"Report as a compact record. First line: exact title, authors, year. (1) What is validated against human judgments: a judge model, a metric, a grader, or a benchmark? (2) Human label type: graded scores, pairwise preferences, categorical labels, or rankings; annotators per item if stated. (3) Every metric used to measure agreement with humans, named exactly (Pearson, Spearman, Kendall tau, accuracy or agreement rate, Cohen's kappa, Krippendorff's alpha, ICC, F1, AUC, NLL or cross-entropy, Brier, ranked probability score, KL or JS divergence, Wasserstein, total variation, ECE), and for each one whether it is computed per item or per system. (4) Which metric is the headline in the abstract or the first main results table? (5) Was the judge, prompt, rubric or hyperparameter set chosen, tuned or trained by optimizing agreement or correlation with human ratings? Answer 'correlation-selected', 'trained with a loss on human labels', 'no selection stated' or 'unclear'. (6) Does it compare the judge's output distribution with a human rating distribution or use a proper scoring rule? Write 'not found' wherever the text is silent. Do not guess. Quote at most ten words in total."

**PROMPT-B (verification re-read):**

"Answer yes, no or not found for each, with the section or table: (a) Pearson correlation between automatic scores and human scores reported? (b) Spearman? (c) Kendall tau? (d) Any correlation computed over systems or models rather than items? (e) Accuracy or agreement rate with human labels or preferences? (f) Cohen's kappa, Fleiss kappa, Krippendorff's alpha or ICC? (g) F1? (h) Cross-entropy, log loss, NLL, Brier, ranked probability score, KL, JS, Wasserstein, total variation or ECE against human labels? (i) Was the judge, prompt or rubric chosen or tuned to maximize agreement or correlation with human ratings? (j) What is the first agreement number or metric named in the abstract?"

## 5. Verification

A seeded script (`tally.py --verify-sample`, seed 0) picks at least 12 coded rows at random, stratified so that every lineage in the base-rate set appears at least once. Each is re-read with PROMPT-B and compared with the stored codes for `corr_any`, `corr_level`, `acc_any`, `kappa_any`, `dist_any`, `selection_by_corr` and `headline_family`. The report gives the field-level agreement rate, lists every disagreement and says how it was resolved (by re-reading the page again with a narrowed question). Rows corrected this way are corrected in `census.csv`, and the correction is listed.

## 6. Tally and decision rules

1. **Denominators.** `D_all` = rows with `human_validation = Y`, `era = llm`, lineages L1 to L5, `coverage` full or partial. `D_graded`, `D_pair`, `D_cat`, `D_sys` are the subsets by `label_type`.
2. **Descriptive intervals.** Wilson 95% intervals for every proportion, with the label "descriptive of this purposive sample". The pooled estimate is given twice: pooled over sources, and lineage-balanced (equal weight per lineage, with a stratified bootstrap over sources within lineage, seed 0, 2000 draws), because lineage sizes were chosen by me.
3. **Sensitivity.** Repeat with `abstract_only` rows added, and with ICC and concordance counted as correlation-like.
4. **Verdicts, fixed now:**
   - **V1 (graded-score ruler).** If at least 80% of L1 rows have a correlation-family headline and the Wilson lower bound is at least 65%, the answer is "correlation is the standard ruler for graded-score LLM judging".
   - **V2 (general ruler).** Pooled L1 to L5 correlation-family headline share: at least 70% with Wilson lower bound at least 55% means "standard for LLM-judge validation in general"; 40% to 70% means "one of several common rulers"; below 40% means "not the standard ruler".
   - **V3 (selection objective).** If fewer than 15% of the sources that select or tune a judge on human data have `selection_by_corr = Y`, the answer is "correlation is a reporting metric far more than a selection objective, so formulations ① and ② describe few papers".
   - **V4 (distributional).** If fewer than 10% of D_all have `dist_any = Y`, the answer is "proper or distributional scoring against humans is rare".
5. **What the verdicts mean for the paper (written now).** If V1 and V2 both hold, scope is unchanged. If only V1 holds, scope the claim to graded-score judging and to the habit of fitting or selecting on correlation, and say that accuracy and agreement rate for pairwise and categorical judges ignore the human distribution in their own way. If neither holds, the paper would be breaking a ruler that few use as the headline, and the target should move to the objective used to fit or select judges.

## 7. Predictions (my honest priors, before reading any source)

1. L1: correlation-family headline in about 90% of rows (instance or summary level for most, system level for a few).
2. L2: correlation-family headline in about 15%; accuracy or agreement in about 80% to 90%.
3. L3: correlation-family headline in about 85%, almost all at system level (rank correlation with a human leaderboard).
4. L4: correlation-family headline in about 25%; F1, accuracy or agreement in most of the rest.
5. L5: correlation-family headline in about 30% (system-ranking rank correlation); kappa or accuracy in most of the rest.
6. Pooled L1 to L5 (over sources): correlation-family headline about 50%, with any correlation reported about 60%. So V1 holds, V2 lands in "one of several common rulers", and I expect the lineage-balanced figure to be a few points lower than the pooled one.
7. `dist_any = Y` in D_all: about 3% (V4 holds).
8. `selection_by_corr = Y`: at most 2 sources of the base-rate set (AutoCalibrate is the one I expect), so V3 holds.
9. Overall verdict I expect: correlation is the standard ruler for graded-score judging and for system-level ranking, it is not the standard ruler for pairwise-preference, categorical or retrieval judging, it is rarely the selection objective, and proper scoring against the human distribution is almost absent. If that holds, the "broken ruler" framing is accurate for the graded-score setting and needs scoping elsewhere.

## 8. What this census cannot show

1. It measures what published sources **report**, not how teams validate judges internally.
2. The anchor list is from my memory and skews toward well-known 2023 to 2025 papers, which is where the habit is most visible. The quotas and the two pooled estimates limit this, and they do not remove it.
3. Influence is not measured (no citation counts), and the sample is not random.
4. It cannot show that correlation is inappropriate for ranking systems. Our own SummEval result says it is fine there.
5. The reading engine is a small model. The verification re-read gives an error rate for the fields that matter, and rows outside the verification sample may still contain errors.

## 9. Files this experiment writes

`README.md`, this protocol, `census.csv`, `tally.py`, `tally_output.txt`, `verification.csv`, `REPORT_11-27-2026.md`, `HOW_TO_READ_RESULTS.md`, a paste-ready summary for Google Docs, a `.docx` copy of the report, and a row in `experiments/README.md`.
