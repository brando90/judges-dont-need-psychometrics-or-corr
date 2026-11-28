# Prompt: abstract update from the advisor's line-by-line notes (27 Nov 2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/docs/prompts/abstract_advisor_line_notes_update_27Nov2026.md>

**TLDR:** Self-contained prompt for a coding agent (Claude Code style). It rewrites the paper abstract to about 200 words in answer to the advisor's notes S1 to S9 on the abstract committed in `42a4dd5`, after checking the Experiment 43 audit coding, re-expressing the Experiment 41 result as skill scores with confidence intervals, and finding published examples of raw judge scores used where level and scale matter. Paste the fenced block into the agent.

```text
CONTEXT
Repo: https://github.com/brando90/judges-dont-need-psychometrics-or-corr (position paper on validating LLM judges).
Paper: paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/ (abstract in 1-abstract.tex, with
"% CS197 Move N" comments above each sentence; build with `make` in that folder: pdfLaTeX + BibTeX, both
editions 00_main_aaai_2027.pdf and 00_main_arxiv.pdf). The current abstract is the one committed in 42a4dd5.
The advisor sent line-by-line notes on it (S1-S9 below). Address every note.

GROUND RULES (from AGENTS.md / CLAUDE.md / .cursor/rules)
- Work on main, no branches or PRs. git pull (or fetch the latest SHA) before every push; others commit concurrently.
- Never invent numbers or citations. Every number in the abstract must exist in a committed result file; cite the file.
- Any new mathematical claim: numerical check first (fixed seed, exact fractions or a named tolerance), then Lean 4
  (zero sorry, standard axioms only, #print axioms, compiler output saved). If Lean is not done, label the claim
  "numerically checked, Lean pending" and add it to docs/LEAN_TODO.md.
- New analyses go in a new numbered folder experiments/NN_<setup>_start_27Nov2026/ with PROTOCOL.md committed
  before the analysis runs, scripts, seeds, commands, outputs, README and HOW_TO_READ_RESULTS.md. Do not edit
  other workers' experiment folders (46-52 are owned by a parallel calibration follow-up worker); read their results.
- Result clarity: every number names its split, metric (with direction), difference order and interval.
- Refer to the reviewer only as "the advisor". No names, mail links or third-party model text in the repo.
- Do not send anything to anyone. $0 API spend, CPU only.

STEP A: FACTS FIRST
A1. Experiment 43 audit (experiments/43_llm_judge_validation_practice_survey_start_27Nov2026/): report how the 49
    papers were selected, which papers make up 49 - 43, and how Elangovan et al. (arXiv 2410.03775) and Li et al.
    (arXiv 2510.00263) are coded. Li et al. report Brier, which IS an item-level proper score: if either is coded
    "no item-level proper score", recode honestly and propagate the counts (README, tables, paper appendix,
    abstract). Also tally the advisor's narrower claim: an item-level, STRICTLY proper score (CE/log, KL to the
    item distribution, Brier, RPS) as opposed to a non-proper divergence (JSD, binned JSD), and whether any is
    applied to an ordinal scale. Add the stricter tally to tally.py only if the folder owner allows; otherwise
    compute it in the new experiment folder, reading papers.py read-only. Run tallies twice (determinism).
A2. Experiment 41 reanalysis (new folder): re-express the headline as skill scores
    Skill = 1 - S(head)/S(base rate), S = test normalized RPS, base rate = training-split marginal, per dataset and
    view (K = 2, 3, 5), for CE-, Pearson- and MSE-trained heads, raw and after each head's validation-fitted
    calibration map. Report skill differences (CE-trained minus Pearson-trained; positive = CE-trained better),
    item-clustered bootstrap 95% CIs (>= 1000 draws, test items resampled within each split seed, pooled over the
    5 split seeds), and the range over seeds. Reproduce Experiment 41's mean RPS exactly first (sanity check).
    Be explicit about the advisor's circularity point: CE-fit versus correlation-fit heads scored by RPS favors the
    proper fit by construction, and the small post-calibration gap is what r^2 = 1 - MSE(best affine fit)/Var(y)
    predicts. So the result supports "select by correlation, then recalibrate with labeled data", and the open
    question is how many labels and whether the map transfers: use Experiments 47 (label budget) and 48 (shift)
    results if they are on main, and Experiment 46 (real-judge rank reversal) only if it has results.
A3. Find 2-4 verified published examples where raw judge scores are used where level and scale matter:
    pass rates or filtering at a fixed score threshold, absolute scores compared across judges or benchmarks
    (e.g., MT-Bench 1-10 scores, AlpacaEval win rates, threshold-based data filtering or reward thresholds).
    Verify each against the paper text; add bib entries from arXiv/ACL Anthology metadata.

STEP B: ABSTRACT (about 200 words, hard ceiling 220), CS197 move comments kept
- S1 (context sentence): keep.
- S2: the claim "judge validation has not adopted metrics that evaluate a judge against that distribution" is false
  (binned JSD between human and judge rating distributions; MSE between judge probability and the multi-annotator
  mean). Narrow it to: no item-level, strictly proper score on ordinal scales, and match the audit sentence.
- S3: most people use correlation for ranking. Name the uses where level and scale matter (thresholded pass rates,
  scores compared across judges) and say raw judge scores are used there (cite A3 in the intro). Give the
  construct its own sentence: the target is the conditional rating distribution p*(y | item, rater pool, rubric).
- S4: drop psychometric reliability as validity evidence; name judge-human kappa / Krippendorff's alpha as
  agreement. In "discard calibration or collapse it to a mean or mode", "it" must be the distribution. Give the
  sampled-score result its own sentence and define it: judges sample at nonzero temperature; on such sampled scores
  correlation prefers a judge that outputs the conditional mean over one that samples from p*.
- S5: split the long sentence. State r^2 = 1 - MSE(best affine fit)/Var(y) exactly (inline math that renders in
  AAAI). Say the equal-means tie is immediate because the listed statistics are computed on the mean, and frame it
  as the point rather than a deep theorem. Drop the Lean sentence from the abstract (keep Lean in the body).
- S6: "adopt" not "propose" the ranked probability score (RPS; Epstein, 1969) for judges. Keep one reduction (RPS
  on hard predictions = normalized absolute error). Cut cross-entropy from the abstract. Normalized Brier for
  categorical labels only if words allow.
- S7: say exactly what is unbiased: the mean per-item score with one rating per item is unbiased for the expected
  score against p* (enough to rank judges); the distance to p* also needs the uncertainty term, which needs
  multiple ratings per item.
- S8: cut "We test this view in two complementary ways". State the audit selection and corrected counts, including
  what the remaining papers are. Report skill-score differences, not the max raw RPS gap, and present the
  experiment honestly given the circularity (correlation-selected judges need recalibration with labeled data,
  quantified by skill), not as proof of an artifact.
- S9: drop "artifact" unless a real-judge rank reversal result exists on main. Recommend reporting a proper score
  alongside correlation, with the calibration (reliability) and resolution terms split out (Murphy decomposition).
- Overall: about 200 words.

STEP C: BODY ALIGNMENT (minimal)
- Intro/contributions: narrowed gap claim (credit binned JSD and MSE-to-mean work), "adopt" not "propose", the
  sampled-score result in its own sentence, the level/scale uses with A3 citations, and the decomposition
  reporting recommendation. Update audit counts in the body, tables and appendix if A1 changed them.
- Add a verified Epstein (1969) bib entry (J. Applied Meteorology 8(6):985-987, "A scoring system for probability
  forecasts of ranked categories"; check DOI 10.1175/1520-0450(1969)008<0985:ASSFPF>2.0.CO;2).
- Any new math claim: numerics plus Lean, or labeled.

STEP D: BUILD AND REPORT
- Compile both editions: zero errors, zero undefined citations/references, zero "??". Report the venue page count
  and the page where the Conclusion starts. PDFs that cannot be pushed stay on the build machine.
- Write docs/notes/advisor_abstract_notes_response_27Nov2026.md mapping each note S1-S9 to what changed, or why
  not, with links to evidence. Draft only; send nothing.
- Final report: commit SHAs, new abstract (plain text) with word count, audit counts and the remaining papers,
  skill-score results with CIs, level/scale citations, note-by-note summary, page counts, unresolved items.
```
