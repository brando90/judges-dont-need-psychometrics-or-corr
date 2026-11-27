# Real multi-annotator judging data: what we hold, what was run, what is missing | 11-26-2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/gemini_plan_claims_audit_11-26-2026/REAL_JUDGING_DATA_INVENTORY_11-26-2026.md>

**TLDR:** Brando asked (11-26-2026, 14:53 PST) whether we already have real judging data for the top-conference claim, and whether the VeriBench formal specification coverage (FSC) ratings can join the suite. Yes to both. Of the plan's five datasets (SummEval, FSC, POPQUORN, DICES, ChaosNLI, with HelpSteer2 optional), every one except POPQUORN already has a real-data run in this repository, and POPQUORN's files load from a repository that Experiment 36 did not find. The FSC ratings are usable only through the sealed splits in the sister repo, and neither Cursor experiment (37, 38) follows them.

**How to read this file.** It is an inventory, so it reports counts, not results. The counts for SummEval, ChaosNLI, DICES, POPQUORN and FSC are printed by scripts in this folder: `dataset_facts.py` (public sets, sister commit `f0d0fc3`, output `results/dataset_facts.out`) and `fsc_counts_11_26.py` (the frozen 11-26 FSC files, sister commit `7855746`, output `results/fsc_counts_11_26.out`). Both scripts were run twice with identical output. The counts for the HelpSteer2 splits, KonIQ-10k, the homework data, and every earlier experiment's own splits and test-item overlap (Experiments 24, 28, 29) are copied from that experiment's README or data note and were not recomputed here. No metric, difference or interval appears in this file. No rater name or id appears in this file or in the scripts' output.

## The standard this inventory serves

1. Every paper-facing or conference-facing empirical claim needs real multi-annotator human judging data. A synthetic law may check a proof against a known target and is labelled as that check (repository CLAUDE.md, 11-26-2026).
2. The report uses the unbiased per-sample proper risk at test time, a prompt-level train, validation and test split (no rater of a prompt in two splits), and item-clustered bootstrap 95% confidence intervals.
3. The bar is a top machine-learning conference (NeurIPS, ICML, ICLR).

## The inventory

Each item gives the unit of a prompt, the scale (K categories), the number of raters per prompt (the ceiling for any sweep over the number of ratings k), where it is already used, and what to watch.

1. **VeriBench FSC (formal specification coverage), private plus anonymized.** A judge scores whether a Lean formalization covers the specification of a Python task. Scale 0 to 5 (K = 6). A prompt is one artifact (one formalization candidate); the split group is the task, because several artifacts share a task.
   - Old set: 75 artifacts, 25 tasks, 195 expert ratings (30 artifacts have 2 ratings, 45 have 3). Share of fives 0.113. Split by task: train 10, validation 5, test 10 (30, 15, 30 artifacts).
   - New set (live collection, frozen 11-26-2026 15:47 PST in the sister repo): 100 artifacts, 25 tasks, 250 ratings (52 with 2, 46 with 3, 2 with 4). Share of fives 0.740. Split by task: train 10, validation 5, test 10 (40, 20, 40 artifacts).
   - Pooled: 175 artifacts, 50 tasks, 445 ratings. The 11-24 sealed files held 247 new ratings (442 pooled); three more ratings arrived by the 11-26 freeze. Use the 11-26 files and say which freeze you used.
   - Sealed splits: the 11-26 files carry `exp27_split` (identical to the sealed 11-24 split for all 50 tasks, checked by script) and `cv5_fold` (stratified 5-fold over tasks, seed 20261126, constant within a task).
   - Prior exposure (sister `PREREG.md`, quoted by meaning): the 10 old test tasks were scored by an earlier judge in sister Experiment 22 and all 20 test tasks were scored once in sister Experiment 27, so `exp27_split` is a benchmark on an opened test split, not a pristine population.
   - Already used here: Experiments 09, 10, 15, 16, 19, 20 and 21 on the old 75 artifacts (the ratings themselves are private and not committed here); Experiment 21 also ran judges on the 100 fresh artifacts. Experiment 37 (Cursor) pooled the native main and warm-up rows of the new collection (106 items) and Experiment 38 (Cursor) used the native main rows (100 items, 250 ratings), each with its own split.
   - Watch: only two to four ratings per artifact, so item histograms are not population distributions; use per-rating proper losses. Names are roster metadata, never data. The two Cursor experiments split the FSC items themselves and do not follow `exp27_split` (see the QA note in `qa_of_experiments_37_38_11-26-2026/`).

2. **SummEval (Fabbri et al. 2021, arXiv:2007.12626), public.** Summaries of CNN/Daily Mail articles, graded 1 to 5 on four aspects. K = 5. A prompt is one summary on one aspect.
   - 1,600 summaries, 100 articles, 16 systems; three expert records and five crowd records per summary; 6,400 summary-aspect rows. Split group: the article (all 16 systems of one article stay together).
   - Ceiling: 3 expert ratings (5 crowd) per summary.
   - Caveat verified in the paper text: experts re-checked scores that differed by more than 2 points and could see the other experts' first-round scores, so the three ratings are not independent draws (Krippendorff alpha 0.4132 after round one, 0.7127 after round two).
   - Used in Experiments 23, 36, 37 and 38.
   - Source file pinned in Experiment 36 `data_io.py`: `model_annotations.aligned.jsonl` from the SFR summarization repository.

3. **ChaosNLI (Nie et al. 2020, arXiv:2010.03532), public.** Natural-language-inference items, each with 100 crowd votes. Three-class sets: SNLI 1,514 and MNLI-matched 1,599 items (K = 3, 3,113 items in all). Binary set: alphaNLI 1,532 items (K = 2). All three files: 4,645 items. The three-class and binary sets are different tasks; do not add them.
   - Ceiling: 100 votes per item, the only set that can show a full sweep of k from 1 to 100 next to a stable reference distribution.
   - Split group: Experiment 29 groups SNLI and MNLI items by premise (713 test items fall in 659 premise groups); Experiments 37 and 38 group by item id, which can put two hypotheses of one premise in different splits. Use the premise. alphaNLI items are their own groups.
   - Used in Experiments 28, 29, 36, 37 and 38. Experiment 28 scored 600 SNLI, 600 MNLI-m and 400 alphaNLI items; Experiment 29 re-split them (1,200 / 600 / 600 / 713 items for the three-class set) and records which test items Experiment 28 had already scored.
   - Source: the `earino/chaosnli` Hugging Face mirror at revision `eb7cc18d`, hashes pinned in Experiment 36 `data_io.py` and matched by script here.

4. **DICES-350 and DICES-990 (Aroyo et al. 2023, arXiv:2306.11247), public.** Conversation-safety ratings, `Q_overall` with three answers (No, Unsure, Yes; K = 3).
   - DICES-350: 350 items, 123 raters each as released (104 after the README quality filter). DICES-990: 990 items, 69 to 76 raters each as released (60 to 69 after the filter), 64,886 ratings after the filter.
   - Ceiling: about 60 to 123 per item, enough for a sweep of k up to about 60.
   - Split group: the item. Raters are crossed with items (the same raters rate many items), so item-clustered intervals condition on the observed rater pool; say so in the caption.
   - Used in Experiments 28, 29, 36, 37 (DICES-990) and 38 (DICES-990).
   - Source: `google-research-datasets/dices-dataset`, `350/` and `990/` CSV files; hashes pinned and matched by script.

5. **POPQUORN (Pei and Jurgens 2023, arXiv:2306.06826), public, not yet run anywhere.** Crowd ratings of offensiveness and politeness on a 1 to 5 scale (K = 5).
   - Offensiveness: 1,500 items, 13,036 ratings, 262 annotators, 5 to 10 ratings per item (mean 8.69). Politeness: 3,718 items, 25,042 ratings, 506 annotators, 3 to 8 per item (mean 6.74). Both match Table 1 of the paper (checked by script).
   - Ceiling: 8 to 10 (offensiveness) and 8 (politeness) ratings per item, so a sweep of k past 8 to 10 is impossible; do not promise a longer curve.
   - Split group: the item; annotators are crossed with items, as for DICES.
   - Not run anywhere in this repository. Experiment 36 records it as "NOT LOADED" because `github.com/Jiaxin-Pei/POPQUORN` returned 404. The files load from `https://github.com/Jiaxin-Pei/Potato-Prolific-Dataset` at commit `dbb118c29b3d9bcce5aa172d8249ded9ece4df84`, folders `dataset/offensiveness/raw_data.csv` and `dataset/politeness_rating/raw_data.csv`. I did not check the data licence; check it before redistributing any file.
   - The paper states a rate of 12 US dollars per hour and 13 minutes per 50 offensiveness ratings, which gives about 0.052 US dollars per offensiveness rating by arithmetic; the paper gives no timing for politeness.

6. **HelpSteer2 (Wang et al. 2024, arXiv:2406.08673), public.** Helpfulness of assistant answers, integer 0 to 4 (K = 5). Individual annotations are released as lists.
   - Experiment 24 pins `nvidia/HelpSteer2` at commit `990b2711a36180dd19d9c94b8627844866f8982a` and holds out 519 source prompts (1,038 test items). Experiment 29 uses 1,600 / 800 / 800 / 1,038 items.
   - Ceiling: three to five annotators, collected adaptively (extra annotators when the first ones disagree), so the number of raters depends on how contentious an item is. Do not treat k as exogenous; an outcome-dependent stopping rule biases the per-sample risk.
   - The card describes the released label as a rounded mean of the three most-agreeing annotators; the 2-point figure is the agreement rule that triggers extra annotators, not an item filter.
   - Used in Experiments 24, 25 and 29.

7. **Other real sets already in the repository.**
   - KonIQ-10k image quality (Experiment 23): 1,078,154 votes on 10,073 photographs, authors' split 7,058 / 1,000 / 2,015 images, 93 to 157 votes per image.
   - University of North Texas homework grading (Experiments 21 and 26): two graders each scored 160 student answers to 67 questions; the README's RPS is taken over five grade thresholds. Ten folds, each holding out one whole homework assignment.
   - MovieLens and dialogue replications (Experiment 23): not audited here.

## Which of the Gemini plan's four experiments exist on real data

Read from `experiments/36_.../PLAN_STATUS.md` and `results.md`; I did not rerun them.

1. **Evaluation audit and rank reversals.** Real-data run exists in Experiment 36 on SummEval, ChaosNLI and DICES. Not on POPQUORN, HelpSteer2 or FSC.
2. **Objective by metric matrix.** Experiment 36 on the same three sets. Experiments 24, 25, 28 and 29 compare CE-trained and Pearson-trained judges with LoRA or frontier judges on HelpSteer2, ChaosNLI and DICES under their own protocols.
3. **k-crossover and iso-budget.** Experiment 36 on the same three sets; Experiments 23, 28 and 29 hold the budget sweeps for public sets. POPQUORN and FSC have no run, and their per-item ceilings (8 to 10 and 2 to 4) cap what a sweep can show.
4. **Single-pass logit readout against M-sample rollouts.** Experiment 36 records the rollout comparison as done on its three sets with linear heads; the GPU LoRA part is recorded as not run.

## Gaps against the conference standard

1. **POPQUORN has no run.** The source is now known. A run needs an item-level split, and annotators are crossed with items.
2. **FSC is under-powered and its sealed splits are not followed by Experiments 37 and 38.** Their FSC test sets hold 20 (Experiment 38) and 24 (Experiment 37) artifacts. On both, the unpaired 95% interval of the trained head's test cross-entropy overlaps widely with the interval of the train-marginal constant forecast (see the QA note for the numbers; a paired interval was not computed here), so neither run can show a trained head beating the constant on FSC.
3. **Prior exposure must be stated per set.** The FSC `exp27_split` test tasks have been scored before. Experiments 28 and 29 name which public test items were scored earlier. Any new run must carry the same disclosure.
4. **k is not a free knob on four of the sets.** FSC (2 to 4), SummEval (3), HelpSteer2 (3 to 5, adaptive), POPQUORN (3 to 10) and the homework data (2) cap the sweep. Only ChaosNLI (100), DICES (60 to 123) and KonIQ (93 to 157) support a long k axis.
5. **The five-dataset suite in the Gemini plan is not the same as the suite we have.** The plan names DICES with "60 to 112 raters per item", a three-class ChaosNLI of "4.6k items" and an FSC of "343 items, about 5 raters per item". The files give the counts above; the ledger in this folder records each difference.

## Where each input came from

1. SummEval: `https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl` (hash pinned in Experiment 36 `data_io.py`).
2. ChaosNLI: `https://huggingface.co/datasets/earino/chaosnli`, revision `eb7cc18dbfa0f161d93a2b22c7856ca0faafcf2b`, folder `raw/`.
3. DICES: `https://github.com/google-research-datasets/dices-dataset`, `350/diverse_safety_adversarial_dialog_350.csv` and `990/diverse_safety_adversarial_dialog_990.csv`.
4. POPQUORN: `https://github.com/Jiaxin-Pei/Potato-Prolific-Dataset`, commit `dbb118c29b3d9bcce5aa172d8249ded9ece4df84`.
5. HelpSteer2: `nvidia/HelpSteer2` dataset card, read for the annotator rules.
6. FSC: `trustworthy-theorem-judges` (sister repo), `experiments/28_multi_family_distributional_judge_and_open_weight_lora_on_new_ratings_start_26Nov2026/data_11-26-2026/` at commit `7855746` (anonymized, sorted ratings with rater identity removed) and `experiments/27_build_a_judge_that_beats_the_constant_forecast_on_held_out_tasks_old_new_ratings_start_24Nov2026/data_11-24-2026/task_splits_11-24-2026.csv` (sealed split).
7. The raw public files are not committed here (DICES alone is about 80 MB). `dataset_facts.py` prints each input's sha256 so a reader can re-fetch the same bytes and compare.

## What this file does not claim

1. It does not say any dataset is pristine or unseen; exposure is stated per set where it is known and "not audited" where it is not.
2. It does not measure costs. The 0.052 US dollars per offensiveness rating is arithmetic on two sentences of the POPQUORN paper, not an observed price.
3. It does not rerun any experiment. Which experiment used which set is read from the experiment READMEs and Experiment 36's status file.
4. The FSC old-set ratings are private and not in this repository. The counts above for the old set come from the anonymized sister files only.
