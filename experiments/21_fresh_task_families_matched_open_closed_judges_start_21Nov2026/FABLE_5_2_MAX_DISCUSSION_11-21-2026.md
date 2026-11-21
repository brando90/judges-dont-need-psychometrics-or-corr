# Claude Fable 5.2 Max discussion on the Codex question and the fresh-data design | 11-21-2026

As of 11-21-2026 (about 14:30 to 14:45 PST). Brando asked his assistant (ub9sc, "ultimate brando9 scalable coworker") to put Codex's open question to Claude Fable 5.2 at Max effort and to discuss what is best for the main goal of the Corr vs CE paper. Codex had already consulted Fable once ([first consultation](DESIGN_CONSULTATION_11-21-2026.md)); this is an independent second discussion with the written protocol in view.

**Harness and model.** Claude Code 2.1.318 (first-party client), run on skampere1 under Brando's personal Claude plan, extra usage disabled (no paid overflow), no tools, no files read. Model `claude-fable-5-2` (reported by the client's model usage and by the model itself), effort `max`. Two rounds, the second a resume of the first session. Round 1 took 704 s, round 2 took 145 s. No judge data was changed and no paid call was made.

**Codex's question.** "I'm checking for fresh VeriBench ratings. I also found computer-science short answers with two independent human grades on the same 0-5 scale. Which result would you like if the new VeriBench forms are still empty?"

## Round 1: brief sent
You are advising Brando Miranda (Stanford PhD, defense Dec 10) on one methodological choice in his paper, via his assistant ub9sc. Be direct and concrete. Plain words. No tools needed: answer from this brief.

MAIN GOAL OF THE PAPER (Brando's words, paraphrased): show honestly WHEN a judge trained with a proper loss (cross-entropy, CE) beats a judge trained to maximize correlation (Pearson), including with limited training data, and report where correlation wins. Not a sales pitch: cases where Pearson wins stay in.

WHAT IS ALREADY ESTABLISHED (experiments 18-20, same repo):
- Theory: Pearson sees only the first moment (mean rating per item) up to level and scale; any judge with means a + b*(human means), b>0, scores Pearson=1 whatever its spread/shape. CE and RPS are proper, so only minimized by the full human distribution (fixes all moments). This is an infinite-data result; the small-data claim is NOT derived, so it is tested.
- Synthetic (known true distributions, 200 seeds): CE beats the Pearson-trained judge on distribution score from 16 items x 3 ratings (+0.034, better in 96% of seeds) and from 32 items x 1 rating; at 4 items CE loses, at 8 it ties.
- Real VeriBench-type data (75 artifacts, 59-judge pools): CE head beats Pearson head by +0.027 to +0.058 normalized RPS from 7 training families up; ties at 3 families; loses in the open-weight 39 pool (-0.037). Selecting one judge by Pearson beats selecting by CE in the 44 historical 3-vote judges.
- All of that is on the same family of VeriBench-style tasks. The new request is a FRESH test on task families never used before, same prompt for an open-weight and a closed-weight judge, varying the number of training ratings, asking: at which budget does CE beat BOTH Pearson AND uniform on held-out RPS? Also report test CE and TVD.

CODEX'S QUESTION (Brando's local Codex agent, pasted to Brando, who does not understand it): "I'm checking for fresh VeriBench ratings. I also found computer-science short answers with two independent human grades on the same 0-5 scale. Which result would you like if the new VeriBench forms are still empty?"

WHAT CODEX HAS SINCE DONE (in the repo, experiment 21): it already consulted Fable 5.2 Max once and wrote a protocol (summary):
- Dataset: UNT intro-CS short answers (Mohler et al. 2011), 10 homework assignments, 16 answers each = 160 answers, two graders per answer, integer 0-5. Ceiling-skewed grades. Public corpus, so pretraining exposure is possible.
- Judges: Qwen3.5-9B (open, local) and Claude Sonnet 5 (closed, native subscription), one forecast per item, same prompt, six-probability output. No new paid calls.
- Leave-one-homework-out cross-fitting (10 folds), 200 seeded nested training draws per fold, budgets 4..256 training ratings (two-rating primary arm; one-rating secondary), 27-transformation calibration grid, CE-fit vs Pearson-fit vs uniform vs original judge vs training histogram. Metrics: normalized RPS (primary), test CE, TVD. 
- Inference: homework is the unit; one-sided t(9) with Nadeau-Bengio correction, alpha 0.025 per judge; budgets tested 256 down to 4, stop at first failure; both Pearson and uniform differences must be > 0; sign-flip permutation as sensitivity.
- VeriBench: new forms are empty, so only hashed forecasts for the prospective Set N (100 items) are frozen now; no fresh VeriBench claim is possible yet. Confirmation on Set N when ratings arrive.

WHAT BRANDO NEEDS FROM YOU (answer in this order):
1. One recommendation, 3-5 sentences, for what Brando should tell Codex, serving the paper's main goal. Say explicitly whether the design above is right or what you would change.
2. Which data/protocol choices best serve the paper: (a) UNT as a separately labelled second dataset with the protocol written first, vs substituting it for VeriBench; (b) keeping VeriBench primary when its ratings arrive; (c) how to treat two ratings per item; (d) what must be reported (including where Pearson wins and where CE fails at small budgets); (e) any claim the paper must NOT make from this data (domain mismatch: correctness grading vs theorem-translation judging).
3. The 3 biggest risks to honesty or validity of this plan, and one cheap fix each.
4. A paste-back text for Codex of at most 6 lines.
Keep the whole answer under 700 words. Do not invent numbers; say "unknown" where you do not know.

## Round 1: Fable 5.2 Max answer (verbatim)
**1. Recommendation**

Tell Codex: run the UNT short-answer test now as a clearly labelled second dataset in a different domain, and keep VeriBench Set N as the primary fresh test when its ratings arrive. The structure is right: whole homeworks held out, nested budgets, one prompt for both judges, and CE-fit compared with Pearson-fit, uniform, original judge and training histogram. Change four things:
- Hash the protocol and analysis code before any held-out score is computed.
- Fix how the Pearson arm breaks ties.
- Report every budget in both directions, not only up to the first failure.
- Treat total variation distance (TVD) as descriptive only.

**2. Choices**

- **(a) Second dataset, not a substitute.** UNT tests whether the pattern holds on different human ratings, not on new VeriBench families. Give it its own section and table, never pooled with VeriBench.
- **(b) VeriBench stays primary.** Freeze the Set N analysis script beside the hashed forecasts now, so UNT results cannot tune it. Log later changes as deviations.
- **(c) Two ratings.**
  - Score each forecast against each grade separately. For CE and ranked probability score (RPS) that is proper, and raw method differences equal those from the two-grade histogram.
  - TVD against a two-grade histogram is not proper; it rewards overconfident forecasts.
  - In the one-rating arm, pick the grader at random by seed. That arm tops out at 144 ratings per fold (9 homeworks x 16 answers), so 256 exists only in the two-rating arm.
  - Report the mean gap between graders. If it is systematic, the target is "these two graders", not "humans".
- **(d) Must report.**
  - Every budget from 4 to 256, both judges, both arms: CE-fit minus Pearson-fit, uniform, training histogram and original judge, with intervals.
  - The number of homeworks out of 10 where CE wins.
  - Held-out Pearson for every arm, so correlation's wins on its own metric show.
  - Grades are ceiling-skewed, so uniform is an easy baseline; the training histogram is the hard one.
- **(e) Must not claim.** A fresh VeriBench result; that the UNT threshold carries over to theorem-translation judging; a universal budget number; that the judges never saw this data.

**3. Risks and cheap fixes**

1. **Strawman Pearson arm.** Pearson cannot separate grid members that differ mainly in spread or shape, so its pick among the 27 may be arbitrary. Fix: write the tie-break rule first, and report tie frequency plus the best and worst RPS among Pearson-tied members.
2. **Threshold set by low power.** Ten homeworks of 16 answers is small, and a one-sided test that stops at the first failure can never show a Pearson win. Fix: call it "smallest budget where the win was confirmed" and print two-sided intervals at every budget.
3. **Dataset chosen after VeriBench came up empty, and public.** Fix: commit the protocol hash before scoring, state whether any held-out score was already seen (unknown from this brief), and label it "public corpus, possible pretraining exposure". Both heads use the same forecasts, so exposure affects both arms.

**4. Paste-back for Codex**

```
1. Run UNT now as a separately labelled second-domain dataset; do not substitute it for VeriBench. Set N stays the primary fresh test.
2. Before any held-out score: commit hashes of the protocol, analysis code and Set N analysis script. Say whether any score was already seen.
3. Pearson arm: pre-specify the tie-break; report tie frequency and best/worst RPS among Pearson-tied grid members.
4. Report all budgets 4-256, both judges, both arms, two-sided intervals, past the stopping point: CE-fit vs Pearson-fit, uniform, training histogram, original judge; plus held-out Pearson per arm.
5. Score per individual grade (CE, RPS); TVD descriptive only. One-rating arm: random grader by seed, max 144 ratings per fold. Report the grader gap.
6. No fresh-VeriBench claim, no transfer of the UNT threshold to theorem-translation judging; note possible pretraining exposure.
```

**TLDR-end:** [CE vs Pearson paper: experiment 21 design] Run UNT now as a labelled second-domain test under a pre-hashed protocol, keep VeriBench Set N primary, and report every budget in both directions, including Pearson wins.

**Snapshot:** none; advice from the brief only, no files read or run.

## Round 2: message sent
Thanks. Round 2, short. Two things.

A. Your four changes against the written protocol (exact text, not my summary):
- Tie rule IS already fixed: "Use the grid in order of squared parameter distance to identity, then lexicographic order, with ties within 1e-12; a wholly undefined Pearson fit returns identity." Reported already: "selection counts and failures".
- Scoring per individual grade IS already specified: "Compute the two proper scores separately for both raters, averaging within each answer." TVD is vs the empirical two-grader histogram and is "descriptive".
- Stopping: "Stop formal claims for a judge at its first failure; report all point estimates and local results beyond that budget descriptively." Test is one-sided 97.5% lower bound, t(9) with Nadeau-Bengio factor sqrt(1/10+1/9).
- Hashing: prompts (SHA-256 of the 160 label-free requests) were hashed and the design was fixed before any judge scored a study prompt or any score was computed. The analysis code is not yet hashed.
Given this, which of your four changes still stand, and what is the single most valuable remaining change?

B. New instruction from Brando at 2:23 PM, which changes the context: he wants real, public human-rating data used in practice (not only synthetic or VeriBench), a deep search of open benchmarks with human judgments, to see whether our findings hold, and "if N is large enough for CE at the practical cost of getting ratings, we shouldn't use Corr". He also says evaluating with correlation alone is circular ("we are making the eval show us corr is better by using corr itself") and wants the evaluation protocol that makes most sense, using all the metrics we discussed (Sanmi's comments included). Two coworkers (Codex and me) both receive this, so they must be coordinated.
Answer in under 300 words: (1) is UNT a good first public dataset or should other public human-rating sets be prioritized first (name candidate TYPES, not invented dataset names; say if unsure a dataset exists); (2) the evaluation protocol you would use so it is not circular: which metric is primary, which are diagnostics, and why Pearson-on-the-eval cannot be the headline; (3) one line on how Codex and the other coworker should split work to avoid overlap.

## Round 2: Fable 5.2 Max answer (verbatim)
**A. What still stands**

- **Withdrawn, already covered:** the tie rule, scoring per individual grade, and TVD as descriptive. Defaulting to identity is fair: when Pearson is indifferent, the Pearson arm keeps the original judge. The best/worst band is now optional.
- **Stands in part (hashing):** hash the analysis code before the first score is computed. If scores already exist, hash now and state that order.
- **Stands in part (direction):** all budgets are reported, but a one-sided lower bound can confirm a CE win and can never show a Pearson win.

**Single most valuable change:** add the matching upper bound at every budget (same t(9), same factor), with the reading written first. Upper bound below zero is reported as "Pearson (or uniform) ahead", descriptive. An interval covering zero is "not resolved", not "tie". Put that in, then hash the code.

**B.**

**(1) UNT first?** Yes as the first run: it is frozen, free, and automated graders on it are usually scored by correlation (my recollection; verify). Its limit is two graders per answer, which barely shows a rating distribution. Next, prioritize sets that release each rater's score, with about five or more ratings per item:
- Text-generation quality ratings (summarization, dialogue), Likert scale. These exist; check per-rater release.
- Machine-translation human assessments. These exist; continuous scale, raters per item unknown.
- Many-annotator label sets (inference, toxicity). These exist; categorical, so no RPS.
- Peer-review scores, several reviewers per paper. These exist.
- Human ratings of generated code, closest to VeriBench. Unsure a multi-rater public set exists.

**(2) Protocol.** Each fitting method wins on its own metric, so neither CE nor Pearson can referee. Pearson also ignores level, spread and shape, the exact errors in dispute, so it cannot be the headline.
- **Primary:** normalized RPS on held-out groups, scored against individual held-out ratings. It is proper, respects order, and neither arm trained on it. For categorical sets, use the Brier score.
- **Diagnostics, labelled with the arm each one helps:** test CE (CE arm); Pearson, Spearman and Kendall of means (Pearson arm); TVD; calibration.
- **Practical check:** system ranking agreement and error in each system's mean score, where correlation may fairly win.
- **References:** uniform, training histogram, original judge, and a human ceiling from half the raters predicting the other half.
- **Budget rule:** per dataset, the smallest number of ratings where CE is confirmed. Rating cost is unknown.
- Sanmi's comments are not in this brief, so they are not reflected here.

**(3) Split.** You own the dataset search and deliver one ranked shortlist file; Codex alone writes protocol and analysis code, keeps UNT's frozen, finishes UNT and Set N, then implements the shortlist.

**TLDR-end:** [CE vs Pearson paper: round 2 protocol check] Two of my four changes were already in the protocol; the one that still matters is adding the upper bound so a Pearson win can show, then hashing the analysis code. For public data, RPS stays the headline because neither arm trained on it.

**Snapshot:** none; advice from the brief only, no files read or run.

## Notes added by ub9sc after reading the answers
- Fable saw only the briefs. Its claim that automated graders on UNT are usually scored by correlation is its own recollection and is unverified.
- The tie rule, per-grade scoring and TVD-as-descriptive were already in the protocol, so Fable withdrew those three changes in round 2.
- By 14:44 Codex had finished the UNT run ([results](short_answers/RESULTS_11-21-2026.md)); the remaining actions from this discussion apply to how it is reported and to the next datasets.
