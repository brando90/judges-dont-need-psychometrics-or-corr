# Mega QA 11-20-2026 (cross-check of the corr vs CE brief, Version 27, and experiments 00 to 16)

**TLDR:** Three independent audits of the one-pager and the experiment reports, run at the highest reasoning setting. Every claim below was checked against the repo files; what was right is fixed in the files (dated correction lines), what is open is listed here.

- Model and harness: GPT-6 Astra, reasoning effort high, OpenAI Responses API, Koyejo lab OpenAI key (read at call time, never stored here). Three jobs (bundles A, B, C), about $4.40 in total.
- Why not Gemini 4 Carbon (as asked for 11-20): no Gemini key of Brando's or the lab's is stored on the machine, and the Google agent route is on hold, so the strongest other cross-check reachable was used.
- Prompt: `PROMPT.txt` (first person, asks for mismatches, overclaims, done vs not done, coverage, top 5 next steps). Raw outputs: `qa_bundle_A.md`, `qa_bundle_B.md`, `qa_bundle_C.md`. Each auditor saw only the text of the bundle it was given, not the code.
- Extends the 11-16 QA in `../mega_qa_11-16-2026/`.

## Reconciliation (claim, verdict against the files, what changed)
| Auditor claim | Verdict | Action |
|---|---|---|
| Experiment 16: "best arm on every pool", "as bad as CE or correlation" | Right: the constant beats every arm on the open-weight pool; RPS-picking is clearly worse than CE-picking there | Wording corrected in `experiments/16_*/results.md`, correction line added |
| Experiment 15: "changes no judge ranking", "severity is not what the ratings measure" | Right: the best judge and RPS ranking are unchanged, but the Spearman and Pearson rankings have Kendall tau 0.996, and only constant offsets were estimated | Corrected in `experiments/15_*/README.md` |
| Experiment 11: "eliminates" the small-n spike; "CE" columns are KL | Right | TLDR reworded to "reduces"; KL note added |
| Index: experiment 07 "crossovers unchanged"; 01 and 02 "paper integration remains open" | Right (integration is in commits 3bfd866 and 18f529e) | `experiments/README.md` fixed |
| Experiment 00 README heading says 01 | Right | Fixed |
| Version 27: "1D bottleneck" | Overclaim: a nonlinear scalar calibrator was untested | Experiment 17 tested it (done 11-20-2026): one number loses nothing in the toys; the gap is score quality |
| Version 27: rater severity "cannot be separated" | Contradicted by experiment 15 (constant offsets are estimable and small) | Open in the paper text; the status file says so |
| Version 27: "small-sample gains on human ratings remain untested" | Stale: experiment 10's learning curve is retrospective human small-n | Open in the paper text |
| Experiment 09 TLDR-end: blanket "native probabilities beat the constant"; digit-logprob pool is worse than the constant on CE | Right | Open: wording not yet changed |
| Experiment 09 "AST scaffold" is a regex summary; experiment 10 key names `H2_ast_gpt61__*` ran `gpt-5.4-mini` | Right, already disclosed in the reports | Open: label change needs a rename of result keys |
| Experiment 10 grid: registered five-fold five-value, executed three folds and c in {0.1, 1, 10} | Right, a disclosed deviation | Open |
| Human-ceiling and theorem-count numbers differ between experiments 09 and 10 | Different definitions (leave-one-rater-out loss vs estimated floor; count vs ratio) | Open: definitions to be put side by side |
| Experiment 05 forecast with zero mass on ratings 1 to 4 gives finite CE | Not confirmed or refuted in this pass | Open |
