# Mega QA of the learning study, experiments 06–08 and the paper: checkpoint

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/paper_latex_and_notes/notes/mega_qa_11-16-2026/CKPT_mega_qa_11-16-2026.md>

**TLDR:** Brando requested a Mega QA on 11-16-2026 ("mega qa, go") of this session's work (paper learning study, RPS appendix and edits; experiments 06–08; experiment 13 plot script; one-page summary; result-clarity rules). Three sequential stages run in an owned detached worktree; all three stages finished and the reviewed work landed on main on 11-16-2026.

Created: 11-16-2026 10:08 PST
Last updated: 11-16-2026 12:50 PST

## State
- Coordinator/builder: Claude Code `claude-opus-5-5` (desktop app session d45eb9ca). Notifier for RUN START/DONE: this coordinator only.
- Worktree: `/Users/brandomiranda/.worktrees/judges-dont-need-psychometrics-or-corr/mega-qa-1116` (detached at `233d504` = origin/main 11-16-2026; local brief commit `60f71a5`). Review base `af810f5`.
- Runtime: `/Users/brandomiranda/.local/state/judges/mega-qa-1116/` (launch script, logs, markers, this checkpoint until landing).
- Stage 1 `qa1_codex`: COMPLETE 10:38 (32 min) on the Vals Codex profile after the personal account hit its usage limit (resets 11-17-2026 4:11 PM). Verdict FAIL with 1 open major (QA1-F1, frozen CE gradient) and 1 open minor (QA1-F11, appendix page limit); 10 fixes. Builder resolved QA1-F1 with `check_ce_gradient.py` (no reported number changes) and fixed B1-B5; local commit `beed30a`; reconciliation in `BUILDER_RECONCILIATION_qa1.md`.
- Stage 2: planned `qa2_claude` (claude-fable-5-1/max) failed twice at launch: personal Claude account Fable limit (resets 11:00 PM PST), Stanford Claude login expired (needs Brando's sign-in). Third and last invocation: `qa2_carbon`, Antigravity `gemini-4-carbon-high`/high, personal Google AI Ultra account, COMPLETE 11:40 (39 min): verdict FIXED, 1 major (clipped figure footer) and 4 minor fixed, 1 minor open (submission packaging, incl. QA1-F11). Builder verified fixes, corrected the report's inaccurate description of the check script; local commit `a3efff2`; reconciliation in `BUILDER_RECONCILIATION_qa2.md`.
- Stage 3 `qa3_codex`: COMPLETE 12:20 (37 min), fresh `gpt-6-astra`/`ultra`, Vals profile, session `01a1e5e2-c280-7842-a0ee-5ecaae9b0e71`. Verdict FIXED: 1 major (overstated CE-check claims) and 5 minor fixed; open: QA3-F7 (appendix page limit, for Brando) and QA3-F8 (acronyms), which the builder then fixed within the four-page body. Reconciliation in `BUILDER_RECONCILIATION_qa3.md`.
- Landed: `233d504..adf7ec9` pushed to main 11-16-2026 12:40; one-page summary republished as Version 24 (https://claude.ai/artifact/Rg2hYLxYMGJAte7jcV7kKV).
- Emails: RUN START sent 10:08 (Gmail id 1a1e58aa69f54342). RUN DONE pending.

## Next steps
- Brando: decide whether references and appendices may follow the 4-page position-paper body (QA1-F11 / QA2-F6 / QA3-F7) before 11-20-2026 06:00 UTC; the reliability appendix (another session's) has no pointer from the body.
- Stanford Claude login expired (re-login needed for Fable on that account); personal Claude Fable resets 11-16-2026 11:00 PM PST; personal Codex resets 11-17-2026 4:11 PM PST.

## Log
11-16-2026 10:06 PST -- Briefs written and committed locally (60f71a5); stage 1 attempt 1 (personal Codex) hit usage limit; attempt 2 launched on Vals profile.
11-16-2026 10:08 PST -- RUN START email sent.
11-16-2026 10:38 PST -- Stage 1 done; builder ran the CE impact check (161 s), reconciled, committed beed30a.
11-16-2026 10:56 PST -- Stage 2 Claude attempts failed (Fable limit; Stanford login expired); 10:59 stage 2 launched on Gemini 4 Carbon High.
11-16-2026 11:40 PST -- Stage 2 (Carbon) done; builder verified, committed a3efff2; 11:43 stage 3 launched on Vals Codex.
11-16-2026 12:20 PST -- Stage 3 done; builder fixed QA3-F8 acronyms within four body pages, final checks passed, committed adf7ec9.
11-16-2026 12:40 PST -- Pushed 233d504..adf7ec9 to main; one-page summary Version 24 published.
