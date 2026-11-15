# Mega QA of experiment 04 and its reliability appendix: checkpoint

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/qa/mega_qa_11-14-2026/CKPT_mega_qa_11-14-2026.md>

**TLDR:** Brando requested a Mega QA (Carbon high in Antigravity, Codex on the Stanford account) of experiment 04 and the new paper appendix on 11-14-2026. Three sequential stages run in an owned detached worktree; nothing lands on main until stage 3 accepts.

Created: 11-14-2026 19:24 PST
Last updated: 11-14-2026 20:58 PST

## State
- Coordinator/builder: Claude Code `claude-opus-5-5` (desktop app session). Notifier for RUN START/DONE: this coordinator only.
- Worktree: `/Users/brandomiranda/.worktrees/judges-dont-need-psychometrics-or-corr/mega-qa-04-1114` (detached; base `d9d1c1d`; builder commit `03ada1e` adds `9-appendix-reliability.tex`, bib entry, 4-proposal ref fix, PDF).
- Runtime: `/Users/brandomiranda/.local/state/judges/mega-qa-04-1114/` (briefs, launch script, native and terminal logs, markers).
- Stage 1 `qa1_carbon`: COMPLETE 19:56 (35 min). Verdict accept with minor open issues; 0 critical, 1 major fixed (appendix/report overstated D-study selection failures), 4 minor fixed, 3 nits (2 open, out of scope). Changes committed locally as `9e4b243`; tmux session closed.
- Stage 2 `qa2_codex_vals`: COMPLETE 20:17 (19 min). Accept with minor open issues; 2 major fixed (every-moment and every-proper-loss wording; same-split pair selection made descriptive), 5 minor, 1 nit. Committed `a283556`. Builder then closed QA2-F7 (freeze amendment 3: secondary three-expert analysis, 10,000 all-75 draws; primary unchanged) and fixed report-generator wording: `be39e2a`; verification 140/140.
- Stage 3 `qa3_carbon`: COMPLETE 20:52 (29 min). Verdict **accept**; 0 critical, 0 major, 2 minor and 1 nit, all fixed. Builder verification passed (hashes, 140/140, byte-exact report, clean 6-page build); reviewer-report slips recorded in BUILDER_NOTES.md.
- Emails: RUN START sent 19:23 (Gmail id 1a1dd3a495c3f302). RUN DONE pending.
## Next steps
- On `qa1_carbon.json`: read report, check diff, close tmux session, launch stage 2.
- After stage 3 accepts: rebuild/verify, rebase onto origin/main, push, update Google Doc if numbers changed, RUN DONE email, remove worktree (Rule 19).
- Before landing, add the task origin to the experiment 04 README: Brando's Vals AI weekly update, 11-13-2026 entry "Q: is g-theory, d-theory (from psychometrics I think?) needed at all when we have CE learning (just standard stats/ml/info learning)? I conjecture we don't, CE learns the true distribution eq all moments" (https://docs.google.com/document/d/1xkEn8vW800jBucuJANuXD8ph-sMQMofwNZw_nJzjJls/edit?tab=t.0).

## Log
11-14-2026 19:24 PST -- Built appendix in worktree (clean pdfLaTeX build, 6 pages, no section refs), launched stage 1 Carbon, opened Stanford Codex login, sent RUN START.
11-14-2026 19:31 PST -- Stage 2 reviewer switched to Codex Vals (gpt-6-astra/ultra) after Stanford login failure and personal usage limit; Stanford login tab closed.
11-14-2026 19:40 PST -- Brando identified the originating weekly-update question; answered in chat; weekly doc not edited (Rule 56: only on request).
11-14-2026 20:01 PST -- Stage 1 done (accept with minor; major F1 fixed); briefs renamed BRIEF_<stage>.md after case-insensitive collision; stage 2 Codex Vals launched; Google Doc section and results email repeat the F1 overstatement, correct after QA.
11-14-2026 20:24 PST -- Stage 2 done; builder amendment 3 and report fixes committed (be39e2a); stage 3 fresh Carbon launched.
11-14-2026 20:58 PST -- Stage 3 accepted; builder checks passed; README origin note and BUILDER_NOTES added; landing next.
