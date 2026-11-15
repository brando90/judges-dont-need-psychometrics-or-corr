# Stage 3 brief: qa3_carbon

**TLDR:** You are the final, fresh reviewer. Read COMMON.md, inspect independently first, then reconcile every earlier finding and fix; verify everything end to end and write `qa3_carbon.md` plus the marker `qa3_carbon.json` with an accept or reject verdict.

- Start by stating your actual model and effort. You have no memory of stage 1; treat its report as another reviewer's.
- Read `COMMON.md` (binding) and inspect `git diff d9d1c1d` (all changes since the base) independently before reading `qa1_carbon.md`, `qa2_codex_vals.md` and their markers (`qa1_carbon.json`, `qa2_codex_vals.json` in the runtime directory).
- `git log --oneline d9d1c1d..HEAD` shows the builder commit and one commit per earlier stage.
- After stage 2 the builder made one more commit (`git show HEAD`): freeze amendment 3 (secondary three-expert analysis, 10,000 all-75 draws, rerun of `run_historical.py`), corrected report-generator wording, and the reading guide's limits section. Review it as strictly as the reviewers' changes, including that primary test-split outputs are unchanged.
- Reconcile every earlier finding (resolved, still open, or wrong), check every fix, and run every verification command in COMMON.md plus the paper build.
- Your verdict decides landing: accept only if no critical or major issue remains open.
