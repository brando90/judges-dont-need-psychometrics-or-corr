# Stage 3 brief: qa3_codex

**TLDR:** You are the final, clean-eyes reviewer in a fresh context. Read `COMMON.md` (binding), review the current state independently, then check every earlier finding, fix and builder decision; fix what you can prove wrong, verify, and write `qa3_codex.md` plus the marker `qa3_codex.json`.

- Start by stating your actual model, reasoning effort and account profile (`CODEX_HOME`).
- Review the current checkout first, without reading earlier reports: `git diff af810f5..HEAD -- <scope paths>` and the files themselves, plus the rendered PDF, figures and one-page summary.
- Then read `qa1_codex.md`, `BUILDER_RECONCILIATION_qa1.md`, `qa2_carbon.md` and `BUILDER_RECONCILIATION_qa2.md` in this folder, and the markers in `/Users/brandomiranda/.local/state/judges/mega-qa-1116/`. Commits after `60f71a5` hold each stage's fixes and the builder's follow-ups (`git log --oneline 233d504..HEAD`). For every material finding, confirm it is fixed correctly, disagree with evidence, or extend it. A later PASS cannot erase an earlier unresolved issue.
- Weight your effort toward what ships soonest: the paper (due 11-20-2026 06:00 UTC), then the one-page summary, then the experiment records.
- The post-hoc check `experiments/08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/scripts/check_ce_gradient.py` (output `results/ce_gradient_check.json`, rerun with `python3 <script> 8`, about 3 minutes) supports the claim that the frozen CE gradient changes no reported number; check that claim against the code and output, not against the reports' prose (stage 2's report describes the script's internals inaccurately).
- The paper body ends exactly at the bottom of page 4 (Conclusion on page 4, References from page 5). Any sentence you add to the body must be offset by a cut; after rebuilding, confirm the Conclusion still ends on page 4.
- You have full write access to the owned checkout. Make fixes directly, run every verification command in `COMMON.md`, and leave your changes uncommitted for the builder.
