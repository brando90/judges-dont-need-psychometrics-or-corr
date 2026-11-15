# Stage 2 brief: qa2_codex_vals

**TLDR:** You are the second reviewer, on a different model family. Read COMMON.md, review independently first, then reconcile stage 1's findings and fixes; fix, verify, and write `qa2_codex_vals.md` plus the marker `qa2_codex_vals.json`.

- Start by stating your actual model, effort and account profile.
- Read `COMMON.md` (binding), then inspect independently (`git diff d9d1c1d..03ada1e` is the builder's appendix change; the experiment 04 files are on the base) before reading stage 1's report `qa1_carbon.md` and its marker `qa1_carbon.json`.
- Stage 1's fixes are the commit after `03ada1e` (`git log --oneline d9d1c1d..HEAD`, then `git show HEAD`). Check every stage 1 finding: agree, disagree with evidence, or extend; confirm its fixes are correct and broke nothing. Known nit to check: the appendix sentence "correlated raters understated deployment error" (it is the naive prediction that understated the error).
- You have full write access. Make fixes directly, run every verification command, and leave your changes uncommitted for the builder.
