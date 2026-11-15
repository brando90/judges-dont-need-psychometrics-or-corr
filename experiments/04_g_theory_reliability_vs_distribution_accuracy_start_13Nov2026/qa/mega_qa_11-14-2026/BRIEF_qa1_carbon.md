# Stage 1 brief: qa1_carbon

**TLDR:** You are the first, clean-eyes reviewer. Read COMMON.md in full, inspect everything in scope independently, fix what you can prove wrong, verify, and write `qa1_carbon.md` plus the marker `qa1_carbon.json`.

- Start by stating your actual model and effort.
- Read `COMMON.md` (binding), then the builder's report `RESULTS_reliability_vs_accuracy.md`, `PLAN.md`, `protocol.json`, every script, and the appendix `.tex`. Read the source paper sections that define variance components and coefficients (printed pages 924–926).
- Recompute at least: one A1 artifact by hand (moments, RPS excess = (6/5)ε²), the A4 table entries for 8×2 and 2×8, and the Phase B population error for one design; check them against the code and outputs.
- Then run all verification commands in COMMON.md, review claims and clarity, and fix.
- No earlier stage exists; do not wait for one.
