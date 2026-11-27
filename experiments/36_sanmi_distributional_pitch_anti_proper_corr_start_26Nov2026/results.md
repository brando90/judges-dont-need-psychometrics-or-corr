# Results ledger

**Status:** COMPLETE for SummEval, ChaosNLI, DICES, POPQUORN, VeriBench FSC (`R_v1`, `native_main`, pooled), and CertJudge real panels (11-26-2026).

**Last updated:** 11-26-2026.

| Phase | State |
| --- | --- |
| Numeric proof checks | DONE |
| Lean compile | DONE (`lean_check.txt`, 0 `sorry`) |
| Real-data matrix, crossover, iso-budget, rollout | DONE on SummEval, ChaosNLI, DICES, POPQUORN, VeriBench FSC |
| CertJudge real LLM judge bridge (`E63`, `E39`, `Exp 67`) | DONE (`$0` extra spend) |
| Test-set power & MDD | DONE on DICES-990, POPQUORN Offensiveness, POPQUORN Politeness |
| GPU LoRA & LLM CoT logit readout (`scripts/snap_*.py`) | NOT RUN (no GPU on this machine) |

Runtime seconds: 166.2868411540985.

**TLDR-end:** [judges-corr-vs-loss: sanmi distributional pitch] Correlation is affine-invariant and anti-proper for an independently sampled judge; the held-out human tables across SummEval, ChaosNLI, DICES, POPQUORN, VeriBench FSC, and CertJudge real LLM panels are in `RESULTS_SUMMARY.md`.
