# Experiment 22: non-circular evaluation protocol for CE-trained vs correlation-trained judges (started 11-21-2026)

**Status as of 11-21-2026:** protocol written and frozen before any new data were looked at; nothing run except a 40-line numeric illustration of the algebra. **Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/22_non_circular_evaluation_protocol_and_public_rating_sets_start_21Nov2026/README.md>

**TLDR:** Scoring a Pearson-trained judge with Pearson correlation is circular: the metric is the training objective, and it sees only per-item mean ratings up to a shift and a positive rescale (R1, R2). The protocol scores with held-out ranked probability score (RPS) first, then cross-entropy and total variation distance, adds metrics neither objective optimises (top-1 agreement, calibration, pairwise accuracy), maps Sanmi's feedback one by one, fixes baselines, training sizes and intervals, defines the break-even number of human ratings N\* against the cost of collecting them, and gives a data checklist plus a shortlist of public rating sets (ChaosNLI, DICES, HelpSteer2).

## Files
- [EVAL_PROTOCOL_11-21-2026.pdf](EVAL_PROTOCOL_11-21-2026.pdf): typeset note. Sources: [.md](EVAL_PROTOCOL_11-21-2026.md), [.tex](EVAL_PROTOCOL_11-21-2026.tex).
- `code/toy_circular.py`, `code/toy_circular_output.txt`: the five-item numeric illustration in Section 2 (exact expectations; run `python3 toy_circular.py`, needs numpy and scipy).

Related: experiment 18 (maths), 20 (limited data), 21 (Codex's frozen fresh-family protocol, not changed here). The public-set literature survey and the runs are separate and not started.
