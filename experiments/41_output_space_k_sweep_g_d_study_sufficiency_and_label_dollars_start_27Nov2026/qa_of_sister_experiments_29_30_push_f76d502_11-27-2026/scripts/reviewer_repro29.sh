#!/bin/bash
cd /tmp/qa2930/repo/experiments/29_refit_cumulative_logit_heads_ce_versus_rps_on_445_ratings_start_27Nov2026
export PYTHONDONTWRITEBYTECODE=1
for cmd in "python3 -B scripts/build_snapshot.py" "python3 -B -m unittest tests/test_heads_and_join.py" "python3 -B scripts/run_experiment.py --stage select" "python3 -B scripts/run_experiment.py --stage eval" "python3 -B scripts/run_experiment.py --stage nested"; do
  echo "=== $(date -u +%T) RUN: $cmd"
  $cmd > /tmp/qa2930/log29_$(echo $cmd | tr ' /-' '___' | cut -c1-60).txt 2>&1
  echo "=== $(date -u +%T) EXIT $? : $cmd"
done
echo "=== $(date -u +%T) DEVIATION: restore committed selection JSON and rerun eval"
cp results/validation_selection_11-27-2026.json /tmp/qa2930/regenerated_selection_29.json
cp /tmp/qa2930/committed/e29/results/validation_selection_11-27-2026.json results/validation_selection_11-27-2026.json
python3 -B scripts/run_experiment.py --stage eval > /tmp/qa2930/log29_eval_with_committed_selection.txt 2>&1
echo "=== $(date -u +%T) EXIT $? : eval with committed selection"
