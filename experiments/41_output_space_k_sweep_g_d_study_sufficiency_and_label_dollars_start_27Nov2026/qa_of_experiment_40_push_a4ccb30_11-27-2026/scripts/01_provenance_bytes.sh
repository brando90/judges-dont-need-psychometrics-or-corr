#!/usr/bin/env bash
# Compare the six files under provenance/ttj_efdc8c3/ with the sister repo's objects at commit efdc8c3.
# Uses `git show <full sha>:<path>` from a local clone's object store (content-addressed, read-only).
# Prints sha256 of both sides, byte counts and line counts. No file contents are printed.
set -u
TTJ=${TTJ:-/tmp/ttj_exp28_efdc8c3}
PROV=/tmp/e40_cursor/experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/provenance/ttj_efdc8c3
REV=$(git -C "$TTJ" rev-parse efdc8c3)
echo "sister commit resolved: $REV"
E28=experiments/28_multi_family_distributional_judge_and_open_weight_lora_on_new_ratings_start_26Nov2026
declare -A MAP=(
  [RESULTS_11-26-2026.md]="$E28/RESULTS_11-26-2026.md"
  [human_ratings_new_11-26-2026.csv]="$E28/data_11-26-2026/human_ratings_new_11-26-2026.csv"
  [human_ratings_old_11-26-2026.csv]="$E28/data_11-26-2026/human_ratings_old_11-26-2026.csv"
  [qwen3_4b_logit_6144.json]="$E28/results/qwen3_4b_logit_6144.json"
  [qwen3_8b_logit_6144.json]="$E28/results/qwen3_8b_logit_6144.json"
  [qwen3_4b_lora_ce_rps_pearson.json]="$E28/results/qwen3_4b_lora_ce_rps_pearson.json"
)
for f in $(printf '%s\n' "${!MAP[@]}" | sort); do
  a=$(sha256sum "$PROV/$f" | cut -d' ' -f1)
  b=$(git -C "$TTJ" show "$REV:${MAP[$f]}" | sha256sum | cut -d' ' -f1)
  na=$(wc -c < "$PROV/$f"); la=$(wc -l < "$PROV/$f")
  if [ "$a" = "$b" ]; then s=IDENTICAL; else s=DIFFERENT; fi
  echo "$s  $f  bytes=$na lines=$la  copy_sha256=$a  efdc8c3_sha256=$b"
done
# Are any receipts (sha256 values) for these files written inside the experiment 40 folder?
echo "--- sha256 receipts mentioned in exp40 text files (count of 64-hex strings per file):"
grep -rEoc '[0-9a-f]{64}' /tmp/e40_cursor/experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026 --include=*.md --include=*.py --include=*.txt 2>/dev/null | grep -v ':0$' || echo "none"
