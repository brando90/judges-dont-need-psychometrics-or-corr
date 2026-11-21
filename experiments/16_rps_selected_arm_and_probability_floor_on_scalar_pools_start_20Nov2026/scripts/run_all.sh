#!/bin/bash
# Experiment 16 launcher: 3 pools x 4 probability floors, 1,000 splits each, CPU only. End condition: every config writes DONE (or this script exits non-zero).
# Usage: bash run_all.sh <dataset_pool.json> <out_root> [workers]
set -u
DATA=$1; OUT=$2; W=${3:-24}; cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT"; echo "START $(date -Is) host=$(hostname) data_sha256=$(sha256sum "$DATA" | cut -d' ' -f1)" | tee -a "$OUT/launcher.log"
rc=0
for pool in old5 open all; do for eps in 1e-6 1e-3 1e-2 5e-2; do
  d="$OUT/${pool}_eps${eps}"
  python3 run_rps_floor.py --data "$DATA" --pool $pool --eps $eps --out "$d" --splits 1000 --boot 10000 --workers $W >> "$OUT/launcher.log" 2>&1 || { echo "FAILED $pool $eps" | tee -a "$OUT/launcher.log"; rc=1; }
done; done
echo "END $(date -Is) rc=$rc" | tee -a "$OUT/launcher.log"; [ $rc -eq 0 ] && touch "$OUT/ALL_DONE"; exit $rc
