#!/bin/bash
# Exp 18 real-data learning curve. Usage: bash run_all_n.sh <dataset_pool.json> <out_root> [workers]
set -u
DATA=$1; OUT=$2; W=${3:-32}; cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT"; echo "START $(date -Is) host=$(hostname) data_sha256=$(sha256sum "$DATA" | cut -d' ' -f1)" | tee -a "$OUT/launcher.log"
rc=0
for pool in old5 all; do for eps in 1e-6 5e-2; do for n in 4 8 12 16; do
  d="$OUT/${pool}_eps${eps}_ntrain${n}"
  python3 run_real_n.py --data "$DATA" --pool $pool --eps $eps --ntrain $n --out "$d" --splits 400 --boot 4000 --spearman_draws 1000 --workers $W >> "$OUT/launcher.log" 2>&1 || { echo "FAILED $pool $eps $n" | tee -a "$OUT/launcher.log"; rc=1; }
done; done; done
echo "END $(date -Is) rc=$rc" | tee -a "$OUT/launcher.log"; [ $rc -eq 0 ] && touch "$OUT/ALL_DONE"; exit $rc
