#!/usr/bin/env bash
# Reproduce experiment 40 in the review copy: run_all.py (full grid, B=1000), then pytest.
set -u
R=/tmp/opus_review_exp40c
E=$R/repo/experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026
export SANMI36_DATA=$R/data_cache TTJ_ROOT=/tmp/ttj_exp28_efdc8c3 MPLCONFIGDIR=$R/mplcfg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
cd $R/repo
time nice -n 10 python3 experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/run_all.py
echo "run_all exit=$?"
cd $E && nice -n 10 python3 -m pytest tests -q 2>&1 | tail -5
