"""Run the rank / mixed correlation objectives on experiment 13's exact grid: scenarios A/B x 11 values of n x 100 seeds.

Writes results/raw.jsonl incrementally (smallest n first). For seed 0 of every (scenario, n) it also refits
experiment 13's CE judge to confirm the regenerated data match the saved run.
Run from the repository root:
  python3 experiments/06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026/scripts/run_rank_sweep.py
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))
from multiprocessing import Pool  # noqa: E402

import rank_objectives as ro  # noqa: E402

NS = [4, 8, 16, 20, 32, 64, 128, 256, 512, 1024, 2048]
SEEDS, K, NTEST, PROCS = 100, 3, 500, 9
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "raw.jsonl")

if __name__ == "__main__":
    t0 = time.time()
    jobs = [(sc, n, s, K, NTEST, s == 0) for n in NS for sc in "AB" for s in range(SEEDS)]
    with Pool(PROCS) as pool, open(OUT, "w") as f:
        for i, (sc, n, s, o) in enumerate(pool.imap(ro.run_one, jobs, chunksize=2)):
            f.write(json.dumps(dict(scenario=sc, n=n, seed=s, **{k: {a: float(b) for a, b in v.items()} for k, v in o.items()})) + "\n")
            f.flush()
            if i % 100 == 0:
                print(i, len(jobs), round(time.time() - t0), "s", flush=True)
    print("done", round(time.time() - t0), "s", flush=True)
