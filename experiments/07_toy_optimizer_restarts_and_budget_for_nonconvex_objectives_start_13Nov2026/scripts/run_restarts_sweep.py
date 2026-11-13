"""Run experiment 07: scenarios A/B x 9 training sizes x 30 seeds; writes results/raw.jsonl incrementally.

Run from the repository root:
  python3 experiments/07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives_start_13Nov2026/scripts/run_restarts_sweep.py
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

import restarts as rs  # noqa: E402

NS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
SEEDS, K, NTEST, PROCS = 30, 3, 500, 9
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "raw.jsonl")

if __name__ == "__main__":
    t0 = time.time()
    jobs = [(sc, n, s, K, NTEST) for n in NS for sc in "AB" for s in range(SEEDS)]
    with Pool(PROCS) as pool, open(OUT, "w") as f:
        for i, (sc, n, s, o) in enumerate(pool.imap(rs.run_one, jobs, chunksize=1)):
            f.write(json.dumps(dict(scenario=sc, n=n, seed=s, fits=o)) + "\n")
            f.flush()
            if i % 50 == 0:
                print(i, len(jobs), round(time.time() - t0), "s", flush=True)
    print("done", round(time.time() - t0), "s", flush=True)
