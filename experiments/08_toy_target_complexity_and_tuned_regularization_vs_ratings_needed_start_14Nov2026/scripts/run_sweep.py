"""Run experiment 08: 4 scenarios x 11 training sizes x 100 seeds; resumable; writes results/raw.jsonl.

Run from the repository root:
  python3 experiments/08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/scripts/run_sweep.py [processes]
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

import complexity as cx  # noqa: E402

NS = [4, 8, 16, 20, 32, 64, 128, 256, 512, 1024, 2048]
SEEDS, K, NTEST = 100, 3, 500
PROCS = int(sys.argv[1]) if len(sys.argv) > 1 else 9
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "raw.jsonl")

if __name__ == "__main__":
    t0 = time.time()
    done = set()
    if os.path.exists(OUT):
        done = {(r["scenario"], r["n"], r["seed"]) for r in (json.loads(l) for l in open(OUT))}
    jobs = [(sc, n, s, K, NTEST) for n in NS for sc in "ABCD" for s in range(SEEDS) if (sc, n, s) not in done]
    print("already done", len(done), "remaining", len(jobs), flush=True)
    with Pool(PROCS) as pool, open(OUT, "a") as f:
        for i, (sc, n, s, o) in enumerate(pool.imap(cx.run_one, jobs, chunksize=4)):
            f.write(json.dumps(dict(scenario=sc, n=n, seed=s, fits=o)) + "\n")
            f.flush()
            if i % 400 == 0:
                print(i, len(jobs), round(time.time() - t0), "s", flush=True)
    print("done", round(time.time() - t0), "s", flush=True)
