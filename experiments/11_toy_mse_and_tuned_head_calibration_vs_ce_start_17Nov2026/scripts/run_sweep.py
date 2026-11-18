"""Run Experiment 11 full sweep: 4 scenarios x 11 sample sizes x 100 seeds = 4,400 cells.

Resumable; writes results/raw.jsonl. Run from repo root:
  python3 experiments/11_toy_mse_and_tuned_head_calibration_vs_ce_start_17Nov2026/scripts/run_sweep.py 9
"""
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mse_and_tuned_head as ex11  # noqa: E402

NS = [4, 8, 16, 20, 32, 64, 128, 256, 512, 1024, 2048]
SEEDS, K, NTEST = 100, 3, 500
PROCS = int(sys.argv[1]) if len(sys.argv) > 1 else 9
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw.jsonl")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t0 = time.time()
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["scenario"], r["n"], r["seed"]))
    jobs = [(sc, n, s, K, NTEST) for n in NS for sc in "ABCD" for s in range(SEEDS) if (sc, n, s) not in done]
    print(f"already done {len(done)}, remaining {len(jobs)} (procs={PROCS})", flush=True)
    with Pool(PROCS) as pool, open(OUT, "a") as f:
        for i, (sc, n, s, o) in enumerate(pool.imap_unordered(ex11.run_one, jobs, chunksize=4), 1):
            f.write(json.dumps({"scenario": sc, "n": n, "seed": s, "fits": o}) + "\n")
            if i % 200 == 0 or i == len(jobs):
                f.flush()
                print(f"{i}/{len(jobs)} done in {time.time() - t0:.1f}s", flush=True)
    print(f"COMPLETE in {time.time() - t0:.1f}s", flush=True)
