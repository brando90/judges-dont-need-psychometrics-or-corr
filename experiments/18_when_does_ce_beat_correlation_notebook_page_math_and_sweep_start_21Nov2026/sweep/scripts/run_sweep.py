"""Exp 18 sweep A (fitted judges). Resumable; writes ../results/rawA.jsonl, then ../results/DONE_A.
Run: python3 run_sweep.py 4        (4 worker processes)  Date 11-21-2026."""
import json, os, sys, time
from multiprocessing import Pool
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_core as sc_

NS = [8, 16, 32, 64, 128, 256, 512]; KS = [1, 3, 10]; JS = [1, 3, 6]; SEEDS = 20; NTEST = 500
PROCS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"); OUT = os.path.join(RES, "rawA.jsonl")
if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True); t0 = time.time(); done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            if line.strip():
                r = json.loads(line); done.add((r["scenario"], r["n"], r["seed"], r["k"], r["J"]))
    jobs = [(s, n, sd, k, J, NTEST) for n in NS for k in KS for J in JS for s in "ABCD" for sd in range(SEEDS)
            if (s, n, sd, k, J) not in done]
    print(f"done {len(done)}, remaining {len(jobs)}", flush=True)
    with Pool(PROCS) as pool, open(OUT, "a") as f:
        for i, (s, n, sd, k, J, o) in enumerate(pool.imap_unordered(sc_.run_one, jobs, chunksize=2), 1):
            f.write(json.dumps({"scenario": s, "n": n, "seed": sd, "k": k, "J": J, "fits": o}, default=float) + "\n")
            if i % 100 == 0 or i == len(jobs):
                f.flush(); print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    open(os.path.join(RES, "DONE_A"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("COMPLETE", time.time() - t0)
