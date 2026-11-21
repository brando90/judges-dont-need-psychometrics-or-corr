"""Experiment 17 sweep: 4 scenarios x 11 sample sizes x 100 seeds, same grid and seeds as experiment 11. Resumable; writes results/raw.jsonl.
Usage: python3 run_sweep.py <processes>"""
import json, os, sys, time
from multiprocessing import Pool
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flex_head  # noqa: E402
NS = [4, 8, 16, 20, 32, 64, 128, 256, 512, 1024, 2048]; SEEDS, K, NTEST = 100, 3, 500
PROCS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw.jsonl")
if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True); t0 = time.time(); done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            if line.strip(): r = json.loads(line); done.add((r["scenario"], r["n"], r["seed"]))
    jobs = [(sc, n, s, K, NTEST) for n in NS for sc in "ABCD" for s in range(SEEDS) if (sc, n, s) not in done]
    print(f"already done {len(done)}, remaining {len(jobs)} (procs={PROCS})", flush=True)
    with Pool(PROCS) as pool, open(OUT, "a") as f:
        for i, (sc, n, s, o) in enumerate(pool.imap_unordered(flex_head.run_one, jobs, chunksize=2), 1):
            f.write(json.dumps({"scenario": sc, "n": n, "seed": s, "fits": o}) + "\n")
            if i % 100 == 0 or i == len(jobs): f.flush(); print(f"{i}/{len(jobs)} done in {time.time()-t0:.1f}s", flush=True)
    open(os.path.join(os.path.dirname(OUT), "DONE"), "w").write("done\n"); print(f"COMPLETE in {time.time()-t0:.1f}s", flush=True)
