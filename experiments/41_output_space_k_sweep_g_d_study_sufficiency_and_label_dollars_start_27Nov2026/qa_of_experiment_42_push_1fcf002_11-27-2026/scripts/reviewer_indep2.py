import runpy, sys, io, contextlib, numpy as np
sys.argv = ["indep.py", "/tmp/qa42/repo/experiments/42_multimodal_simulated_population_ce_rps_vs_corr_gtheory_start_27Nov2026/data/empirical_shapes.json"]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path("indep.py", run_name="x")
addone_exact, strict, lenient, pol, chaos, B = g["addone_exact"], g["strict"], g["lenient"], g["pol"], g["chaos"], g["B"]
n=256
print("strict personaCE exact", round(addone_exact(n,[strict],strict),6), "pooled on strict exact", round(addone_exact(n,[strict,lenient,pol,chaos],strict),6))
ms = strict @ np.arange(6.0)
print("strict persona add-one bias exact", float(np.mean((n*ms+15)/(n+6) - ms)))
# max-Pearson tie: pooled add-one means are an affine map of the pooled empirical means (same N in every bin)
N=4*n; print("pooled add-one mean map: slope", N/(N+6), "intercept", 15/(N+6))
