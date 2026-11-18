"""Smoke checks and timing extrapolation for Experiment 11 before full 4,400-cell sweep."""
import os
import sys
import time
import numpy as np
from scipy.optimize import check_grad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mse_and_tuned_head as ex11  # noqa: E402

EXP08_SCRIPTS = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026",
    "scripts",
))
sys.path.insert(0, EXP08_SCRIPTS)
import complexity as cx08  # noqa: E402


def main():
    rng = np.random.default_rng(42)
    Phi = ex11.toy.phi(rng.uniform(-1, 1, 20))
    P = np.array([rng.multinomial(3, p) for p in ex11.truth(np.linspace(-0.8, 0.8, 20), "B")]) / 3.0
    w0 = rng.normal(scale=0.3, size=Phi.shape[1] * 6)
    g_ce = check_grad(lambda w: ex11.ce_exact_obj_grad(w, Phi, P, 0.05)[0],
                      lambda w: ex11.ce_exact_obj_grad(w, Phi, P, 0.05)[1], w0)
    g_mse = check_grad(lambda w: ex11.mse_obj_grad(w, Phi, P, 0.05)[0],
                       lambda w: ex11.mse_obj_grad(w, Phi, P, 0.05)[1], w0)
    z = rng.normal(size=20)
    p_bar = np.clip(P.mean(0), 1e-6, None); p_bar /= p_bar.sum()
    p0 = rng.normal(scale=0.2, size=12)
    g_h = check_grad(lambda p: ex11.stage2_obj_grad(p, z, P, np.log(p_bar), 0.1)[0],
                     lambda p: ex11.stage2_obj_grad(p, z, P, np.log(p_bar), 0.1)[1], p0)
    print(f"check_grad errors: ce_exact={g_ce:.2e}, mse={g_mse:.2e}, stage2_head={g_h:.2e}")
    assert max(g_ce, g_mse, g_h) < 1e-6

    # Check Stage-2 c_b -> inf limit recovers constant marginal P.mean(0)
    a_inf, b_inf = ex11.fit_stage2_head(z, P, 1e8)
    q_inf = ex11.predict_head(z, a_inf, b_inf)
    max_marginal_diff = float(np.abs(q_inf - P.mean(0)[None, :]).max())
    print(f"Stage-2 c_b=1e8 max |q - P.mean(0)| = {max_marginal_diff:.2e}, max|b| = {np.abs(b_inf).max():.2e}")
    assert max_marginal_diff < 1e-6

    # Check seed=0 reproduction vs Exp 08 constant and pearson_fixed across Scenarios A-D at n=16 & n=256
    for sc in "ABCD":
        for n in (16, 256):
            _, _, _, o11 = ex11.run_one((sc, n, 0, 3, 500))
            _, _, _, o08 = cx08.run_one((sc, n, 0, 3, 500))
            dc = abs(o11["constant"]["rps_excess"] - o08["constant"]["rps_excess"])
            dp = abs(o11["pearson_fixed"]["rps_excess"] - o08["pearson_fixed"]["rps_excess"])
            assert dc < 1e-12 and dp < 1e-11, f"Mismatch {sc} n={n}: dc={dc}, dp={dp}"
    print("Exp 08 parity check PASSED (<1e-11) for seed=0 across A-D at n=16, 256.")

    # Quick small-n vs large-n diagnostic + timing benchmark on 12 jobs
    t0 = time.time()
    sample_jobs = [(sc, n, 0, 3, 500) for sc in ("A", "B") for n in (4, 8, 16, 32, 256, 2048)]
    for job in sample_jobs:
        sc, n, s, _, _ = job
        _, _, _, o = ex11.run_one(job)
        if n in (8, 32, 2048):
            print(
                f"{sc} n={n:4d}: const={o['constant']['rps_excess']:.4f} | "
                f"ce={o['ce_fixed']['rps_excess']:.4f} | pear={o['pearson_fixed']['rps_excess']:.4f} | "
                f"p_naive={o['pearson_head_naive']['rps_excess']:.4f} | "
                f"p_in_tuned={o['pearson_head_in_tuned']['rps_excess']:.4f}(c={o['pearson_head_in_tuned']['c_b']:g}) | "
                f"p_tuned={o['pearson_head_tuned']['rps_excess']:.4f}(c={o['pearson_head_tuned']['c_b']:g}) | "
                f"mse_fix={o['mse_fixed']['rps_excess']:.4f} | mse_val={o['mse_valrps']['rps_excess']:.4f} | "
                f"mse_head={o['mse_head_tuned']['rps_excess']:.4f}"
            )
    dt = time.time() - t0
    est_9proc = (dt / len(sample_jobs)) * (4400.0 / 9.0)
    print(f"12 benchmark jobs took {dt:.2f}s single-core; extrapolated 4,400 jobs on 9 cores ~ {est_9proc:.1f}s")


if __name__ == "__main__":
    main()
