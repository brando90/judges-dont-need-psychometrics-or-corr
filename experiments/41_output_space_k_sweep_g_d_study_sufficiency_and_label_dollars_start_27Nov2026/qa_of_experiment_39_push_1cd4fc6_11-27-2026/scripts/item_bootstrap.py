"""Two-way (replication x test-item) bootstrap of U1-minus-PS test RPS, irreducible RPS constant,
and effect sizes relative to reducible (excess) RPS. argv: WORK dir."""
import sys, json
import numpy as np, pandas as pd
from scipy import stats
WORK = sys.argv[1]
sys.path.insert(0, WORK + "/src")
import study_a as A
bundle = A.load_bundle(); df = pd.read_csv(WORK + "/results/study_a_replications.csv")
S = json.load(open(WORK + "/results/study_a_summary.json"))
rng = np.random.default_rng(20261127)
NB = 4000
print("dataset B | U1-PS mean | Student-t CI (reps only) half-width | 2-way bootstrap 95% CI | width ratio | item-only bootstrap CI | sig t | sig 2-way")
summ = []
for name in A.DATASET_ORDER:
    d = bundle[name]; K = int(d["K"]); Q = A.transforms(d["p_mix"])
    test_idx = np.asarray(d["test_idx"], int); h = (d["counts"] / d["counts"].sum(1, keepdims=True))[test_idx]
    N = len(test_idx)
    # per-transform per-item RPS (code's definition) [27, N]
    per = np.stack([A.rps_tvd_ce(Q[test_idx, t, :], h)[0] for t in range(Q.shape[1])])
    G = h.cumsum(-1)[:, :-1]
    irr = (G * (1 - G)).mean(-1)          # forecast-independent item constant
    hist_rps = per - irr[None, :]         # (1/(K-1)) sum (F-G)^2 : documented formula
    sub = df[df.dataset == name]
    # constant forecasts: dev-pool pooled histogram, and uniform
    dev = np.asarray(d["dev_idx"], int); pooled = d["counts"][dev].sum(0) / d["counts"][dev].sum()
    rps_pool = A.rps_tvd_ce(np.repeat(pooled[None, :], N, 0), h)[0].mean()
    rps_unif = A.rps_tvd_ce(np.full((N, K), 1.0 / K), h)[0].mean()
    print(f"## {name}: N_test={N}, mean irreducible constant={irr.mean():.5f} (item SD {irr.std(ddof=1):.5f}); "
          f"constant pooled-dev-histogram forecast RPS={rps_pool:.5f}, uniform RPS={rps_unif:.5f}")
    for B in A.BUDGETS:
        if len(A.designs_for_budget(B)) < 2: continue
        iu = sub[f"B{B}_u1_transform"].to_numpy().astype(int); ip = sub[f"B{B}_ps_transform"].to_numpy().astype(int)
        # sanity: reconstruct stored RPS
        rec_u = per[iu].mean(1); rec_p = per[ip].mean(1)
        assert np.allclose(rec_u, sub[f"B{B}_u1_rps"].to_numpy(), atol=1e-12) and np.allclose(rec_p, sub[f"B{B}_ps_rps"].to_numpy(), atol=1e-12)
        Dm = per[iu] - per[ip]                      # [R, N]
        R = Dm.shape[0]
        diff = Dm.mean(1); m = diff.mean()
        se = diff.std(ddof=1) / np.sqrt(R); tq = stats.t.ppf(0.975, R - 1)
        t_lo, t_hi = m - tq * se, m + tq * se
        C = rng.multinomial(R, np.full(R, 1 / R), size=NB).astype(float)   # replication resampling
        W = rng.multinomial(N, np.full(N, 1 / N), size=NB).astype(float)   # test-item resampling
        both = ((C @ Dm) * W).sum(1) / (R * N)
        item_only = (W @ Dm.mean(0)) / N
        b_lo, b_hi = np.percentile(both, [2.5, 97.5]); i_lo, i_hi = np.percentile(item_only, [2.5, 97.5])
        ratio = (b_hi - b_lo) / (t_hi - t_lo) if t_hi > t_lo else float("nan")
        sig_t = (t_lo > 0) or (t_hi < 0); sig_b = (b_lo > 0) or (b_hi < 0)
        # effect sizes
        u1_mean = sub[f"B{B}_u1_rps"].mean(); u1_excess = u1_mean - irr.mean()
        or_mean = sub[f"B{B}_rps_oracle_rps"].mean()
        print(f"{name} {B} | {m:+.5f} | ±{(t_hi-t_lo)/2:.5f} [{t_lo:+.5f},{t_hi:+.5f}] | [{b_lo:+.5f},{b_hi:+.5f}] | x{ratio:.2f} | [{i_lo:+.5f},{i_hi:+.5f}] | {sig_t} | {sig_b} | "
              f"U1 RPS {u1_mean:.5f}, U1 excess over irreducible {u1_excess:.5f}, gap/excess={m/u1_excess*100:.1f}% , gap/U1={m/u1_mean*100:.2f}%")
        summ.append(dict(ds=name, B=B, mean=m, sig_t=bool(sig_t), sig_b=bool(sig_b), ratio=ratio, gap_rel_excess=m / u1_excess, gap_rel_total=m / u1_mean))
s = pd.DataFrame(summ)
print("cells significant (Student-t, reps only):", int(s.sig_t.sum()), "| significant under 2-way bootstrap:", int(s.sig_b.sum()))
print("width ratio 2-way/Student-t: median %.2f, range %.2f-%.2f" % (s.ratio.median(), s.ratio.min(), s.ratio.max()))
sig = s[s.sig_t]
print("significant cells: gap as %% of U1 total RPS: %.2f-%.2f%%; as %% of U1 excess RPS: %.1f-%.1f%%" % (100*sig.gap_rel_total.min(), 100*sig.gap_rel_total.max(), 100*sig.gap_rel_excess.min(), 100*sig.gap_rel_excess.max()))
print("significant under t but not 2-way:", sig[~sig.sig_b][["ds","B"]].values.tolist())
