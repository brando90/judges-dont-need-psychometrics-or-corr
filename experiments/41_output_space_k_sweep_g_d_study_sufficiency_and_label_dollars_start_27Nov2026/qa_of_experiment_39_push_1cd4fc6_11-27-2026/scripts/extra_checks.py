import sys, json
import numpy as np, pandas as pd
from scipy import stats
WORK = sys.argv[1]; sys.path.insert(0, WORK + "/src")
import study_a as A
Bs = json.load(open(WORK + "/results/study_b_summary.json"))
# 1. Study B multiplicity over 55 cells: REML-chosen minus equal 4x4
rows = [(k, c["paired_obs_chosen_minus_equal"]["mean"], c["paired_obs_chosen_minus_equal"].get("paired_t_pvalue")) for k, c in Bs["cells"].items()]
rows = [r for r in rows if r[2] is not None and np.isfinite(r[2])]
p = np.array([r[2] for r in rows]); m = len(p); order = np.argsort(p)
holm = [];
for i, j in enumerate(order):
    if p[j] <= 0.05 / (m - i): holm.append(rows[j][0])
    else: break
kmax = max([i for i, j in enumerate(order) if p[j] <= (i + 1) / m * 0.05], default=-1)
bh = [rows[order[i]][0] for i in range(kmax + 1)]
worse = [r for r in rows if r[1] > 0 and r[2] < 0.05]
print(f"Study B: {m} cells with p; raw p<0.05 & REML worse than equal: {len(worse)}; Holm survivors: {holm}; BH survivors: {bh}")
print("   smallest p among REML-worse cells: %.4g; Holm threshold for smallest p: %.4g" % (min(r[2] for r in worse), 0.05 / m))
# 2. third-design pick rate near crossovers and true v_Delta gap
df = pd.read_csv(WORK + "/results/study_b_replications.csv")
for t, third in ((0.30, "2x8"), (0.70, "8x2")):
    sub = df[np.isclose(df.t, t)]
    print(f"t={t}: third design {third} picked {int((sub.chosen == third).sum())}/{len(sub)} over all pilots; "
          f"true vDelta: 8x2 {0.2875+0.375*t:.4f}, 4x4 0.3875, 2x8 {0.6625-0.375*t:.4f}")
# exact (noise-free) expected regret vs equal from choice counts, region where 4x4 is optimal
for k in ("t0.50_16x4", "t0.50_32x8", "t0.45_64x4", "t0.55_64x4", "t0.60_16x4", "t0.60_64x4", "t0.30_16x4", "t0.70_64x8"):
    c = Bs["cells"][k]; print(f"   {k}: true regret vs oracle (=equal here) mean {c['true_regret']['mean']:.4f} "
                             f"({100*c['true_regret']['mean']/0.3875:.1f}% of vDelta(4x4)); observed chosen-minus-equal {c['paired_obs_chosen_minus_equal']['mean']:+.4f}")
# 3. absolute-level interval widening: two-way bootstrap of U1 mean test RPS (SNLI B=64, DICES-350 B=128)
bundle = A.load_bundle(); dfa = pd.read_csv(WORK + "/results/study_a_replications.csv")
rng = np.random.default_rng(7)
for name, B in (("chaosnli_snli", 64), ("dices350", 128)):
    d = bundle[name]; Q = A.transforms(d["p_mix"]); ti = np.asarray(d["test_idx"], int)
    h = (d["counts"] / d["counts"].sum(1, keepdims=True))[ti]
    per = np.stack([A.rps_tvd_ce(Q[ti, t, :], h)[0] for t in range(Q.shape[1])])
    sub = dfa[dfa.dataset == name]; iu = sub[f"B{B}_u1_transform"].to_numpy().astype(int)
    M = per[iu]; R, N = M.shape
    x = M.mean(1); se = x.std(ddof=1) / np.sqrt(R); tq = stats.t.ppf(0.975, R - 1)
    C = rng.multinomial(R, np.full(R, 1 / R), size=4000).astype(float); W = rng.multinomial(N, np.full(N, 1 / N), size=4000).astype(float)
    both = ((C @ M) * W).sum(1) / (R * N); lo, hi = np.percentile(both, [2.5, 97.5])
    print(f"{name} B={B} U1 mean test RPS {x.mean():.5f}: Student-t over reps ±{tq*se:.5f}; two-way bootstrap [{lo:.5f},{hi:.5f}] half-width {(hi-lo)/2:.5f} (x{(hi-lo)/(2*tq*se):.1f})")
# 4. fixed max-items rule (=U2) vs U1, for comparison with PS
S = json.load(open(WORK + "/results/study_a_summary.json"))
n = sum(1 for ds in S["datasets"] for B, c in S["datasets"][ds]["budgets"].items() if len(c["candidates"]) > 1 and c["paired"]["u1_minus_u2_rps"]["second_beats_first"])
print("fixed max-items rule (U2) significantly beats U1 on test RPS in", n, "of 20 cells")
# 5. how much rating data the PS selector touches vs the 80x16 pilot
for name in A.DATASET_ORDER:
    d = bundle[name]; dv = np.asarray(d["dev_idx"], int)
    print(f"{name}: pilot table 80x16 = 1280 ratings; PS selector draws from up to 80 x mean pool {d['n_avail'][dv].mean():.1f} = ~{80*d['n_avail'][dv].mean():.0f} ratings of the pilot items")
