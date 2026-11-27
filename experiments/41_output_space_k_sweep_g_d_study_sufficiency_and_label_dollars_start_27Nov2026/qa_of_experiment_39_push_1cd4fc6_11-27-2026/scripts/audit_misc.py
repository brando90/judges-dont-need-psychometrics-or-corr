import json, sys
import numpy as np, pandas as pd
from scipy import stats
A = json.load(open(sys.argv[1])); Bs = json.load(open(sys.argv[2])); Bcsv = pd.read_csv(sys.argv[3])
c = A["datasets"]["chaosnli_snli"]["budgets"]["64"]
for tag in ("u1","ps","u2","rps_oracle"):
    r = c["metrics"]["rps"][tag]; print("SNLI64 RPS", tag, round(r["mean"],5), [round(x,5) for x in r["ci95"]])
for ds,B,met in (("chaosnli_snli","64","item_mse"),("dices350","128","item_mse"),("chaosnli_snli","64","grand_mse")):
    r = A["datasets"][ds]["budgets"][B]["paired"][f"u1_minus_u2_{met}"]; print(ds,B,met,"U1-U2",round(r["mean"],5),[round(x,5) for x in r["ci95"]],"p=%.3g"%r["paired_t_pvalue"])
# grand mse inconclusive cells
for ds in A["datasets"]:
    for B, cell in A["datasets"][ds]["budgets"].items():
        if len(cell["candidates"])<2: continue
        r = cell["paired"]["u1_minus_u2_grand_mse"]
        if not r["ci_excludes_0"]: print("grand-mse inconclusive:", ds, B, round(r["mean"],5), [round(x,5) for x in r["ci95"]])
# CE/TVD in the 10 PS-beats-U1 cells
for ds in A["datasets"]:
    for B, cell in A["datasets"][ds]["budgets"].items():
        if len(cell["candidates"])<2: continue
        r = cell["paired"]["u1_minus_ps_rps"]
        if r["second_beats_first"]:
            ce = cell["paired"]["u1_minus_ps_ce"]; tv = cell["paired"]["u1_minus_ps_tvd"]
            print("PS>U1 cell", ds, B, "CE sig" if ce["ci_excludes_0"] else "CE ns %s %s"%(round(ce["mean"],5),[round(x,5) for x in ce["ci95"]]), "| TVD sig" if tv["ci_excludes_0"] else "| TVD ns", "CEmean>0" if ce["mean"]>0 else "CEmean<=0", "TVDmean>0" if tv["mean"]>0 else "TVDmean<=0")
        for met in ("ce","tvd"):
            q = cell["paired"][f"u1_minus_ps_{met}"]
            if q["first_beats_second"]: print("  significant contradiction U1 better on", met, ds, B)
# Study B table
rows = [(0.0,"16x4"),(0.0,"64x8"),(0.15,"16x4"),(0.15,"64x8"),(0.30,"16x4"),(0.30,"64x8"),(0.50,"16x4"),(0.50,"64x8"),(0.70,"16x4"),(0.70,"64x8"),(0.85,"16x4"),(0.85,"64x8"),(1.0,"16x4"),(1.0,"64x8")]
for t,p in rows:
    cell = Bs["cells"][f"t{t:.2f}_{p}"]; w = cell["chose_oracle"]; e = cell["paired_obs_chosen_minus_equal"]
    print(f"t={t:.2f} {p} oracle={cell['oracle_design']} P={w['proportion']:.3f} [{w['ci95'][0]:.3f},{w['ci95'][1]:.3f}] ({w['k']}/{w['n']}) vs equal {e['mean']:+.3f} [{e['ci95'][0]:+.3f},{e['ci95'][1]:+.3f}] p={e.get('paired_t_pvalue',float('nan')):.3g} counts={cell['choice_counts']}")
# all t for W shape
print("P(match) by t (rows) and pilot (cols):")
pil = Bs["pilots"]
print("t     " + "  ".join(pil))
for t in Bs["t_grid"]:
    print(f"{t:.2f}  " + "  ".join(f"{Bs['cells'][f't{t:.2f}_{p}']['chose_oracle']['proportion']:.3f}" for p in pil))
# REML-worse-than-equal significant cells
print("cells where REML chosen is significantly worse than equal 4x4:")
for k, cell in Bs["cells"].items():
    e = cell["paired_obs_chosen_minus_equal"]
    if e["ci_excludes_0"] and e["mean"] > 0: print("  ", k, round(e["mean"],4), [round(x,4) for x in e["ci95"]], "p=%.3g" % e["paired_t_pvalue"], "true_regret_mean=%.4f" % cell["true_regret"]["mean"])
print("n Study B cells:", len(Bs["cells"]))
# hand recompute: Wilson and t-interval for t=0.50 16x4
sub = Bcsv[(Bcsv.t==0.5)&(Bcsv.pilot=="16x4")]
k = int(sub.match_oracle.sum()); n = len(sub); z = stats.norm.ppf(0.975); ph = k/n
cen = (ph + z*z/(2*n))/(1+z*z/n); half = z*np.sqrt(ph*(1-ph)/n + z*z/(4*n*n))/(1+z*z/n)
d = (sub.obs_chosen - sub.obs_equal).to_numpy(); m = d.mean(); se = d.std(ddof=1)/np.sqrt(n); tc = stats.t.ppf(0.975, n-1)
print(f"hand t=0.50 16x4: Wilson {k}/{n} [{cen-half:.4f},{cen+half:.4f}]; paired diff {m:.5f} [{m-tc*se:.5f},{m+tc*se:.5f}] p={stats.ttest_1samp(d,0).pvalue:.4g}")
# expected true regret vs equal at t=0.5
print("t=0.5 16x4 mean(true vDelta chosen - true vDelta equal) =", round(float((sub.true_vDelta_chosen - sub.true_vDelta_equal).mean()),5))
sub = Bcsv[(Bcsv.t==0.0)&(Bcsv.pilot=="64x8")]
d = (sub.obs_chosen - sub.obs_equal).to_numpy(); m = d.mean(); se = d.std(ddof=1)/np.sqrt(len(d)); tc = stats.t.ppf(0.975, len(d)-1)
print(f"hand t=0 64x8: paired {m:.4f} [{m-tc*se:.4f},{m+tc*se:.4f}] p={stats.ttest_1samp(d,0).pvalue:.3g}")
print("REML convergence: min reml_converged/n over cells =", min(c['reml_converged']/c['n_reps'] for c in Bs['cells'].values()))
