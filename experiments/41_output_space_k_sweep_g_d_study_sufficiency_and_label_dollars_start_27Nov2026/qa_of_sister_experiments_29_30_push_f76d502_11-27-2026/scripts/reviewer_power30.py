"""Own numerical experiments for Exp 30 claims (no import of experiment code)."""
import sys, json, numpy as np
from scipy import stats
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
exec(open('/tmp/qa2930/indep/recompute29.py').read().split('# ---------- 20-task split')[0])  # reuse MY OWN helper definitions (load, fit, rps)
fitm = split != 'test'; te = np.where(split == 'test')[0]; fit_idx = np.where(fitm)[0]
cm = cohort_marginals(fitm, 1e-6)
pred, res = fit_cumlogit(fit_idx, feat_sets['f8'], 'ce', 0.0, 0.0)
Q = pred(te, 1.0); C = np.array([cm[cohort[i]] for i in te])
# per-rating paired values g_i(r) so that d_i = mean_j g_i(r_ij)  (linearity of the RPS difference in the empirical CDF)
FQ = np.cumsum(Q,1)[:,:5]; FC = np.cumsum(C,1)[:,:5]
g = []; d = []
for k,i in enumerate(te):
    vals = [np.mean((FQ[k]-(r<=np.arange(5)))**2) - np.mean((FC[k]-(r<=np.arange(5)))**2) for r in R[i]]
    g.append(vals); d.append(np.mean(vals))
d = np.array(d); T = task[te]; ut = sorted(set(T.tolist()))
print('F8 paired point', round(d.mean(),6))
# noise variance of each d_i: s_i^2/n_i (unbiased s^2 within artifact)
noise_i = np.array([np.var(v, ddof=1)/len(v) for v in g])
# variance decomposition for the ratio estimator: task totals
N_t = np.array([np.sum(T==t) for t in ut]); tot = np.array([d[T==t].sum() for t in ut]); m = len(ut)
mean = tot.sum()/N_t.sum(); resid = tot - mean*N_t
var_obs = np.sum(resid**2)/N_t.sum()**2            # bootstrap-like (no m/(m-1))
noise_tot = np.array([noise_i[T==t].sum() for t in ut]).sum()/N_t.sum()**2
print('SE_obs %.5f  noise share of variance %.3f' % (np.sqrt(var_obs), noise_tot/var_obs))
se_inf = np.sqrt(max(var_obs - noise_tot, 0))
for lab, se in (('observed', np.sqrt(var_obs)), ('infinite ratings (MoM)', se_inf)):
    print(f'{lab}: normal-approx upper = {mean + 1.96*se:+.5f}')
# ratings needed: with k ratings per artifact (instead of current n_i), noise scales n_i/k
n_i = np.array([len(v) for v in g])
for k in (4, 6, 8, 12, 20, 50):
    nv = np.array([np.var(v, ddof=1)/k for v in g]); nt = np.array([nv[T==t].sum() for t in ut]).sum()/N_t.sum()**2
    se_k = np.sqrt(max(var_obs - noise_tot + nt, 0)); extra = int(np.sum(np.maximum(k - n_i, 0)))
    print(f'  {k} ratings/artifact (+{extra} ratings): expected normal upper {mean + 1.96*se_k:+.5f}')
# ---- Synthetic check: does a task bootstrap shrink with more ratings per artifact? ----
rng = np.random.default_rng(12345)
def boot_upper(dv, tv, draws=2000, seed=1):
    u = sorted(set(tv.tolist())); gi = [np.where(tv==t)[0] for t in u]; s = np.array([dv[x].sum() for x in gi]); c = np.array([len(x) for x in gi], float)
    p = np.random.default_rng(seed).integers(0, len(u), (draws, len(u))); smp = s[p].sum(1)/c[p].sum(1); return np.percentile(smp, [2.5, 97.5])
mT, A = 20, 3.5
res_syn = {}
for nrat in (2, 3, 4, 8, 16, 64):
    widths = []
    for rep in range(60):
        r2 = np.random.default_rng(1000+rep)
        dv=[]; tv=[]
        for t in range(mT):
            task_eff = r2.normal(-0.035, 0.03)
            for a in range(int(A)):
                # true per-rating values: task effect + artifact effect; rating noise sd 0.12 (order of observed within-artifact sd)
                art = task_eff + r2.normal(0, 0.02)
                dv.append(np.mean(art + r2.normal(0, 0.12, nrat))); tv.append(t)
        lo, hi = boot_upper(np.array(dv), np.array(tv), 2000, rep)
        widths.append(hi - lo)
    res_syn[nrat] = float(np.median(widths))
print('synthetic median 95% task-bootstrap width by ratings/artifact:', {k: round(v,4) for k,v in res_syn.items()})
print('observed within-artifact sd of per-rating paired values: median %.4f' % np.median([np.std(v, ddof=1) for v in g]))
# ---- extrapolation sensitivity (G3) ----
pt, U20 = -0.035133, 0.002010; dist = U20 - pt; se20 = dist/1.96; sd_task = se20*np.sqrt(20)
def m_cross(z_fn, effect=-pt, sd=sd_task):
    for mm in range(3, 2000):
        if effect - z_fn(mm)*sd/np.sqrt(mm) > 0: return mm
print('normal (their rule, ~50%% power): m=%d' % m_cross(lambda mm: 1.96))
print('t_{m-1} with m/(m-1) correction:', m_cross(lambda mm: stats.t.ppf(.975, mm-1)*np.sqrt(mm/(mm-1))))
print('80%% power (z 1.96+0.84): m=%d ; 90%% power: m=%d' % (m_cross(lambda mm: 1.96+0.8416), m_cross(lambda mm: 1.96+1.2816)))
# uncertainty in SD from 20 tasks: chi-square 95% CI for sd
lo_sd = sd_task*np.sqrt(19/stats.chi2.ppf(.975,19)); hi_sd = sd_task*np.sqrt(19/stats.chi2.ppf(.025,19))
print('sd_task %.4f [%.4f, %.4f]; crossing m at sd bounds: %d / %d' % (sd_task, lo_sd, hi_sd, m_cross(lambda mm:1.96, sd=lo_sd), m_cross(lambda mm:1.96, sd=hi_sd)))
# effect uncertainty: at lower end of a plausible effect (e.g., half the point estimate)
print('if true effect were -0.025 / -0.020 / -0.0176 (half): m=%d / %d / %d' % (m_cross(lambda mm:1.96, effect=0.025), m_cross(lambda mm:1.96, effect=0.020), m_cross(lambda mm:1.96, effect=0.0176)))
print('seed-7 upper bound +0.0029 -> crossing m with that distance: %d' % (lambda dd: next(mm for mm in range(20,500) if pt + dd*np.sqrt(20/mm) < 0))(0.00287 - pt))
