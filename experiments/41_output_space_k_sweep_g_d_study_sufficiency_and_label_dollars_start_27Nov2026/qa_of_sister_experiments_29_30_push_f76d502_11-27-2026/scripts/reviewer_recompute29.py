"""Independent recomputation for Exp 29 (does NOT import the experiment's code)."""
import csv, json, sys
import numpy as np
from scipy.optimize import minimize
SNAP = sys.argv[1]
rows = list(csv.DictReader(open(SNAP)))
N = len(rows)
R = [[int(x) for x in r['ratings_all'].split()] for r in rows]
task = np.array([r['task'] for r in rows]); cohort = np.array([r['set'] for r in rows])
split = np.array([r['split'] for r in rows]); fold = np.array([int(r['cv5_fold']) for r in rows])
cnt = np.array([np.bincount(x, minlength=6) for x in R], float)
P = cnt / cnt.sum(1, keepdims=True)

def rps_per_rating(Q, ratings):
    """mean over individual ratings of (1/5) sum_{t=0..4} (F_Q(t) - 1[r<=t])^2"""
    FQ = np.cumsum(Q)[:5]
    return np.mean([np.mean((FQ - (r <= np.arange(5)).astype(float))**2) for r in ratings])
def rps_art(Qs):
    return np.array([rps_per_rating(Qs[i], R[i]) for i in range(N)])
def ce_art(Qs):
    return np.array([np.mean([-np.log(Qs[i][r]) for r in R[i]]) for i in range(N)])
def tvd_art(Qs):
    return 0.5*np.abs(P - Qs).sum(1)

def cohort_marginals(fit_mask, eps=0.0):
    out = {}
    for s in ('old','new'):
        c = cnt[fit_mask & (cohort==s)].sum(0); p = c/c.sum()
        if eps: p = np.maximum(p, eps); p = p/p.sum()
        out[s] = p
    return out

feat_sets = {'h2_h4_struct': ['astra','opus','h4_astra','h4_argon','h4_opus','thm_ratio','excess_thm_log','ident_jaccard','edit_similarity'],
             'f8': ['astra','opus','h3_score','h3_expected'],
             'h2_h3_h4_struct': ['astra','opus','h3_expected','h4_astra','h4_argon','h4_opus','thm_ratio','excess_thm_log','ident_jaccard','edit_similarity']}
X_all = {c: np.array([float(r[c]) for r in rows]) for fs in feat_sets.values() for c in fs}

def fit_cumlogit(idx, feats, objective, alpha, mix, start='quantile', seed=0):
    X = np.column_stack([X_all[c][idx] for c in feats]); mu = X.mean(0); sd = X.std(0) + 1e-9; Xs = (X-mu)/sd
    pooled = cnt[idx].sum(0); pooled = pooled/pooled.sum(); pooled = np.maximum(pooled,1e-6); pooled/=pooled.sum()
    Pi = P[idx]
    def unpack(v):
        th = np.cumsum(np.r_[v[0], np.exp(v[1:5])]); return th, v[5:]
    def probs(v, Xs_, T=1.0):
        th, w = unpack(v); z = np.clip((th[None,:] - (Xs_@w)[:,None])/T, -40, 40)
        F = 1/(1+np.exp(-z)); F = np.hstack([np.zeros((len(Xs_),1)), F, np.ones((len(Xs_),1))])
        Q = (1-mix)*np.diff(F, axis=1) + mix*pooled
        Q = np.maximum(Q, 1e-6); return Q/Q.sum(1, keepdims=True)
    def loss(v):
        th, w = unpack(v); Q = probs(v, Xs)
        if objective == 'ce': s = -(Pi*np.log(Q)).sum(1)
        else:
            Fq = np.cumsum(Q,1)[:,:5]; Fp = np.cumsum(Pi,1)[:,:5]; s = np.mean((Fq-Fp)**2 + Fp*(1-Fp), 1)
        return s.mean() + 0.5*alpha*(w@w)
    if start == 'quantile':
        cdf = np.clip(np.cumsum(pooled)[:5], 1e-4, 1-1e-4); th0 = np.log(cdf/(1-cdf))
        v0 = np.r_[th0[0], np.log(np.maximum(np.diff(th0),1e-3)), np.zeros(len(feats))]
    else:
        rng = np.random.default_rng(seed); v0 = np.r_[-2.0, np.log(np.full(4, 0.8)), rng.normal(0, 0.5, len(feats))]
    res = minimize(loss, v0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-14})
    def predict(jdx, T=1.0):
        Xj = (np.column_stack([X_all[c][jdx] for c in feats]) - mu)/sd; return probs(res.x, Xj, T)
    return predict, res

def task_boot(vals, tasks, draws, seed):
    ut = sorted(set(tasks.tolist())); g = [np.where(tasks==t)[0] for t in ut]
    s = np.array([vals[x].sum() for x in g]); c = np.array([len(x) for x in g], float)
    rng = np.random.default_rng(seed); pick = rng.integers(0, len(ut), (draws, len(ut)))
    samp = s[pick].sum(1)/c[pick].sum(1)
    return float(vals.mean()), np.percentile(samp, [2.5, 97.5]), float(np.mean(samp >= 0))

out = {}
# ---------- 20-task split ----------
fitm = split != 'test'; te = np.where(split == 'test')[0]; fit_idx = np.where(fitm)[0]
cm = cohort_marginals(fitm); Qc = np.array([cm[cohort[i]] for i in range(N)])
cm_s = cohort_marginals(fitm, 1e-6); Qc_s = np.array([cm_s[cohort[i]] for i in range(N)])
rps_c = rps_art(Qc); rps_cs = rps_art(Qc_s)
# check histogram formula == per-rating formula
Fq = np.cumsum(Qc,1)[:,:5]; Fp = np.cumsum(P,1)[:,:5]; hist_formula = np.mean((Fq-Fp)**2 + Fp*(1-Fp), 1)
out['max_abs_perrating_minus_histformula'] = float(np.abs(hist_formula - rps_c).max())
out['test20_constant_RPS_nosmooth'] = float(rps_c[te].mean()); out['test20_constant_RPS_smooth'] = float(rps_cs[te].mean())
heads20 = {'F8': ('f8','ce',0.0,0.0,1.0), 'primary_rps_unconverged': ('f8','rps',0.0,0.25,2.0), 'best_converged': ('h2_h3_h4_struct','ce',0.1,0.0,0.5)}
for name,(fs,obj,a,m,T) in heads20.items():
    pred, res = fit_cumlogit(fit_idx, feat_sets[fs], obj, a, m)
    Q = np.zeros((N,6)); Q[te] = pred(te, T)
    r = rps_art(Q); d = r[te] - rps_cs[te]
    pt, ci, pge = task_boot(d, task[te], 10000, 20261127)
    pt2, ci2, _ = task_boot(d, task[te], 10000, 7)
    out['test20_'+name] = dict(RPS=float(r[te].mean()), CE=float(ce_art(Q)[te].mean()), dRPS=pt, ci_seed20261127=ci.tolist(), ci_seed7=ci2.tolist(), frac_ge0=pge, converged=bool(res.success), wnorm=float(np.linalg.norm(res.x[5:])))
    if name == 'F8': out['_d_F8_test'] = d.tolist(); out['_te'] = te.tolist()
# ---------- 50-task nested, using per-fold chosen configs from the receipt ----------
chosen = json.load(open(sys.argv[2]))['chosen_per_fold']
Qn = np.zeros((N,6)); Qcn = np.zeros((N,6))
for c in chosen:
    f = c['fold']; tem = np.where(fold == f)[0]; trm = fold != f
    pred, res = fit_cumlogit(np.where(trm)[0], feat_sets[c['feature_set']], c['objective'], c['alpha'], c['mix'])
    Qn[tem] = pred(tem, c['temperature'])
    cmf = cohort_marginals(trm, 1e-6); Qcn[tem] = np.array([cmf[cohort[i]] for i in tem])
    out[f'fold{f}_converged'] = bool(res.success)
rn = rps_art(Qn); rc = rps_art(Qcn); d = rn - rc
pt, ci, pge = task_boot(d, task, 10000, 20261127); _, ci7, _ = task_boot(d, task, 10000, 7); _, ci99, _ = task_boot(d, task, 10000, 99)
out['nested'] = dict(RPS_head=float(rn.mean()), RPS_const=float(rc.mean()), dRPS=pt, ci_seed20261127=ci.tolist(), ci_seed7=ci7.tolist(), ci_seed99=ci99.tolist(), CE_head=float(ce_art(Qn).mean()), CE_const=float(ce_art(Qcn).mean()), TVD_head=float(tvd_art(Qn).mean()), TVD_const=float(tvd_art(Qcn).mean()))
# pooled (literature-standard) constant: fold-wise pooled train marginal
Qpool = np.zeros((N,6))
for f in range(5):
    tem = np.where(fold == f)[0]; c_ = cnt[fold != f].sum(0); Qpool[tem] = c_/c_.sum()
out['nested_pooled_constant_RPS'] = float(rps_art(Qpool).mean())
# per-task analysis
ut = sorted(set(task.tolist())); per_task = {t: float(d[task==t].mean()) for t in ut}
out['nested_tasks_favoring_head'] = int(sum(v < 0 for v in per_task.values())); out['nested_tasks_favoring_const'] = int(sum(v > 0 for v in per_task.values()))
contrib = sorted(((float(d[task==t].sum()/N), t, cohort[task==t][0]) for t in ut))
out['nested_top5_contrib_to_mean'] = contrib[:5]; out['nested_worst3'] = contrib[-3:]
loto = []
for t in ut:
    k = task != t; p_, ci_, _ = task_boot(d[k], task[k], 4000, 20261127); loto.append((t, p_, ci_[1]))
out['nested_LOTO_max_upper'] = max(x[2] for x in loto); out['nested_LOTO_point_range'] = [min(x[1] for x in loto), max(x[1] for x in loto)]
lofo = []
for f in range(5):
    k = fold != f; p_, ci_, _ = task_boot(d[k], task[k], 10000, 20261127); lofo.append((f, round(p_,4), [round(x,4) for x in ci_]))
out['nested_LOFO'] = lofo
out['nested_per_fold_dRPS'] = [(f, round(float(d[fold==f].mean()),4)) for f in range(5)]
for s in ('old','new'):
    k = cohort == s; p_, ci_, _ = task_boot(d[k], task[k], 10000, 20261127); out['nested_'+s] = (round(p_,4), [round(x,4) for x in ci_])
# leave-one-cohort-family: drop most influential task
out['nested_drop_most_influential_task'] = None
t0 = contrib[0][1]; k = task != t0; p_, ci_, _ = task_boot(d[k], task[k], 10000, 20261127); out['nested_drop_most_influential_task'] = (t0, round(p_,4), [round(x,4) for x in ci_])
# t-based interval on task means (artifact-weighted ratio -> use task totals)
tm = np.array([d[task==t].sum() for t in ut]); tc = np.array([np.sum(task==t) for t in ut])
from scipy import stats
# cluster-robust SE for ratio mean
mean = tm.sum()/tc.sum(); resid = tm - mean*tc; se = np.sqrt(len(ut)/(len(ut)-1)*np.sum(resid**2))/tc.sum()
out['nested_cluster_t_interval'] = [mean - stats.t.ppf(.975, len(ut)-1)*se, mean + stats.t.ppf(.975, len(ut)-1)*se]
json.dump(out, open('/tmp/qa2930/indep/recompute29_out.json','w'), indent=1, default=str)
for k,v in out.items():
    if not k.startswith('_'): print(k, v)
