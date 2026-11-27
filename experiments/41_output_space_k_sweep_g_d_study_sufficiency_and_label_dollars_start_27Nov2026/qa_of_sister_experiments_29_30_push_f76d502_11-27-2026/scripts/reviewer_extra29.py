import sys, csv, numpy as np, collections
exec(open('/tmp/qa2930/indep/recompute29.py').read().split('# ---------- 20-task split')[0])
fitm = split != 'test'; te = np.where(split == 'test')[0]; fit_idx = np.where(fitm)[0]
cm = cohort_marginals(fitm, 1e-6); rc = rps_art(np.array([cm[cohort[i]] for i in range(N)]))
def loto_report(name, fs, obj, a, m, T):
    pred, res = fit_cumlogit(fit_idx, feat_sets[fs], obj, a, m); Q = np.zeros((N,6)); Q[te] = pred(te, T)
    d = rps_art(Q)[te] - rc[te]; tt = task[te]; ut = sorted(set(tt.tolist()))
    pt, ci, _ = task_boot(d, tt, 10000, 20261127)
    fav = sum(d[tt==t].mean() < 0 for t in ut)
    ups = []
    for t in ut:
        k = tt != t; p_, c_, _ = task_boot(d[k], tt[k], 4000, 20261127); ups.append(c_[1])
    print(f'{name}: point {pt:+.4f} CI [{ci[0]:+.4f},{ci[1]:+.4f}] tasks favoring head {fav}/20; LOTO upper range [{min(ups):+.4f},{max(ups):+.4f}]; #LOTO with upper<0: {sum(u<0 for u in ups)}/20')
    return float(rps_art(Q)[te].mean()), float(ce_art(Q)[te].mean())
loto_report('F8 (CE,h2_h3,a0,m0,T1)', 'f8','ce',0.0,0.0,1.0)
loto_report('primary (RPS,h2_h3,a0,m.25,T2)', 'f8','rps',0.0,0.25,2.0)
loto_report('best converged (CE,h2_h3_h4_struct,a.1,m0,T.5)', 'h2_h3_h4_struct','ce',0.1,0.0,0.5)
print('--- matched CE vs RPS objective on the 20-task test (same features/alpha/mix/T) ---')
for fs,a,m,T in (('f8',0.0,0.0,1.0),('f8',1.0,0.0,1.0),('h2_h3_h4_struct',0.1,0.0,0.5),('h2_h3_h4_struct',0.1,0.0,1.0),('h2_h3_h4_struct',1.0,0.0,1.0)):
    r = {}
    for obj in ('ce','rps'):
        pred, res = fit_cumlogit(fit_idx, feat_sets[fs], obj, a, m); Q = np.zeros((N,6)); Q[te] = pred(te, T); r[obj] = rps_art(Q)[te]
    d = r['rps'] - r['ce']; pt, ci, _ = task_boot(d, task[te], 10000, 20261127)
    print(f'  {fs} a={a} m={m} T={T}: test RPS of RPS-fit minus CE-fit {pt:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]')
# validation grid matched comparison
g = list(csv.DictReader(open('/tmp/qa2930/committed/e29/results/validation_grid_11-27-2026.csv')))
key = lambda r: (r['family'], r['feature_set'], r['alpha'], r['mix'], r['const_mode'], r['temperature'])
byk = collections.defaultdict(dict)
for r in g:
    if r['primary_eligible'] == 'True': byk[key(r)][r['objective']] = float(r['val_rps'])
diffs = np.array([v['rps'] - v['ce'] for v in byk.values() if 'ce' in v and 'rps' in v])
print('validation grid, primary-eligible matched pairs:', len(diffs), 'RPS-fit better in', int((diffs<0).sum()), 'median diff %+.4f' % np.median(diffs))
print('grid rows', len(g), 'primary-eligible', sum(r['primary_eligible']=='True' for r in g), 'unconverged rows', sum(r['optimizer_success']=='False' for r in g), 'unconverged primary-eligible', sum(r['optimizer_success']=='False' and r['primary_eligible']=='True' for r in g))
pe = sorted([r for r in g if r['primary_eligible']=='True'], key=lambda r: float(r['val_rps']))
print('top-5 primary-eligible val RPS:', [(r['feature_set'], r['objective'], r['alpha'], r['mix'], r['temperature'], round(float(r['val_rps']),5), r['optimizer_success']) for r in pe[:5]])
conv = [r for r in pe if r['optimizer_success']=='True']; print('best converged in grid:', conv[0]['feature_set'], conv[0]['objective'], conv[0]['alpha'], conv[0]['mix'], conv[0]['temperature'], conv[0]['val_rps'])
