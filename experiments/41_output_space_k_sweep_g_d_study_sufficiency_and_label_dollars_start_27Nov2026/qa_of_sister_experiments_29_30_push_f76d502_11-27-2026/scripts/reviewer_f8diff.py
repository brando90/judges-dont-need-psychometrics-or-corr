import numpy as np
exec(open('/tmp/qa2930/indep/recompute29.py').read().split('# ---------- 20-task split')[0])
fitm = split != 'test'; te = np.where(split == 'test')[0]; fit_idx = np.where(fitm)[0]
def head(fs,obj,a,m,T):
    pred,res = fit_cumlogit(fit_idx, feat_sets[fs], obj, a, m); Q=np.zeros((N,6)); Q[te]=pred(te,T); return rps_art(Q)[te]
f8 = head('f8','ce',0,0,1); bc = head('h2_h3_h4_struct','ce',0.1,0,0.5); pr = head('f8','rps',0,0.25,2)
for lab,d in (('best_converged - F8', bc-f8), ('primary(my longer fit) - F8', pr-f8)):
    p,ci,fr = task_boot(d, task[te], 10000, 20261127); print(lab, round(p,4), np.round(ci,4), 'frac>=0', fr)
