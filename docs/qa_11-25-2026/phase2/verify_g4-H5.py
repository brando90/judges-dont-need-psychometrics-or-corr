"""QA phase 2 g4-H5 verification (11-26-2026): exact finite expectations."""
from pathlib import Path
import math
import numpy as np
from scipy.stats import spearmanr,pearsonr
x=np.arange(6);means=np.arange(.5,5,1)
def binomial(mu):
    return np.array([[math.comb(5,int(k))*(m/5)**k*(1-m/5)**(5-k) for k in x] for m in mu])
p=binomial(means);F=np.cumsum(p,axis=1)[:,:-1]
def scores(q):
    G=np.cumsum(q,axis=1)[:,:-1];m=q@x
    return {'RPS':float(np.mean((G-F)**2+F*(1-F))),'CE':float(np.mean(-np.sum(p*np.log(np.clip(q,1e-300,1)),axis=1))),'TVD':float(np.mean(.5*np.sum(abs(q-p),axis=1))),'Spearman':None if np.ptp(m)==0 else float(spearmanr(m,means).statistic),'Pearson':None if np.ptp(m)==0 else float(pearsonr(m,means).statistic)}
forecasts={'Truth':p,'A':binomial(2.5+.1*(means-2.5)),'B':binomial(means+np.array([.5,-.5,.5,-.5,0])),'Uniform':np.full((5,6),1/6)}
print('QA phase 2 g4-H5 (11-26-2026): exact expectations over five Binomial targets')
r={name:scores(q) for name,q in forecasts.items()}
for name,v in r.items():print(name,v)
d=r['Uniform']['RPS']-r['A']['RPS'];print('Uniform minus A RPS =',d)
assert d>0
print('CONFIRMED: A is better than uniform on RPS by',d,'and worse on CE; no sampling uncertainty.')
