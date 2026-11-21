"""Toy numeric illustration of R1/R2 (algebra check, not an experiment): Corr ties/inverts judges that differ in level and scale."""
import numpy as np
from scipy.stats import binom
K=6; r=np.arange(K)
t=np.array([0.5,1.5,2.5,3.5,4.5])
def bino(m): return binom.pmf(r,K-1,m/(K-1))
P=np.array([bino(x) for x in t])            # human distributions p*_i, mean t_i
def rps_exp(p,q):  # expected normalized RPS of forecast q when Y~p
    Fp=np.cumsum(p)[:-1];Fq=np.cumsum(q)[:-1]
    return np.mean([ (Fq[k]**2*(1-Fp[k])+(Fq[k]-1)**2*Fp[k]) for k in range(K-1)])
def ce_exp(p,q): return -(p*np.log(np.clip(q,1e-12,1))).sum()
def rep(name,Q):
    m=Q@r; rho=np.corrcoef(m,t)[0,1] if m.std()>0 else float("nan")
    rps=np.mean([rps_exp(P[i],Q[i]) for i in range(5)])
    ce=np.mean([ce_exp(P[i],Q[i]) for i in range(5)])
    tvd=np.mean([0.5*np.abs(P[i]-Q[i]).sum() for i in range(5)])
    print(f"{name}\t rho={rho:.3f}\t RPS={rps:.4f}\t CE={ce:.3f}\t TVD={tvd:.3f}\t means={np.round(m,2).tolist()}")
rep("truth",P)
rep("A compressed means m=2.5+0.1(t-2.5)",np.array([bino(2.5+0.1*(x-2.5)) for x in t]))
e=np.array([0.5,-0.5,0.5,-0.5,0.0])
rep("B noisy means m=t+e",np.array([bino(np.clip(x+d,0.05,4.95)) for x,d in zip(t,e)]))
rep("U uniform",np.tile(np.ones(K)/K,(5,1)))
rep("D reversed spread: shape-wrong, true means",np.array([ (lambda m: (0.5*bino(m)+0.5*np.eye(K)[int(round(m))]))(x) for x in t]))
