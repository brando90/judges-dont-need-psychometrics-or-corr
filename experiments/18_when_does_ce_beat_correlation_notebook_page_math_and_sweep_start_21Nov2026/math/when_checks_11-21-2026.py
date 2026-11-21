"""Numerical spot checks for the WHEN section of MATH_11-21-2026 (W1-W7). Run: python3 when_checks_11-21-2026.py (numpy, scipy). 11-21-2026. By ultimate_brando9 (Claude Sonnet 5.5, Claude Code). Output: when_checks_output_11-21-2026.txt"""
import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(0)
out=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.append(s)
# W1 binary
t=np.array([.2,.5,.8]); a,b=.3,.5; m=a+b*t
P("W1 corr(m,t)=",np.corrcoef(m,t)[0,1])
kl=lambda t,m: t*np.log(t/m)+(1-t)*np.log((1-t)/(1-m))
P("W1 Brier excess mean=",np.mean((m-t)**2),"KL mean nats=",np.mean(kl(t,m)))
# W2 K=3 fibre mean 1
for s in [0,.25,.5]:
    p=np.array([s,1-2*s,s]); r=np.arange(3)
    P("W2 s=",s,"mean",p@r,"var",p@(r-1)**2)
ps=np.array([.5,0,.5]); p=np.array([0,1,0])
F=lambda p:np.cumsum(p)[:-1]
P("W2 RPS(unnorm) between (0,1,0) and (1/2,0,1/2)=",np.sum((F(p)-F(ps))**2))
# W3 Bregman local
K=6;r=np.arange(K)
def tilt(eta):
    w=np.exp(eta*r);return w/w.sum()
from scipy.optimize import brentq
def eta_of(m): return brentq(lambda e: tilt(e)@r-m,-30,30)
for tt,mm in [(2.5,2.7),(1.5,1.8),(3.5,3.0)]:
    pt=tilt(eta_of(tt));pm=tilt(eta_of(mm));V=pt@(r-tt)**2
    P("W3 t,m",tt,mm,"KL",np.sum(pt*np.log(pt/pm)),"approx (m-t)^2/(2V)",(mm-tt)**2/(2*V))
# W3 CE optimum matches mean for misspecified target
pstar=np.array([.3,.05,.05,.05,.05,.5]); tt=pstar@r
res=minimize(lambda e: -np.sum(pstar*np.log(tilt(e[0]))),[0.0]); P("W3 CE-opt mean",tilt(res.x[0])@r,"target mean",tt)
# W5 OLS direction
n=400;d=4;Phi=rng.normal(size=(n,d))@rng.normal(size=(d,d));tv=np.tanh(Phi@rng.normal(size=d))+0.3*rng.normal(size=n)
Pc=Phi-Phi.mean(0);S=Pc.T@Pc/n;c=Pc.T@(tv-tv.mean())/n;w=np.linalg.solve(S,c)
rho=lambda w:np.corrcoef(Phi@w,tv)[0,1]
best=max((minimize(lambda w:-rho(w),rng.normal(size=d)).x for _ in range(5)),key=rho)
P("W5 rho at Sigma^-1 c=",rho(w),"numeric max=",rho(best),"cos(angle)=",best@w/np.linalg.norm(best)/np.linalg.norm(w))
# W4 binary logistic rigidity
n=50;phi=rng.normal(size=n);sg=lambda z:1/(1+np.exp(-z));tb=sg(1.3*phi-.4)
sols=[]
for _ in range(30):
    x0=rng.normal(size=2)*2
    rr=minimize(lambda w:-np.corrcoef(sg(w[0]*phi+w[1]),tb)[0,1],x0,method='BFGS');sols.append((rr.x,-rr.fun))
ok=[(x,f) for x,f in sols if f>1-1e-6]
P("W4 starts reaching rho>1-1e-6:",len(ok),"of 30; their (w,b):",sorted(set((round(x[0],2),round(x[1],2)) for x,f in ok)))
# W4b softmax small logits near affine closure
Wm=rng.normal(size=(K,6));X=rng.normal(size=(2000,6))
def meanof(W):
    u=X@W.T;e=np.exp(u-u.max(1,keepdims=True));p=e/e.sum(1,keepdims=True);return p@r
m1=meanof(Wm)
for cc in [1.0,.3,.1,.03]:
    m2=meanof(cc*Wm);A=np.c_[np.ones_like(m1),m1];co=np.linalg.lstsq(A,m2,rcond=None)[0]
    P("W4b c=",cc,"corr(m(cW),m(W))=",np.corrcoef(m1,m2)[0,1],"slope b=",co[1])
# attenuation
n=200000;s2=1.0;sig2=1.5
tt=rng.normal(size=n)*np.sqrt(s2);m=tt+rng.normal(size=n)*.5
for k in [1,3]:
    th=tt+rng.normal(size=n)*np.sqrt(sig2/k)
    P("W7 k=",k,"corr(m,t)=",np.corrcoef(m,tt)[0,1],"corr(m,t_hat)=",np.corrcoef(m,th)[0,1],"formula",np.corrcoef(m,tt)[0,1]/np.sqrt(1+sig2/(k*s2)))
open('checks_output.txt','w').write('\n'.join(out))
