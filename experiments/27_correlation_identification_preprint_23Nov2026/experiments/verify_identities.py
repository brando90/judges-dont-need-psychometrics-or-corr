"""Check distribution identities and Pearson logit derivatives from definitions."""
import json
from fractions import Fraction as F
import numpy as np
rng=np.random.default_rng(20261123)
def rho(q,p):
 r=np.arange(q.shape[1]);a=q@r;b=p@r
 return np.corrcoef(a,b)[0,1]
def ce(q,p):return -np.sum(p*np.log(q),axis=1)
def erps(q,p):
 fq=np.cumsum(q,axis=1)[:,:-1]
 fp=np.cumsum(p,axis=1)[:,:-1]
 return np.mean(fq*fq-2*fq*fp+fp,axis=1)
errs=dict(rps=0.0,cdf_mean=0.0,gradient=0.0,ce_decomposition=0.0)
for _ in range(200):
 n,k=7,6
 p=rng.dirichlet(np.ones(k),n);q=rng.dirichlet(np.ones(k),n)
 r=np.arange(k);t=p@r;m=q@r
 d=np.cumsum(p-q,axis=1)[:,:-1]
 ex=erps(q,p)-erps(p,p)
 split=(m-t)**2/(k-1)**2+np.mean((d-d.mean(axis=1,keepdims=True))**2,axis=1)
 errs['rps']=max(errs['rps'],float(np.max(np.abs(ex-split))))
 errs['cdf_mean']=max(errs['cdf_mean'],float(np.max(np.abs(d.sum(axis=1)-(m-t)))))
 assert np.all(np.abs(m-t)<=(k-1)*.5*np.sum(np.abs(p-q),axis=1)+1e-12)
 sm=m.std();st=t.std();corr=rho(q,p)
 g=((t-t.mean())/st-corr*(m-m.mean())/sm)/(n*sm)
 grad=q*(r[None,:]-m[:,None])*g[:,None]
 u=np.log(q);i,j=2,3;eps=1e-5
 def softmax(v):
  a=np.exp(v-v.max(axis=1,keepdims=True));return a/a.sum(axis=1,keepdims=True)
 up=u.copy();um=u.copy();up[i,j]+=eps;um[i,j]-=eps
 num=(rho(softmax(up),p)-rho(softmax(um),p))/(2*eps)
 errs['gradient']=max(errs['gradient'],abs(float(num-grad[i,j])))
 # Match each target mean with a tilted uniform via deterministic bisection.
 lo=np.full(n,-40.0);hi=np.full(n,40.0)
 for __ in range(90):
  eta=(lo+hi)/2;v=eta[:,None]*r;v-=v.max(axis=1,keepdims=True)
  qm=np.exp(v);qm/=qm.sum(axis=1,keepdims=True)
  mask=qm@r<t;lo=np.where(mask,eta,lo);hi=np.where(mask,hi,eta)
 hp=-np.sum(p*np.log(p),axis=1);hm=-np.sum(qm*np.log(qm),axis=1)
 gap=hm-hp;kl=np.sum(qm*(np.log(qm)-np.log(q)),axis=1)
 cross=-np.sum((p-qm)*np.log(q),axis=1)
 errs['ce_decomposition']=max(errs['ce_decomposition'],float(np.max(np.abs(ce(q,p)-(hp+gap+kl+cross)))))
for e in errs.values():assert e<1e-8,errs
P=[[F(3,5),F(3,10),F(1,10)],[F(1,5),F(3,5),F(1,5)],[F(1,10),F(3,10),F(3,5)]]
Q=[[F(13,20),F(1,5),F(3,20)],[F(2,5),F(1,5),F(2,5)],[F(3,20),F(1,5),F(13,20)]]
means=lambda a:[sum(j*v for j,v in enumerate(row)) for row in a]
assert means(P)==means(Q)==[F(1,2),F(1),F(3,2)]
tvd=sum(sum(abs(a-b) for a,b in zip(pr,qr))/2 for pr,qr in zip(P,Q))/3
xrps=sum(sum((sum(pr[:j+1])-sum(qr[:j+1]))**2 for j in range(2))/2 for pr,qr in zip(P,Q))/3
assert tvd==F(1,5) and xrps==F(3,200)
print(json.dumps({'seed':20261123,'random_cases':200,'maximum_errors':errs,'example':{'means':[str(x) for x in means(P)],'pearson':1,'TVD':str(tvd),'excess_RPS':str(xrps),'expected_RPS_true':float(erps(np.array(P,dtype=float),np.array(P,dtype=float)).mean()),'expected_RPS_forecast':float(erps(np.array(Q,dtype=float),np.array(P,dtype=float)).mean()),'CE_true':float(ce(np.array(P,dtype=float),np.array(P,dtype=float)).mean()),'CE_forecast':float(ce(np.array(Q,dtype=float),np.array(P,dtype=float)).mean())}},indent=2))
