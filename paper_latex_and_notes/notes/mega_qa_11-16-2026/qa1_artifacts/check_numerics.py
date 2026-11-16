"""Bounded QA checks; run from checkout root. Never modifies measurement outputs."""
from pathlib import Path
import sys,json,time,platform
import numpy as np
import scipy
from scipy.optimize._numdiff import approx_derivative
ROOT=Path.cwd()
OUT=Path(__file__).resolve().parent
for p in ['06','07','08']:
 sys.path.insert(0,str(next(ROOT.glob('experiments/'+p+'*/scripts'))))
import rank_objectives as ro
import restarts as rs
import complexity as cx
toy=ro.toy
report={}
def emit(key,value):
 report[key]=value
 (OUT/'numeric_checks.json').write_text(json.dumps(report,indent=2,allow_nan=True)+'\n')
 print(key,json.dumps(value),flush=True)
emit('environment',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__})
rng=np.random.default_rng(20261116)
x=rng.uniform(-1,1,17); phi=toy.phi(x); p=rng.dirichlet(np.ones(6),17); w=rng.normal(0,.25,(13,6)).ravel()
grad={}
for kind in rs.OBJECTIVES:
 f=rs.objective_and_grad(phi,p,kind,len(p)); numeric=approx_derivative(lambda v:f(v)[0],w,method='3-point').ravel()
 grad[kind]=float(np.max(np.abs(numeric-f(w)[1])))
assert max(grad.values())<1e-7
emit('ordinary_gradient_max_absolute_error',grad)
# The inherited CE function clips and renormalizes its value but returns Q-P.
z=np.array([[0.,-30.,-2.,1.,-1.,-3.]])
p1=np.array([[0.,1.,0.,0.,0.,0.]])
v,g=toy.ce_loss_grad(z,p1)
ng=approx_derivative(lambda zz:toy.ce_loss_grad(zz.reshape(1,6),p1)[0][0],z.ravel()).ravel()
emit('clipped_ce_gradient_counterexample',{'loss':float(v[0]),'reported_gradient':g.ravel().tolist(),'finite_difference':ng.tolist(),'max_error':float(np.max(abs(ng-g.ravel())))})
# Verify exact keys and matching CE checks before any fresh fits.
raw={}
for n,pattern in [('06','experiments/06*/results/raw.jsonl'),('07','experiments/07*/results/raw.jsonl'),('08','experiments/08*/results/raw.jsonl'),('13','experiments/archive/13*/results/raw.jsonl')]:
 rows=[json.loads(l) for l in next(ROOT.glob(pattern)).read_text().splitlines()]
 d={(r['scenario'],r['n'],r['seed']):r for r in rows};assert len(d)==len(rows)
 raw[n]=d
emit('raw_rows',{k:len(v) for k,v in raw.items()})
# Instrument inherited CE clipping without changing its return values.
original=toy.ce_loss_grad
clip={'calls':0,'clipped_calls':0,'minimum_probability':1.0}
def observed(z,p):
 q=ro.softmax(z,1); clip['calls']+=1;clip['clipped_calls']+=int(np.any(q<toy.FLOOR));clip['minimum_probability']=min(clip['minimum_probability'],float(q.min()))
 return original(z,p)
toy.ce_loss_grad=observed

def differences(a,b,path=''):
 out=[]
 for k,v in a.items():
  if k not in b: continue
  q=b[k]
  if isinstance(v,dict):out+=differences(v,q,path+k+'.')
  elif isinstance(v,(int,float,bool)) and isinstance(q,(int,float,bool)):
   if np.isnan(v) and np.isnan(q): continue
   if v!=q:out.append((abs(float(v)-float(q)),path+k,float(v),float(q)))
 return out
for exp,sc,n,seed in [('06','A',4,0),('06','B',2048,0),('07','A',4,0),('07','B',1024,0),('08','C',4,9),('08','D',2048,0)]:
 t=time.time(); clip.update(calls=0,clipped_calls=0,minimum_probability=1.0)
 args=(sc,n,seed,3,500)
 result=ro.run_one(args+(True,)) if exp=='06' else (rs.run_one(args) if exp=='07' else cx.run_one(args))
 saved=raw[exp][(sc,n,seed)]; saved=saved if exp=='06' else saved['fits']
 dif=sorted(differences(result[3],saved),reverse=True)
 emit(f'reproduce_{exp}_{sc}_{n}_{seed}',{'seconds':round(time.time()-t,2),'max_differences':dif[:8],'clipping':dict(clip),'fallback':result[3].get('pearson_valcorr',{}).get('fallback')})
# A pre-registered exp06 prediction: rank/mix are at least as bad as Pearson after the CE crossover.
for sc in 'AB':
 n=2048
 for kind in ro.NEW_FITS:
  for met in ['rps_excess','ce_excess']:
   d=np.array([raw['06'][(sc,n,s)][kind][met]-raw['13'][(sc,n,s)]['pearson'][met] for s in range(100)])
   boot=d[np.random.default_rng(0).integers(0,100,(10000,100))].mean(1)
   emit(f'exp06_prediction_{sc}_{kind}_{met}',{'mean':float(d.mean()),'paired_percentile95':np.percentile(boot,[2.5,97.5]).tolist()})
emit('complete',True)
