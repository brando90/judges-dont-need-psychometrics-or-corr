"""Independent finite-difference, RPS identity, selection and bounded cell checks for QA3."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.special import softmax
root=Path.cwd();out=Path(__file__).resolve().parent
exps={n:next((root/'experiments').glob(n+'_*')) for n in ['06','07','08']}
for p in exps.values():sys.path.insert(0,str(p/'scripts'))
import complexity as cx, rank_objectives as ro, restarts as rr
toy=cx.toy
rng=np.random.default_rng(761103)
# Independent finite differences of the actual six penalized training objectives.
x=rng.uniform(-1,1,9);Phi=toy.phi(x);P=rng.dirichlet(np.ones(6),9);w=rng.normal(0,.2,Phi.shape[1]*6)
checks={};eps=1e-6
for kind in rr.OBJECTIVES:
 f=rr.objective_and_grad(Phi,P,kind,len(x));v,g=f(w)
 gn=np.zeros_like(g)
 for j in range(len(w)):
  delta=np.zeros_like(w);delta[j]=eps
  gn[j]=(f(w+delta)[0]-f(w-delta)[0])/(2*eps)
 err=float(np.max(np.abs(g-gn)));checks[kind]=err
 assert err<2e-7,(kind,err)
# RPS normalized by five: compare expected observed loss, survival/CDF/absolute/energy forms.
a=np.arange(6);dist=np.abs(a[:,None]-a[None,:]);maxerr=0.
for _ in range(1000):
 p,q=rng.dirichlet(np.ones(6),2)
 pc,qc=np.cumsum(p)[:5],np.cumsum(q)[:5]
 loss_r=np.array([np.mean((qc-(a[:5]>=r))**2) for r in a])
 expected=float(p@loss_r);floor=float(np.mean(pc*(1-pc)));excess=float(np.mean((pc-qc)**2))
 absolute=np.array([q@dist[:,r]-.5*q@dist@q for r in a])/5
 energy=(2*p@dist@q-p@dist@p-q@dist@q)/10
 survive=(q@np.arange(6))-np.sum(1-np.cumsum(q)[:5])
 err=max(abs(expected-floor-excess),float(np.max(np.abs(loss_r-absolute))),abs(excess-energy),abs(survive))
 maxerr=max(maxerr,err)
assert maxerr<1e-14
# Selection and fallback specified in preregistration/amendment.
assert cx.pick(cx.C_GRID,[1,1,2,3,np.inf],True)==.1
assert cx.pick(cx.C_GRID,[np.nan,np.inf,-np.inf,np.nan,np.nan],False) is None
assert cx.pick(cx.C_GRID,[1,2,2,0,np.nan],False)==1
report={'gradient_max_abs_errors':checks,'rps_identity_trials':1000,'rps_identity_max_abs_error':maxerr,'selection_tests':'pass','cells':[]}
if '--resume' in sys.argv:
 report['cells']=[r for r in json.loads((out/'numeric_checks.json').read_text())['cells'] if not r['new_fields']]
(out/'numeric_checks.json').write_text(json.dumps(report,indent=2)+'\n')

def flatten(d,prefix=''):
 z={}
 for k,v in d.items():
  key=prefix+'/'+str(k)
  if isinstance(v,dict):z.update(flatten(v,key))
  elif isinstance(v,list):
   for j,t in enumerate(v):z[key+'/'+str(j)]=float(t)
  elif isinstance(v,(int,float,bool,np.number)):z[key]=float(v)
 return z
# Full rows at small/large sizes; originals use independent seed grids, so only sampled rows are rerun.
for n_exp,sc,n,seed in [('06','A',4,11),('06','B',1024,17),('07','A',4,11),('07','B',512,17),('08','A',4,11),('08','D',2048,17)]:
 if any((r['experiment'],r['scenario'],r['n'],r['seed'])==(n_exp,sc,n,seed) for r in report['cells']):continue
 start=time.monotonic();raw=[json.loads(l) for l in (exps[n_exp]/'results/raw.jsonl').read_text().splitlines()]
 saved=next(r for r in raw if (r['scenario'],r['n'],r['seed'])==(sc,n,seed))
 fn={'06':ro.run_one,'07':rr.run_one,'08':cx.run_one}[n_exp]
 args=(sc,n,seed,3,500)+( (False,) if n_exp=='06' else () )
 _,_,_,now=fn(args);old=flatten(saved if n_exp=='06' else saved['fits']);new=flatten(now)
 diffs={k:abs(v-old[k]) for k,v in new.items() if k in old and np.isfinite(v) and np.isfinite(old[k])}
 missing=[k for k in new if k not in old];nan_mismatch=[k for k,v in new.items() if k in old and np.isfinite(v)!=np.isfinite(old[k])]
 rec={'experiment':n_exp,'scenario':sc,'n':n,'seed':seed,'seconds':round(time.monotonic()-start,2),'compared_fields':len(diffs),'max_abs_difference':max(diffs.values()),'differences_over_1e-9':{k:v for k,v in diffs.items() if v>1e-9},'new_fields':missing,'nonfinite_mismatches':nan_mismatch}
 report['cells'].append(rec);(out/'numeric_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(rec),flush=True)
 assert not nan_mismatch
 assert all(n_exp=='08' and k.endswith('/fallback') and new[k]==0 for k in missing),missing
report['complete']=True
(out/'numeric_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print('Numeric checks complete',flush=True)
