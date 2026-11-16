"""Reproduce the observed clipped CE evaluation without modifying frozen sources."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.special import softmax,log_softmax
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(next(Path.cwd().glob('experiments/08*/scripts'))))
import complexity as cx
toy=cx.toy
original_fit,original_ce=toy.fit,toy.ce_loss_grad
trace=[]
def fit(phi,p,kind,n,w0,c=toy.RIDGE_C):
 events=[]
 def ce(z,p):
  q=softmax(z,1)
  if np.any(q<toy.FLOOR):events.append({'min_q':float(q.min()),'observed_clipped_mass':float(p[q<toy.FLOOR].sum())})
  return original_ce(z,p)
 toy.ce_loss_grad=ce
 try:w=original_fit(phi,p,kind,n,w0,c)
 finally:toy.ce_loss_grad=original_ce
 if events:trace.append({'kind':kind,'n':n,'c':c,'clipped_calls':events,'final_min_q':float(softmax(phi@w,1).min())})
 return w
toy.fit=fit
baseline=cx.run_one(('C',4,9,3,500))[3]
toy.fit=original_fit
def stable(z,p):return -(p*log_softmax(z,axis=1)).sum(1),softmax(z,1)-p
toy.ce_loss_grad=stable
try:counter=cx.run_one(('C',4,9,3,500))[3]
finally:toy.ce_loss_grad=original_ce
changed={k:{m:[v,counter[k][m]] for m,v in d.items() if isinstance(v,float) and np.isfinite(v) and v!=counter[k][m]} for k,d in baseline.items()}
result={'clipping_trace':trace,'stable_ce_counterfactual':changed}
assert trace and max(abs(b-a) for d in changed.values() for a,b in d.values())<1e-12
(OUT/'clipping_probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
