"""QA phase 2 g4-H1 verification (11-26-2026): raw seed-bootstrap replay."""
from pathlib import Path
import collections, gzip, json
import numpy as np
root=Path(__file__).resolve().parents[3]
e18=next((root/'experiments').glob('18_*'))
cells=collections.defaultdict(dict)
with gzip.open(e18/'sweep/results/rawA.jsonl.gz','rt') as f:
    for line in f:
        r=json.loads(line); cells[(r['scenario'],r['n'],r['k'],r['J'])][r['seed']]=r['fits']
rng=np.random.default_rng(0)
def ci(d):
    d=np.asarray(d,float); d=d[~np.isnan(d)]
    if not len(d): return (np.nan,np.nan)
    return np.percentile(d[rng.integers(0,len(d),(2000,len(d)))].mean(1),[2.5,97.5])
counts=collections.Counter()
for (scenario,n,k,j),seeds in sorted(cells.items()):
    for other in ['pearson_raw','pearson_head_tuned','mse_tuned','rps_tuned','constant']:
        intervals=[ci([seeds[s][other][metric]-seeds[s]['ce_tuned'][metric] for s in sorted(seeds)]) for metric in ['rps','ce_excess']]
        lo_r,hi_r=intervals[0];lo_c,hi_c=intervals[1]
        v='CE' if lo_r>0 and lo_c>0 else 'other' if hi_r<0 and hi_c<0 else 'tie' if lo_r<=0<=hi_r and lo_c<=0<=hi_c else 'mixed'
        if scenario in 'CD' and j==1 and other=='pearson_head_tuned':counts[v]+=1
print('QA phase 2 g4-H1 (11-26-2026)')
print('Raw A cells:',len(cells),'records:',sum(map(len,cells.values())))
print('C/D, J=1 joint RPS/CE-excess bootstrap verdicts:',dict(counts))
print('Head wins:',counts['other'],'of',sum(counts.values()))
assert counts['other']==22 and sum(counts.values())==42
print('CONFIRMED: old 22 of 40; corrected 22 of 42.')
