"""QA g2-H3: Kendall CE and RPS crossovers from paired raw seeds (11-26-2026)."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
p=next((ROOT/'experiments').glob('06_*'))
new=[json.loads(l) for l in (p/'results/raw.jsonl').read_text().splitlines()]
old={ (r['scenario'],r['n'],r['seed']):r for r in (json.loads(l) for l in next((ROOT/'experiments/archive').glob('13_*/results/raw.jsonl')).read_text().splitlines())}
cells={}
for r in new:
 key=(r['scenario'],r['n'],r['seed']); cells[key]={j:old[key][j] for j in ('ce','rps','pearson')}|{j:r[j] for j in ('spearman','kendall','pearson_spearman')}
ns=sorted({r['n'] for r in new}); rng=np.random.default_rng(0)
# Original plotting consumes this stream before the numerical table bootstrap.
# 4 proper-score panels: 6 judges + 4 differences; 4 correlation panels: 6 judges.
for _ in range((4*(6+4)+4*6)*len(ns)): rng.integers(0,100,(10000,100))
print('QA g2-H3 (11-26-2026): CONFIRMED; scenario A, 100 paired seeds, 500 held-out tasks/seed')
for met in ('rps_excess','ce_excess'):
 better=[]
 for n in ns:
  for j in ('rps','pearson','spearman','kendall','pearson_spearman'):
   d=np.array([cells['A',n,s][j][met]-cells['A',n,s]['ce'][met] for s in range(100)])
   ci=np.quantile(d[rng.integers(0,100,(10000,100))].mean(1),[.025,.975])
   if j=='kendall':
    better.append(ci[0]>0)
    if n==8: print(met,'Kendall-trained minus CE-trained, n=8:',f'{d.mean():.12f}','95% interval:',*[f'{v:.12f}' for v in ci])
 start=next((ns[k] for k in range(len(ns)) if all(better[k:])),None)
 print(met,'CE-trained better at all tested sizes from',start,'tasks')
 assert start==({'rps_excess':8,'ce_excess':16}[met])
