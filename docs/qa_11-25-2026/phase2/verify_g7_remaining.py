"""Check g7 MEDIUM/LOW findings against saved outputs (11-26-2026)."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[3]
def ex(n):
    return next((ROOT / 'experiments').glob(f'{n:02d}_*'))
def rows(p):
    return list(csv.DictReader(p.open()))
print('g7 remaining checks (11-26-2026)')
print('g7-M1: saved score aggregation; original labels are not reconstructed')
group = defaultdict(dict)
for r in rows(ex(28)/'results/scores_absolute.csv'):
    group[(r['dataset'], r['judge'], r['d'], r['B'])][r['method']] = float(r['rps'])
wins = [k for k,v in group.items() if v['pearson'] < v['ce']]
significant = [r for r in rows(ex(28)/'results/comparator_wins.csv')
               if r['metric']=='rps' and r['comparator']=='pearson'
               and float(r['lower_975_comparator_better']) > 0]
print('Point RPS wins',len(wins),'/',len(group),Counter((k[2],k[3]) for k in wins))
print('Simultaneous-bound RPS wins',len(significant),Counter((r['d'],r['B']) for r in significant))
assert len(wins)==25 and len(significant)==19 and len(group)==325
raw = [json.loads(line) for line in (ex(7)/'results/raw.jsonl').open()]
print('g7-M2: raw record count',len(raw),'scenarios',len({r['scenario'] for r in raw}),
      'sizes',len({r['n'] for r in raw}),'seeds',len({r['seed'] for r in raw}))
for method in raw[0]['fits']:
    counts={len(r['fits'][method]['restart_train_obj']) for r in raw if 'restart_train_obj' in r['fits'][method]}
    print(method,'restart counts',sorted(counts) if counts else 'derived head; no independent restart array')
assert len(raw)==540 and len({r['seed'] for r in raw})==30
for p in sorted((ex(20)/'results').glob('out_real_lam*.json')):
    x=json.loads(p.read_text());print('g7-M3:',p.name,'splits',x['splits'])
    assert x['splits']==250
print('g7-M4: reconstruct all E21 contrast lower bounds from saved fold scores')
fam = rows(ex(21)/'short_answers/results/families.csv')
lookup={(r['judge'],r['ratings_per_answer'],r['training_ratings'],r['method'],r['assignment']):r for r in fam}
errors=[]
for r in rows(ex(21)/'short_answers/results/contrasts.csv'):
    base=(r['judge'],r['ratings_per_answer'],r['training_ratings'])
    assignments=sorted({f['assignment'] for f in fam if (f['judge'],f['ratings_per_answer'],f['training_ratings'])==base})
    d=np.array([float(lookup[base+(r['comparator'],a)][r['metric']])-float(lookup[base+(r['subtract'],a)][r['metric']]) for a in assignments])
    lo=float(d.mean()-t.ppf(.975,9)*d.std(ddof=1)*np.sqrt(1/10+1/9))
    errors.append(abs(lo-float(r['lower_one_sided_975'])))
print('alpha=0.025:',len(errors),'lower bounds, max error',max(errors))
assert max(errors)<1e-9
print('E26 alpha=0.01 in analysis/recheck.py invocation and saved report; see g5 evidence for decision replication.')
cs=[r for r in rows(ex(29)/'results/contrasts.csv') if r['metric']=='rps' and r['target']=='ce_raw' and r['comparator']=='mse_raw']
panels=defaultdict(list)
for r in cs:panels[(r['study'],r['depth'])].append(r)
print('g7-M5: largest-budget MSE-trained minus CE-trained RPS with stored crossed simultaneous bounds')
for k,v in sorted(panels.items()):
    r=max(v,key=lambda v:int(v['b_total']))
    print(k,'B',r['b_total'],'RPS',r['diff_comparator_minus_target'],'CI',r['cross_sim_lo_95'],r['cross_sim_hi_95'])
print('g7-M6: E20 aggregate match is not a per-seed reconstruction')
x=json.loads((ex(20)/'results/summary_synth.json').read_text())
for k,v in x.items():
    if k.endswith('rps_excess') and (abs(v['diff_corr_minus_ce']-.0335003)<1e-6 or abs(v['diff_corr_minus_ce']+.0268253)<1e-6):
        print(k,v)
print('out_synth.json paths',list(ex(20).rglob('out_synth.json')))
print('g7-M7: searched prepared rating bundle basenames referenced by manifests')
for name in ['prepared_bundle.json','exp25_bundle.json','exp29_bundle.json']:
    print(name,[str(p.relative_to(ROOT)) for p in ROOT.rglob(name)])
print('g7-L1 current top-level numbered folders without reading guide:')
folders=sorted(p for p in (ROOT/'experiments').iterdir() if p.is_dir() and p.name[:2].isdigit())
missing=[p.name for p in folders if not (p/'HOW_TO_READ_RESULTS.md').exists()]
print(len(missing),'/',len(folders),missing)
print('The phase-1 baseline had 11/30 missing; current main adds experiment 34. Guides added during phase 2 alter the final count.')
