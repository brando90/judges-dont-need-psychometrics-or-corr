"""QA g1/g2 MEDIUM/LOW evidence and metric panels (11-26-2026); CPU, read-only.
Run with an experiment number; emits dated evidence and a proper-score-first panel.
"""
import csv,json,sys,subprocess
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import numpy as np
from scipy.stats import kendalltau,spearmanr
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[3]
def exp(n): return next((ROOT/'experiments').glob(f'{n:02}_*'))
def js(n,f): return json.loads((exp(n)/f).read_text())
def raw(n): return [json.loads(s) for s in (exp(n)/'results/raw.jsonl').read_text().splitlines()]
def panel(rows,extra='Dataset / judge',excess=False):
 print('\n| '+extra+(' | RPS excess | CE excess = KL (nats)' if excess else ' | RPS | CE (nats)')+' | TVD | Spearman | Pearson |\n|---|---:|---:|---:|---:|---:|')
 for name,values in rows:
  print('| '+name+' | '+' | '.join(f'{v:.9f}' if isinstance(v,(int,float,np.floating)) else str(v) for v in values)+' |')
def gt(n,file): return subprocess.check_output(['git','log','--diff-filter=A','--format=%h %ad','--date=iso','--',str(exp(n).relative_to(ROOT)/file)],cwd=ROOT,text=True).strip()
def scores(p,q):
 pc=np.cumsum(p,axis=1)[:,:-1];qc=np.cumsum(q,axis=1)[:,:-1]; y=np.arange(6)
 return [float(((qc-pc)**2+pc*(1-pc)).mean()),float(-(p*np.log(q)).sum(1).mean()),float(.5*abs(p-q).sum(1).mean()),float(spearmanr(p@y,q@y).statistic),float(np.corrcoef(p@y,q@y)[0,1])]
n=int(sys.argv[1]);print(f'# Experiment {n:02} QA evidence (11-26-2026)\n\nDate: 11-26-2026. Read-only CPU reconstruction; no model calls.\n')
if n==1:
 trace=js(1,'results/synthetic_trace.json'); c=list(csv.DictReader((exp(1)/'results/candidate_rows.csv').open()));sel=list(csv.DictReader((exp(1)/'results/selection_rows.csv').open()))
 print('g1-M1: test trace records =',sum(r['split']=='test' for r in trace),'; candidate rows with sampled-test CE =',sum(bool(r['test_sample_ce']) for r in c))
 assert sum(r['split']=='test' for r in trace)==9 and len(c)==6912
 lookup={(r['replication'],r['ratings_per_item'],r['candidate']):r for r in c}; groups=defaultdict(list)
 for r in sel:
  selected=r['selected_candidates'].split('|'); vals=[]
  # Inspect the saved selection delimiter, retaining every tie member.
  if len(selected)==1 and ';' in selected[0]: selected=selected[0].split(';')
  for met in ('rps','ce','spearman','pearson'):
   v=np.mean([float(lookup[r['replication'],r['ratings_per_item'],j]['test_sample_'+met]) for j in selected]); assert abs(v-float(r['test_sample_'+met+'_tie_mean']))<1e-12; vals.append(v)
  groups[int(r['ratings_per_item']),r['objective']].append(vals)
 print('g1-L1: integration commits',*[subprocess.check_output(['git','show','-s','--format=%h %ad %s','--date=short',sha],cwd=ROOT,text=True).strip() for sha in ('3bfd866','18f529e')],sep='\n')
 print('g1-L4: historical command directory exists:',(ROOT/'experiments/02_conditional_binomial_distribution_identification').exists(),'; corrected directory exists:',exp(1).exists())
 print('g1-L5: selected sampled-test scores, 256 replications; equal average over ties. TVD not recorded for this target.')
 panel([(f'{m} ratings/item; {j} selected',list(np.mean(v,axis=0)[:2])+['not recorded']+list(np.mean(v,axis=0)[2:])) for (m,j),v in sorted(groups.items())])
elif n==2:
 d=js(2,'results/results.json');h=js(2,'historical_exp06/results/results.json')['primary'];t=js(2,'historical_exp06/training_followup_11-08-2026/results/results.json')
 avg=lambda z,j,k:float(np.mean([r[k] for r in z['per_seed'][j]]))
 print('g1-M2: CE-trained minus Pearson-trained CE from five per-partition scores =',avg(d,'ce','ce')-avg(d,'pearson','ce'))
 print('Saved refit interval (not independently reconstructed):',d['ce_minus_pearson_refit']['ce'])
 print('g1-M3: cached CE-selected minus Spearman-selected CE =',avg(h,'ce','ce')-avg(h,'spearman','ce'))
 print('g1-M3: weighted-trained minus Pearson-trained weighted loss =',avg(t,'mixture_weighted_pmf','weighted_pmf')-avg(t,'mixture_pearson','weighted_pmf'))
 print('Private dataset and per-bootstrap held-out scores are absent from this checkout; aggregate endpoints are not fresh quantile reconstructions.')
 print('g1-L6: held-out panels below precede separate training-correlation diagnostics; training proper-score panel unavailable in saved outputs. Held-out rows average five overlapping partitions of 75 artifacts, 195 ratings, 19 task families; not 375 independent artifacts.')
 rows=[]
 for label,z in [('Matched mixtures',d),('Historical cached selection',h),('Historical fitted heads',js(2,'historical_exp06/results/head_results.json')),('Historical training follow-up',t)]:
  for j,rr in z['per_seed'].items(): rows.append((label+' / '+j,[float(np.mean([r[k] for r in rr])) if k in rr[0] else 'not recorded' for k in ('rps','ce','tvd','spearman','pearson')]))
 panel(rows)
elif n==4:
 a=js(4,'results/phase_a.json');c=js(4,'results/phase_c.json'); s=c['sets']['test'];m=s['metrics'];a_j=s['most_repeatable_judge'];b_j=s['lowest_rps_judge'];print('g1-M4: aggregate RPS gap',a_j,'minus',b_j,'=',m[a_j]['RPS']['point']-m[b_j]['RPS']['point'])
 rel=s['order_by_reliability'];acc=s['order_by_rps'];print('Aggregate ranking Kendall tau =',kendalltau(range(len(rel)),[acc.index(j) for j in rel]).statistic)
 print('Private Phase C dataset missing; scores and 10,000-draw intervals remain aggregate-only.')
 p=np.array([[float(Fraction(v)) for v in r['p']] for r in a['A1']['per_artifact']]);rows=[('A1 population / truth',scores(p,p))]
 for name in ('A1','A1_max'):
  q=np.array([[float(Fraction(v)) for v in r['q']] for r in a[name]['per_artifact']]);rows.append((name+' population / forecast',scores(p,q)))
 for split,ss in c['sets'].items():
  for j,r in ss['metrics'].items():
   if j=='human':continue
   rows.append(('Phase C '+split+' / '+j,[r[k]['point'] if k in r else 'not recorded' for k in ('RPS','CE','TVD','spearman','pearson')]))
 print('g1-L7: A1 population scores recomputed from rational probabilities; Phase C table transcribed from aggregate JSON with provenance limitation above.')
 panel(rows)
elif n==5:
 a=js(4,'results/phase_a.json')['A1']['per_artifact'];p=np.array([[float(Fraction(v)) for v in r['p']] for r in a]);y=np.arange(6);m=p@y
 qm=[]
 for mu in m:
  lam=brentq(lambda l: (np.exp(l*y)@y)/np.exp(l*y).sum()-mu,-60,60);w=np.exp(lam*y);qm.append(w/w.sum())
 q=np.full_like(p,.001);q[:,0]+=(5-m)/5;q[:,5]+=m/5;q/=q.sum(1)[:,None]
 print('g1-L8: exact population distribution panel, 11 A1 artifacts; smoothed endpoints have six positive categories and affine-transformed means.')
 panel([(label,scores(p,v)) for label,v in [('True distribution',p),('Maximum entropy with matching mean',np.array(qm)),('Smoothed endpoint mixture',q)]])
elif n in (6,7,8):
 receipt=js(8,'results/ce_gradient_check.json')['exp08'];moves=receipt['ce_valrps_rows_that_moved']['by_chosen_ridge'];print('g2-M2: changed tuned fits',sum(moves.values()),'=',moves,'; converged',receipt['exact_ce_fits_scipy_converged'],'/',receipt['exact_ce_fits']);print('Historical individual refit outputs and earlier 1e-17 reproduction receipt absent; this verifies receipt arithmetic only.')
 if n==7: raise SystemExit(0)
 rows=raw(n);cells=defaultdict(list)
 if n==6:
  old={(r['scenario'],r['n'],r['seed']):r for r in (json.loads(l) for l in next((ROOT/'experiments/archive').glob('13_*/results/raw.jsonl')).read_text().splitlines())}
  for r in rows:
   v={j:old[r['scenario'],r['n'],r['seed']][j] for j in ('ce','rps','pearson')}|{j:r[j] for j in ('spearman','kendall','pearson_spearman')}
   for j,f in v.items():cells[r['scenario'],r['n'],j].append(f)
 else:
  for r in rows:
   for j,f in r['fits'].items():
    if isinstance(f,dict):cells[r['scenario'],r['n'],j].append(f)
  r=sorted([x for x in rows if x['scenario']=='D' and x['n']==4],key=lambda x:x['seed']);d=np.array([x['fits']['ce_valrps']['rps_excess']-x['fits']['pearson_fixed']['rps_excess'] for x in r]);ci=np.quantile(d[np.random.default_rng(0).integers(0,100,(10000,100))].mean(1),[.025,.975]);print('g2-M1: D n=4 tuned CE minus fixed Pearson RPS=',d.mean(),'95% independent bootstrap CI=',ci.tolist())
  for j in ('ce_valrps','pearson_fixed'): print(j,'RPS excess',np.mean([x['fits'][j]['rps_excess'] for x in r]))
 print('All saved scenarios, training sizes and judges: 100 seeds/cell, 500 held-out tasks/seed. RPS and CE are EXCESS over truth (CE excess = KL). TVD against truth was not recorded; tvd_from_constant has a different target and is not substituted. Correlation NaNs use the original zero convention.')
 panel([(f'{sc}, n={size}, {j}',[float(np.mean([f[k] for f in fs])) for k in ('rps_excess','ce_excess')]+['not recorded']+[float(np.nan_to_num([f.get(k,float('nan')) for f in fs],nan=0).mean()) for k in ('spearman','pearson')]) for (sc,size,j),fs in sorted(cells.items())], excess=True)
elif n==11:
 print('g2-M3: first README appearance:',gt(11,'README.md'));print('First raw sweep appearance:',gt(11,'results/raw.jsonl'));print('No earlier freeze/smoke receipt is preserved here. A current gradient check cannot establish historical pre-registration or historical maximum 2.41e-7.')
 # Run only the published smoke checks before expensive sweep/parity/timing sections.
 p=exp(11)/'scripts/smoke_check.py';sys.path.insert(0,str(p.parent));source=p.read_text().split('    # Check seed=0 reproduction')[0]+'\n';env={'__file__':str(p),'__name__':'qa_smoke_excerpt'};exec(compile(source,str(p),'exec'),env);env['main']()
elif n==15:
 d=js(15,'results/results.json');rr=d['judges_vs_target'];print('g2-M4: variance share from summaries=',1-d['s2_after_offsets']/d['s2_within_item']);print('Residual SD=',np.sqrt(d['s2_after_offsets']),'; rater-offset SD=',np.std([r['estimate'] for r in d['offsets'].values()],ddof=1));print('Counts from saved rater counts:',sum(d['rater_items'].values()),'; artifacts:',sum(d['ratings_per_item'].values()))
 print('g2-M5: saved compute runtime seconds =',d['seconds'],'; run.log exists =',(exp(15)/'results/run.log').exists(),'; launcher.log exists =',(exp(15)/'results/launcher.log').exists());print('Private ratings/judge pool and smoke/launch receipt absent; rating-level estimates, intervals and wall-clock boundaries remain unverifiable.')
 for met in ('rps','spearman','pearson'):
  a,b=('rps_vs_raw_hist_floor0.05','rps_vs_adj_hist_floor0.05') if met=='rps' else (met+'_raw',met+'_adj');print('g2-L3:',met,'ranking Kendall tau',kendalltau([r[a] for r in rr],[r[b] for r in rr]).statistic)
 print('All 44 cached judges, same 75 artifacts; rounded severity adjustment leaves every saved RPS unchanged. CE/TVD absent in this analysis; correlation raw -> adjusted follows RPS raw -> adjusted.')
 panel([(r['judge'],[f"{r['rps_vs_raw_hist_floor0.05']:.9f} → {r['rps_vs_adj_hist_floor0.05']:.9f}",'not recorded','not recorded',f"{r['spearman_raw']:.9f} → {r['spearman_adj']:.9f}",f"{r['pearson_raw']:.9f} → {r['pearson_adj']:.9f}"]) for r in rr])
