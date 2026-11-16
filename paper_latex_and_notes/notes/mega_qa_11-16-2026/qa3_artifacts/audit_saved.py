"""Read-only stage 3 audit of grids, frozen evidence and displayed quantities."""
from pathlib import Path
from collections import Counter
import json,itertools,subprocess,hashlib,re
import numpy as np
root=Path.cwd();out=Path(__file__).resolve().parent
exp={n:next((root/'experiments').glob(n+'_*')) for n in ['01','02','06','07','08']}
arc={n:next((root/'experiments/archive').glob(n+'_*')) for n in ['12','13','14']}
report={};raw={};ns=[4,8,16,20,32,64,128,256,512,1024,2048]
def nonfinite(v,p=''):
 if isinstance(v,dict):
  for k,t in v.items():yield from nonfinite(t,p+'/'+k)
 elif isinstance(v,float) and not np.isfinite(v):yield p
for n,sc,sizes,seeds in [('06','AB',ns,100),('07','AB',[4,8,16,32,64,128,256,512,1024],30),('08','ABCD',ns,100),('13','AB',ns,100)]:
 folder=(exp|arc)[n];rows=[json.loads(l) for l in (folder/'results/raw.jsonl').read_text().splitlines()];keys=[(r['scenario'],r['n'],r['seed']) for r in rows]
 assert len(set(keys))==len(rows) and set(keys)==set(itertools.product(sc,sizes,range(seeds)))
 raw[n]=dict(zip(keys,rows));missing=Counter(k for r in rows for k in nonfinite(r))
 assert all(any(c in k for c in ['pearson','spearman','kendall']) for k in missing)
 report[n]={'rows':len(rows),'unique_complete_grid':True,'nonfinite_fields':dict(missing)}
 if n!='13':
  tab=folder/'results/tables.md';before=subprocess.check_output(['git','show','HEAD:'+str(tab.relative_to(root))],text=True)
  oldrows=[l for l in before.splitlines() if l.startswith('|')];newrows=[l for l in tab.read_text().splitlines() if l.startswith('|')]
  assert oldrows==newrows;report[n]['unchanged_numeric_table_rows']=len(oldrows)
# All best starts were selected by their penalized training objective, not by test metrics.
for row in raw['07'].values():
 for kind,fit in row['fits'].items():
  if kind=='pearson_head':continue
  assert abs(fit['best']['train_obj']-min(fit['restart_train_obj']))<1e-12
  if kind=='ce':assert len(fit['restart_train_obj'])==1
  else:assert len(fit['restart_train_obj'])==5
orig=[r['fits']['pearson']['orig'] for r in raw['07'].values() if r['n']>=32];long=[r['fits']['pearson']['long'] for r in raw['07'].values() if r['n']>=32]
report['restart_selection']={'all_saved_best_starts_match_training_minimum':True,'pearson_n_ge32_fits':len(orig),'orig_converged':sum(r['success'] for r in orig),'long_hit_budget':sum(r['budget_hit'] for r in long)}
report['fallback_rows']=[list(k) for k,r in raw['08'].items() if r['fits']['pearson_valcorr'].get('fallback',False)]
report['pre_amendment_rows_without_fallback_field']=sum('fallback' not in r['fits']['pearson_valcorr'] for r in raw['08'].values())
assert report['fallback_rows']==[['C',4,9]]
shared=[]
for k,r in raw['08'].items():
 if k[0] not in 'AB':continue
 for met in ['rps_excess','ce_excess','pearson','spearman']:
  shared.append(abs(r['fits']['ce_fixed'][met]-raw['13'][k]['ce'][met]))
report['shared_ce08_vs_ce13_max_absolute_difference']=max(shared)
# Historical human-data chart numbers, checked directly against actual tracked JSON locations.
h=(root/'paper_latex_and_notes/notes/correlation_vs_loss_judges_one_pager/judge-objectives-status.html').read_text()
section=h[h.index('var rows = ['):h.index('var svg = document.getElementById("forest")')]
chart=[]
for label,d,lo,hi,a,b in re.findall(r'label: "([^"]+)", d: ([\d.-]+), lo: ([\d.-]+), hi: ([\d.-]+), a: ([\d.-]+), b: ([\d.-]+)',section):chart.append(dict(label=label,d=float(d),lo=float(lo),hi=float(hi),a=float(a),b=float(b)))
assert len(chart)==7
pairs=[(1,arc['12']/'results/results.json','corr_mixture_pearson','ce_mixture'),(2,arc['14']/'results/open/results.json','corr_mixture_pearson','ce_mixture'),(3,arc['14']/'results/all/results.json','corr_mixture_pearson','ce_mixture'),(4,arc['12']/'results/results.json','corr_single','ce_single'),(5,arc['14']/'results/open/results.json','corr_single','ce_single'),(6,arc['14']/'results/all/results.json','corr_single','ce_single')]
for idx,path,a,b in pairs:
 saved=json.loads(path.read_text());pair=saved['paired'][f'rps: {a} minus {b}'];r=chart[idx]
 assert all(abs(x-y)<5.01e-5 for x,y in zip([r['d'],r['lo'],r['hi'],r['a'],r['b']],[pair['point'],*pair['ci95'],saved['metrics']['rps'][a],saved['metrics']['rps'][b]]))
# Exp02 chart reverses CE-minus-Pearson sign and interval endpoints from the tracked comparison.
t=(exp['02']/'RESULTS_comparison.md').read_text();line=next(l for l in t.splitlines() if l.startswith('| rps |'))
nums=[float(v) for v in re.findall(r'[+-]?\d+\.\d+',line)];r=chart[0]
assert abs(r['d']+nums[6])<1e-9 and abs(r['lo']+nums[8])<1e-9 and abs(r['hi']+nums[7])<1e-9
report['html_forest_rows_verified']=len(chart)
saved=json.loads((exp['01']/'results/summary.json').read_text());report['selection01']={}
for m in ['1','2','3']:
 report['selection01'][m]={k:{'true_model_selection_count':int(round(saved[m][k]['metrics']['oracle_uniform_tie_probability']['mean']*256)),'mean_test_ce':saved[m][k]['metrics']['test_expected_ce_tie_mean']['mean']} for k in ['ce','rps']}
assert report['selection01']['1']['ce']['true_model_selection_count']==249
assert report['selection01']['1']['rps']['true_model_selection_count']==245
assert '<span class="num">1.331</span> nats choosing by CE' in h
# Exact-CE diagnostic denominator and single exception, independent of prose.
d=json.loads((exp['08']/'results/ce_gradient_check.json').read_text());x=d['exp08']
report['ce_diagnostic']={'quantities':sum(x['reported_quantities_compared'].values()),'changed_quantities':{k:v for k,v in x['reported_quantities_that_change'].items() if v},'exact_fits':x['exact_ce_fits'],'converged':x['exact_ce_fits_scipy_converged']}
assert report['ce_diagnostic']['quantities']==424
assert list(report['ce_diagnostic']['changed_quantities'])==['tuned_ce_ridge_mode']
# Frozen files and read-only appendix: compare actual bytes against starting HEAD.
paths=[]
for n in ['06','07','08']:
 folder=exp[n]
 paths += [folder/'README.md',folder/'results/raw.jsonl',folder/'results/run_log.txt']
 paths += [p for p in (folder/'scripts').glob('*.py') if p.name not in ['make_report.py','make_paper_figure.py','check_ce_gradient.py']]
paths += [root/'paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/9-appendix-reliability.tex']
allowed={'HOW_TO_READ_RESULTS.md','scripts/plot_without_head_and_crossover.py','fig_excess_vs_n_without_head.png','fig_excess_vs_n_with_head_labeled.png'}
tracked=subprocess.check_output(['git','ls-files',str(arc['13'].relative_to(root))],text=True).splitlines()
paths += [root/p for p in tracked if str((root/p).relative_to(arc['13'])) not in allowed]
report['frozen_files_unchanged']=[]
for p in paths:
 rel=str(p.relative_to(root));assert p.read_bytes()==subprocess.check_output(['git','show','HEAD:'+rel]),rel
 report['frozen_files_unchanged'].append(rel)
assert (root/'AGENTS.md').read_bytes()==(root/'CLAUDE.md').read_bytes()
report['AGENTS_CLAUDE_identical']=True
(out/'saved_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
