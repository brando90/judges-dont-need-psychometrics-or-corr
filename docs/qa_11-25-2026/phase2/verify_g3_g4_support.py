"""QA phase 2 g3/g4 supporting checks (11-26-2026), read-only."""
from pathlib import Path
import csv,gzip,hashlib,json,re,subprocess
import numpy as np
R=Path(__file__).resolve().parents[3];E=lambda n:next((R/'experiments').glob(n+'_*'))
print('QA phase 2 g3/g4 supporting checks (11-26-2026)')
tracked=subprocess.check_output(['git','ls-files'],cwd=R,text=True).splitlines()
# Input absence establishes a limitation, not falsity of a historical number.
for label,tokens in [('historical individual ratings',['dataset.json','dataset_pool.json']),('synthetic E20 raw',['out_synth.json']),('public source prepared arrays',['koniq.npz','summeval_consistency.npz']),('original source archives',['.zip','.parquet'])]:
 hits=[p for p in tracked if any(p.endswith(t) for t in tokens)]
 print('Inventory',label,hits)
for n in ['09','10','16','17','18','19','20','21','22','23']:
 print('Experiment',n,'retained raw/receipt filenames:',[str(p.relative_to(E(n))) for p in E(n).rglob('*') if p.is_file() and any(t in p.name.lower() for t in ['raw','receipt','run.log','launcher.log','boot','token'])][:35])
a=json.loads((E('09')/'results/analysis_summary.json').read_text());branch=a['ii_c_real_weight_training']['pytorch_lora_end_to_end']
for model in ['q3_4b_2507','q25c_7b']:
 for arm in ['ce_lora_val_rps_3','pearson_lora_train_2']:
  r=branch[model]['methods'][arm];d=r['indiv_judge_mean_05']-r['indiv_human_mean_05'];assert abs(d-r['indiv_bias_05'])<1e-12
  print('g3-M1',model,arm,'individual bias',d,'artifact bias',r['bias_05'],'individual human mean',r['indiv_human_mean_05'])
m=a['representation_bridge']['exp09_verbal_6prob_pool']['methods'];print('g3-M2 aggregate RPS difference',m['corr_mix_pearson']['rps']-m['ce_mix_train']['rps'])
a=json.loads((E('10')/'results/phase1_sanity_and_ceilings.json').read_text());print('g3-M3 saved repeat SD',a['repeat_stability_k3_15items']['mean_expected_score_repeat_sd_0_1']);print('g3-L5 proper scores first', {k:a['label_in_prompt_reproduction'][k] for k in ['rps','ce','tvd','spearman','pearson']})
seconds=[];maxerr=0;num=0
for p in sorted((E('16')/'results').glob('*/results.json')):
 a=json.loads(p.read_text());b=json.loads((p.parent/'per_artifact.json').read_text())['per_artifact'];seconds.append(a['seconds'])
 for met in ['rps','ce','tvd']:
  for arm,v in b[met].items():maxerr=max(maxerr,abs(np.mean(v)-a['metrics'][met][arm]));num+=1
 if a['epsilon']==1e-6:print('g3-M4 saved failure counts',a['pool'],a['optimizer_certificate_failures'])
print('g3-M4 per-artifact proper-score means',num,'max error',maxerr,'seconds min/max/sum',min(seconds),max(seconds),sum(seconds));assert maxerr<1e-12
raw=gzip.decompress((E('17')/'results/raw.jsonl.gz').read_bytes());rows=[json.loads(l) for l in raw.splitlines()];print('g3-M5 cells',len({(r['scenario'],r['n'],r['seed']) for r in rows}),'sha256',hashlib.sha256(raw).hexdigest());assert len(rows)==4400
r=next(r for r in csv.DictReader((E('19')/'human_judges/bootstrap_sensitivity.csv').open()) if r['input']=='open_pair' and r['n']=='3' and r['k']=='1' and r['method']=='pearson' and r['metric']=='rps')
print('g3-M6 aggregate RPS difference',float(r['method_score'])-float(r['reference_score']),'saved draws fraction',int(r['draws_le_zero'])/int(r['draws']),'saved endpoints',r['percentile_lower_95'],r['percentile_upper_95'])
print('g4-M1 chronology quoted',*[line for line in (E('18')/'README.md').read_text().splitlines() if '12:20' in line or '12:05' in line]);print('g4-L1 reconciliation cells',len(json.loads((E('18')/'reconcile_single_judge_selection_21Nov2026/results/reconcile.json').read_text())['cells']))
print('g4-M2 first saved historical contrast',next(csv.DictReader((E('18')/'conditional_shape_and_rating_budget/historical_rating_budget/paired_rps.csv').open())))
a=json.loads((E('20')/'results/summary_synth.json').read_text());print('g4-M3 historical headline aggregate',a['mean_spread_shape|nr3|lam0.01|m16|rps_excess'])
manifest=[json.loads(l) for l in (E('21')/'short_answers/design/manifest.jsonl').read_text().splitlines()];print('g4-M4 saved manifest answers',len(manifest))
print('g4-M5 inventory arithmetic',350*123,sum([1162,16390,1314,4724,62]))
count=0;maxerr=0
for p in E('23').rglob('diagnostics.csv'):
 d=list(csv.DictReader(p.open()));s=list(csv.DictReader((p.parent/'scores.csv').open()));lookup={(r['depth'],r['budget'],r['method']):r for r in s}
 for r in d:assert (r['depth'],r['budget'],r['method']) in lookup
 draws=list(csv.DictReader((p.parent/'draw_rps.csv').open()))
 for r in s:
  if not draws or r['method'] not in draws[0]:continue
  vals=[float(v[r['method']]) for v in draws if v['depth']==r['depth'] and v['budget']==r['budget']]
  if vals:maxerr=max(maxerr,abs(np.mean(vals)-float(r['rps'])));count+=1
 print('g4-L5 matched diagnostic rows',str(p.relative_to(E('23'))),len(d))
print('g4-M6 per-draw RPS means',count,'max difference',maxerr);assert maxerr<1e-12
# Derived displays retain all diagnostic rows and source values.
for n in ['21','23']:
 for p in E(n).rglob('proper_scores_first_11-26-2026.md'):
  lines=[l for l in p.read_text().splitlines() if l.startswith('|')];header=lines[0];assert header.index('RPS')<header.index('CE (nats)')<header.index('TVD')<header.index('Spearman')<header.index('Pearson')
  d=list(csv.DictReader((p.parent/'diagnostics.csv').open()));assert len(lines)-2==len(d)
  for line,source in zip(lines[2:],d):
   vals=[x.strip() for x in line.strip('|').split('|')];assert vals[-2:]==[source['spearman'],source['pearson']]
print('Derived-view row counts, correlation identity and metric ordering verified.')
for n in ['18','20','22']:assert (E(n)/'HOW_TO_READ_RESULTS.md').is_file()
for n,rel in [('16','README.md'),('17','README.md'),('18','README.md'),('22','EVAL_PROTOCOL_11-21-2026.md')]:
 p=E(n)/rel;assert p.read_bytes()==subprocess.check_output(['git','show','4ce401c:'+str(p.relative_to(R))],cwd=R)
print('Preregisration/locked files unchanged; missing guides added.')
print('PASS. Aggregate-only checks do not replace unavailable original observations or execution receipts.')
