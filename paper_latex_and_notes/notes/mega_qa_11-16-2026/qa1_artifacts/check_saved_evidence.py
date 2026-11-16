"""Read-only QA of saved grids, source preservation, table numbers and paper hygiene."""
from pathlib import Path
from collections import Counter
import hashlib,json,subprocess,re,ast,itertools
import numpy as np
import fitz
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent
report={}
def git_bytes(p):return subprocess.check_output(['git','show','233d504:'+p.as_posix()])
ns=[4,8,16,20,32,64,128,256,512,1024,2048]
raw={}
for exp,scenes,sizes,seeds in [('06','AB',ns,100),('07','AB',[4,8,16,32,64,128,256,512,1024],30),('08','ABCD',ns,100)]:
 d=next(Path('experiments').glob(exp+'_*'));p=d/'results/raw.jsonl'
 rows=[json.loads(s) for s in p.read_text().splitlines()];keys=[(r['scenario'],r['n'],r['seed']) for r in rows]
 assert len(keys)==len(set(keys));assert set(keys)==set(itertools.product(scenes,sizes,range(seeds)))
 raw[exp]=rows
 report[exp]={'rows':len(rows),'complete_unique_grid':True}
 def recurse(x,path=''):
  if isinstance(x,dict):
   for k,v in x.items():yield from recurse(v,path+'.'+k)
  elif isinstance(x,float) and not np.isfinite(x):yield path
 missing=Counter(k for r in rows for k in recurse(r))
 report[exp]['nonfinite_fields_retained']=dict(missing)
 assert all('pearson' in k or 'spearman' in k or 'kendall' in k for k in missing)
 preserved=[]
 for q in [p,d/'results/run_log.txt',d/'README.md']+list((d/'scripts').glob('*.py')):
  if q.name.startswith('make_'):continue
  assert q.read_bytes()==git_bytes(q),q
  preserved.append(q.as_posix())
 report[exp]['frozen_files_unchanged']=preserved
 tab=d/'results/tables.md';old=[l for l in git_bytes(tab).decode().splitlines() if l.startswith('|')];new=[l for l in tab.read_text().splitlines() if l.startswith('|')]
 assert old==new;report[exp]['numeric_table_rows_identical_to_baseline']=len(old)
 for q in (d/'scripts').glob('*.py'):ast.parse(q.read_text())
rows=raw['07'];original=[r['fits']['pearson']['orig'] for r in rows if r['n']>=32];long=[r['fits']['pearson']['long'] for r in rows if r['n']>=32]
report['pearson_convergence']={'orig_n_ge32_success':sum(r['success'] for r in original),'denominator':len(original),'long_budget_hit':sum(r['budget_hit'] for r in long)}
assert not any(r['success'] for r in original)
for r in rows:
 for obj,d in r['fits'].items():
  if obj=='pearson_head':continue
  assert d['best']['train_obj']<=d['long']['train_obj']+1e-12
  assert abs(d['best']['train_obj']-min(d['restart_train_obj']))<1e-12
report['restart_training_selection_verified']=True
report['fallback_rows']=[{k:r[k] for k in ('scenario','n','seed')} for r in raw['08'] if r['fits']['pearson_valcorr'].get('fallback',False)]
assert report['fallback_rows']==[{'scenario':'C','n':4,'seed':9}]
assert Path('AGENTS.md').read_bytes()==Path('CLAUDE.md').read_bytes();report['AGENTS_CLAUDE_identical']=True
p=Path('paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL');log=(p/'_build/00_main_aaai_2027.log').read_text()
errs=[l for l in log.splitlines() if l.startswith('!') or 'undefined' in l.lower() or 'Overfull' in l]
assert not errs,errs
report['tex_errors_undefined_overfull']=errs;report['underfull_count']=log.count('Underfull')
pdf=fitz.open(p/'00_main_aaai_2027.pdf');text='\n'.join(page.get_text() for page in pdf)
report['pages']=len(pdf);report['references_pages']=[i+1 for i,page in enumerate(pdf) if '\nReferences\n' in page.get_text()]
patterns=['brando','miranda','trustworthy-theorem','github.com','stanford','koyejo']
report['pdf_identifying_hits']=[x for x in patterns if x in text.lower()];assert not report['pdf_identifying_hits']
sources=list(p.glob('*.tex'))+[p/'00_main.bib']
report['source_identifying_hits']=[(q.name,x) for q in sources for x in patterns if x in q.read_text().lower()];assert not report['source_identifying_hits']
for i,page in enumerate(pdf):page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(OUT/f'after_page_{i+1}.png')
f=fitz.open(p/'figures/fig_ratings_needed.pdf');f[0].get_pixmap(matrix=fitz.Matrix(3,3)).save(OUT/'paper_figure.png')
checks={}
for q in sorted(p.glob('*.tex')):
 r=subprocess.run(['python3',str(Path.home()/'agents-config/scripts/latex_sentence_lines.py'),'--check',str(q)],capture_output=True,text=True)
 checks[q.name]={'returncode':r.returncode,'output':r.stdout+r.stderr}
report['sentence_line_checks']=checks
assert all(v['returncode']==0 for v in checks.values()),checks
r=subprocess.run(['git','diff','--check'],capture_output=True,text=True);report['git_diff_check']={'returncode':r.returncode,'output':r.stdout+r.stderr};assert r.returncode==0
report['verification_kind']='deterministic Python/numerical checks and model reading; no Lean; no external reviewer dispatch'
(OUT/'saved_evidence_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
