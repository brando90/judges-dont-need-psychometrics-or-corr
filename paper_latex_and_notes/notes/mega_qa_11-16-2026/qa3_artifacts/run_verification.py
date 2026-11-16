"""Stage 3 reproducibility checks; no measurement sweep or publication."""
from pathlib import Path
import hashlib,json,subprocess,time
root=Path.cwd();out=Path(__file__).resolve().parent
paper=root/'paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL'
exp=lambda n:next((root/'experiments').glob(n+'_*'))
arc=next((root/'experiments/archive').glob('13_*'))
commands=[
 ('rps_examples',['python3',str(paper/'rps_appendix_examples.py')],root),
 ('report06',['python3',str(exp('06')/'scripts/make_report.py')],root),
 ('report07',['python3',str(exp('07')/'scripts/make_report.py')],root),
 ('report08',['python3',str(exp('08')/'scripts/make_report.py')],root),
 ('plot13',['python3',str(arc/'scripts/plot_without_head_and_crossover.py')],root),
 ('paper_figure',['python3',str(exp('08')/'scripts/make_paper_figure.py')],root),
 ('migration',['python3','experiments/archive/verify_migration.py'],root),
 ('paper_build',['make'],paper),
 ('sentence_lines',['python3',str(Path.home()/'agents-config/scripts/latex_sentence_lines.py'),'--check']+[str(paper/f) for f in ['00_main_aaai_2027.tex','1-abstract.tex','2-introduction.tex','5b-learning-study.tex','6-case-study.tex','7-discussion.tex','8-appendix-rps.tex']],root),
 ('diff_check',['git','diff','--check'],root)
]
res={}
for name,cmd,cwd in commands:
 start=time.monotonic();p=subprocess.run(cmd,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (out/(name+'.txt')).write_text(p.stdout)
 res[name]={'command':cmd,'cwd':str(cwd),'exit_code':p.returncode,'seconds':round(time.monotonic()-start,2)}
 if name in ['report06','report07']:
  tab=exp(name[-2:])/'results/tables.md'
  baseline=subprocess.check_output(['git','show','HEAD:'+str(tab.relative_to(root))])
  res[name]['tables_byte_identical']=baseline==tab.read_bytes()
 print(name,json.dumps(res[name]),flush=True)
 (out/'verification_commands.json').write_text(json.dumps(res,indent=2)+'\n')
assert all(v['exit_code']==0 for v in res.values())
