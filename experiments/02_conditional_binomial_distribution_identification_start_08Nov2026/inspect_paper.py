#!/usr/bin/env python3
"""Render every paper page and record source/PDF/toolchain hashes for inspection."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import pymupdf
root=Path(__file__).resolve().parent
paper=root.parents[1]/'paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL'
parser=argparse.ArgumentParser()
parser.add_argument('--render-dir',type=Path,required=True)
parser.add_argument('--confirmed-visual-inspection',action='store_true')
args=parser.parse_args()
args.render_dir.mkdir(parents=True,exist_ok=True)
pdf=paper/'00_main_aaai_2027.pdf'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
doc=pymupdf.open(pdf)
rendered=[]
for i,page in enumerate(doc,1):
    output=args.render_dir/f'page-{i}.png'
    page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(output)
    rendered.append({'page':i,'png_sha256':sha(output),'text_characters':len(page.get_text())})
log=(paper/'_build/00_main_aaai_2027.log').read_text(errors='replace')
errors=[line for line in log.splitlines() if line.startswith('!') or ('undefined' in line.lower() and ('citation' in line.lower() or 'reference' in line.lower()))]
assert not errors,errors
warnings=[line for line in log.splitlines() if 'Overfull' in line or 'Underfull' in line or 'Warning:' in line]
sources={p.name:sha(p) for p in paper.iterdir() if p.suffix in ['.tex','.bib','.sty','.bst'] or p.name=='Makefile'}
receipt={'pdf_sha256':sha(pdf),'pages':len(doc),'source_sha256':sources,'rendered_pages':rendered,
         'final_log_sha256':sha(paper/'_build/00_main_aaai_2027.log'),'final_log_errors':errors,'final_log_warnings':warnings,
         'visual_inspection':'all pages viewed by implementation owner; no clipping, overlap or unresolved citations observed' if args.confirmed_visual_inspection else 'pending all-page visual inspection',
         'engine':subprocess.check_output(['pdflatex','--version'],text=True).splitlines()[0],
         'render_tool':pymupdf.__doc__.splitlines()[0],
         'build_command':'make -C paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL',
         'build_note':'Bundled AAAI-27 requires pdfTeX. Makefile now creates _build and invokes pdfLaTeX/BibTeX; Tectonic is incompatible.'}
(root/'results/paper_build.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='source_sha256'},indent=2))
