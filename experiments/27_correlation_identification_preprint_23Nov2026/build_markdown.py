#!/usr/bin/env python3
"""Build the matching full Markdown article from the sectioned LaTeX source.

Uses only Python's standard library. Run from any working directory.
The source remains the LaTeX sections; this converter preserves displayed math,
proofs, references, and all prose and renders theorem labels and the one table.
"""
from pathlib import Path
import re

root=Path(__file__).resolve().parent
sections=sorted((root/'sections').glob('*.tex'))
source='\n\n'.join(p.read_text() for p in sections)
# Remove genuine LaTeX comments, preserving escaped percent signs in DOI links.
s='\n'.join(line for line in source.splitlines() if not line.lstrip().startswith('%'))
s=re.sub(r'(?<!\\)%[^\n]*','',s)
# Resolve the actual numbered equations without requiring compilation.
labels={}
count=0
for match in re.finditer(r'\\begin\{(equation|align)\}(.*?)\\end\{\1\}',s,re.S):
    env,body=match.group(1,2)
    rows=[body] if env=='equation' else re.split(r'\\\\(?:\[[^\]]*\])?',body)
    for row in rows:
        if not row.strip():
            continue
        count+=1
        for label in re.findall(r'\\label\{([^}]+)\}',row):
            labels[label]=str(count)
# Theorem numbering follows one shared proposition counter.
propcount=0
for match in re.finditer(r'\\begin\{(?:proposition|corollary)\}(?:\[[^\]]*\])?\s*(?:\\label\{([^}]+)\})?',s):
    propcount+=1
    if match.group(1):
        labels[match.group(1)]=str(propcount)
sec=0
for match in re.finditer(r'\\section\{[^}]+\}\s*(?:\\label\{([^}]+)\})?',s):
    sec+=1
    if match.group(1):
        labels[match.group(1)]=str(sec)
labels['tab:exact']='1'
# Add equation tags before stripping label directives.
def equation(m):
    env,body=m.group(1,2)
    if env=='align':
        rows=re.split(r'\\\\(?:\[[^\]]*\])?',body)
        rendered=[]
        for row in rows:
            if not row.strip():
                continue
            labs=re.findall(r'\\label\{([^}]+)\}',row)
            tag=r'\tag{'+labels[labs[-1]]+'}' if labs else ''
            rendered.append('\n$$\n'+r'\begin{aligned}'+row+r'\end{aligned}'+tag+'\n$$\n')
        return '\n'.join(rendered)
    labs=re.findall(r'\\label\{([^}]+)\}',body)
    tag=r'\tag{'+labels[labs[-1]]+'}' if labs else ''
    return '\n$$\n'+body+tag+'\n$$\n'
s=re.sub(r'\\begin\{(equation|align)\}(.*?)\\end\{\1\}',equation,s,flags=re.S)
s=re.sub(r'\\label\{[^}]+\}','',s)
s=re.sub(r'\\eqref\{([^}]+)\}',lambda m:'('+labels[m.group(1)]+')',s)
s=re.sub(r'\\ref\{([^}]+)\}',lambda m:labels[m.group(1)],s)
# The sole tabular display is rendered as a native Markdown table.
def table(m):
    block=m.group(0)
    cap=re.search(r'\\caption\{(.*?)\}\s*',block,re.S).group(1)
    return ('\n| Forecast | Pearson | TVD | Excess RPS | CE (nats) |\n'
            '|---|---:|---:|---:|---:|\n'
            '| True distribution $P$ | $1$ | $0$ | $0$ | $0.915387330$ |\n'
            '| Changed shape $Q$ | $1$ | $1/5$ | $3/200$ | $1.064735095$ |\n\n'
            '**Table 1.** '+cap+'\n')
s=re.sub(r'\\begin\{table\}.*?\\end\{table\}',table,s,flags=re.S)
s=s.replace(r'\begin{abstract}','## Abstract').replace(r'\end{abstract}','')
s=re.sub(r'\\section\{([^}]+)\}',r'## \1',s)
s=re.sub(r'\\subsection\{([^}]+)\}',r'### \1',s)
propcount=0
def theorem(m):
    global propcount
    propcount+=1
    return '\n**Proposition '+str(propcount)+' ('+m.group(1)+').**\n'
s=re.sub(r'\\begin\{proposition\}\[([^\]]*)\]',theorem,s)
s=s.replace(r'\end{proposition}','').replace(r'\begin{proof}','\n*Proof.*\n').replace(r'\end{proof}','\n$\\square$\n')
s=s.replace(r'\begin{enumerate}','').replace(r'\end{enumerate}','')
s=re.sub(r'^\\item\s*','1. ',s,flags=re.M)
s=s.replace(r'\[','\n$$\n').replace(r'\]','\n$$\n')
s=s.replace(r'\begin{thebibliography}{9}','## References').replace(r'\end{thebibliography}','')
s=s.replace(r'\bibitem{gneiting2007}','[1]').replace(r'\bibitem{epstein1969}','[2]')
s=s.replace(r'\cite{gneiting2007}','[Gneiting and Raftery (2007)](https://doi.org/10.1198/016214506000001437)')
s=s.replace(r'\cite{epstein1969}','[Epstein (1969)](https://doi.org/10.1175/1520-0450(1969)008%3C0985:ASSFPF%3E2.0.CO;2)')
# Expand paper macros inside Markdown math.
for name,repl in [('CE',r'\operatorname{CE}'),('KL',r'\operatorname{KL}'),('TV',r'\operatorname{TVD}'),('Var',r'\operatorname{Var}'),('Cov',r'\operatorname{Cov}'),('E',r'\mathbb{E}'),('R',r'\mathbb{R}'),('softmax',r'\operatorname{softmax}')]:
    s=re.sub(r'\\'+name+r'(?![A-Za-z])',lambda m,repl=repl:repl,s)
s=re.sub(r'\\texttt\{([^}]+)\}',lambda m:'`'+m.group(1).replace(r'\_','_')+'`',s)
s=re.sub(r'\\emph\{([^}]+)\}',r'*\1*',s)
s=re.sub(r'\\href\{([^}]+)\}\{([^}]+)\}',lambda m:'['+m.group(2)+']('+m.group(1).replace(r'\%','%')+')',s)
s=s.replace(r'\textless','<').replace(r'\textgreater','>')
s=s.replace(r'\bigskip','').replace(r'\noindent','').replace(r'\small','').replace(r'\copyright\ ','© ')
s=s.replace('~',' ').replace("``correlation''",'“correlation”').replace('--','–')
# Restore TeX operators damaged by no source convention; none use double hyphens.
s=re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+','\n\n',s)
s=re.sub(r'^(#{2,3}[^\n]+)\n(?!\n)',r'\1\n\n',s,flags=re.M)
head='# What Correlation Can and Cannot Identify in Human-Rating Judges\n\nBrando Miranda  \nStanford University  \nbrando9@stanford.edu  \n11-23-2026\n\n'
(root/'PAPER_11-23-2026.md').write_text(head+s.strip()+'\n')
print('Wrote PAPER_11-23-2026.md; resolved',len(labels),'labels.')
