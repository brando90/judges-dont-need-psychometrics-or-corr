#!/usr/bin/env python3
"""Reproduce historical code and store a new receipt without modifying its folder."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='8'
root=Path(__file__).resolve().parent
old=root.parent/'01_synthetic_pearson_vs_kl_squashed_judge_0to5_start_07Nov2026_end_07Nov2026/kl_vs_corr.py'
result=subprocess.run([sys.executable,str(old)],text=True,capture_output=True,check=True)
(root/'results/old_toy_stdout.txt').write_text(result.stdout)
expected=['A: noisy faithful      pearson=0.900 spearman=0.900 KL(P_h||P_j)= 0.001 nMSE=0.023',
          'B: inflated/squashed   pearson=0.877 spearman=0.878 KL(P_h||P_j)= 3.420 nMSE=0.216']
assert result.stdout.splitlines()[:2]==expected
receipt={'exit_code':result.returncode,'source':str(old.relative_to(root.parent.parent)),
         'source_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),
         'stdout_sha256':hashlib.sha256(result.stdout.encode()).hexdigest(),
         'historical_first_two_lines_match':True,
         'interpretation':'Marginal histograms only; two-bin compression introduces ties and is not strictly order preserving. It is not conditional judge selection.'}
(root/'results/old_toy_reproduction.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(result.stdout)
