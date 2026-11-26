"""QA g2-H2: tuned Pearson head versus constant from raw seeds (11-26-2026)."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
p = next((ROOT/'experiments').glob('11_*'))
rows = [json.loads(l) for l in (p/'results/raw.jsonl').read_text().splitlines()]
judges = ['constant','ce_fixed','ce_valrps','pearson_fixed','pearson_head_naive','pearson_head_in_tuned','pearson_head_tuned','mse_fixed','mse_valrps','mse_head_tuned']
target_n, target_j = 32, 'pearson_head_tuned'
rng = np.random.default_rng(0)
# Reproduce only the required bootstrap RNG sequence, independently of make_report.py.
for n in sorted({r['n'] for r in rows}):
    cell = sorted([r['fits']|{'seed':r['seed']} for r in rows if r['scenario']=='A' and r['n']==n],key=lambda r:r['seed'])
    for j in judges:
        for _ in range(3): rng.integers(0,100,(10000,100))
        if j != 'constant':
            idx=rng.integers(0,100,(10000,100))
            if n==target_n and j==target_j:
                d=np.array([r[j]['rps_excess']-r['constant']['rps_excess'] for r in cell]); ci=np.quantile(d[idx].mean(1),[.025,.975])
                print('QA g2-H2 (11-26-2026): CONFIRMED')
                for name in ('pearson_head_tuned','constant'):
                    print(name, 'RPS excess',f"{np.mean([r[name]['rps_excess'] for r in cell]):.12f}", 'CE excess',f"{np.mean([r[name]['ce_excess'] for r in cell]):.12f}", 'TVD to truth: not recorded')
                print('Tuned Pearson head minus constant RPS:',f'{d.mean():.12f}', '95% seed-bootstrap interval:',*[f'{x:.12f}' for x in ci])
                saved=json.loads((p/'results/summary.json').read_text())['scenarios']['A']['cells'][str(n)][j]['minus_constant_rps']
                assert np.allclose([d.mean(),*ci],saved,rtol=0,atol=1e-14)
                assert ci[0]>0
                raise SystemExit(0)
        if j != 'ce_valrps': rng.integers(0,100,(10000,100))
raise AssertionError('target missing')
