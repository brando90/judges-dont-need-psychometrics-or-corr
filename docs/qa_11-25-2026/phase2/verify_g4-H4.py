"""QA phase 2 g4-H4 verification (11-26-2026): retained summary contradictions."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[3];e20=next((root/'experiments').glob('20_*'))
a=json.loads((e20/'results/summary_synth.json').read_text())
s=a['mean_spread_shape|nr3|lam0.01|m8|rps_excess']
print('QA phase 2 g4-H4 (11-26-2026)')
print('8 items, 3 ratings, mean/spread/shape: RPS-excess difference Pearson-trained minus CE-trained',s)
assert s['diff_corr_minus_ce']>0 and s['lo']<0<s['hi']
rows=json.loads((e20/'results/out_real_lam0.1.json').read_text())['rows']
rows=[r for r in rows if r['pool']=='closed_3vote_5' and r['n_train_families']==3 and r['ratings_per_artifact']=='all']
ce=next(r for r in rows if r['arm']=='head_ce');corr=next(r for r in rows if r['arm']=='head_corr');d=next(r for r in rows if r['arm']=='diff_head_corr_minus_ce')
print('Closed 5-judge pool, 3 families, all ratings: RPS CE head',ce['rps'],'Pearson head',corr['rps'],'recomputed difference',corr['rps']-ce['rps'])
print('Saved contrast:',d)
assert corr['rps']-ce['rps']>0
print('CONFIRMED blanket loss claim contradicted. LIMIT: synthetic per-seed and historical per-split outputs absent; saved intervals not regenerated.')
