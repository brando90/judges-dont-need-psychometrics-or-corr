"""QA phase 2 g4-H3 verification (11-26-2026): saved-score arithmetic only."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[3];e18=next((root/'experiments').glob('18_*'))
a=json.loads((e18/'reconcile_single_judge_selection_21Nov2026/results/reconcile.json').read_text())
print('QA phase 2 g4-H3 (11-26-2026)')
for proto in ['C','M']:
 rows=[r for r in a['cells'] if r['pool']=='H5' and r['proto']==proto]
 print(proto,'positive H5 RPS differences:',sum(r['rps_pearson_selected']>r['rps_ce_selected'] for r in rows),'/',len(rows))
 for r in rows:
  if r['k']==1 and r['n_train_units'] in [3,4]:
   print(proto,'training units',r['n_train_units'],'RPS difference',r['rps_pearson_selected']-r['rps_ce_selected'],'saved fields',r)
print('CONFIRMED literal sign discrepancy; the cited intervals include zero, so this is not a statistically resolved reversal.')
print('LIMIT: aggregate arithmetic only; private rating-level inputs absent.')
