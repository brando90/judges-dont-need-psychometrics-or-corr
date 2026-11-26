"""QA phase 2 g4-H2 verification (11-26-2026): saved-score arithmetic only."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[3];e18=next((root/'experiments').glob('18_*'))
a=json.loads((e18/'real_n_sweep/results/all_eps5e-2_ntrain4/results.json').read_text())
m=a['metrics']['rps'];diff=m['corr_single']-m['ce_single'];interval=a['paired']['rps: corr_single minus ce_single']['ci95']
print('QA phase 2 g4-H2 (11-26-2026)')
print('RPS: CE-selected =',m['ce_single'],'Pearson-selected =',m['corr_single'])
print('Pearson-selected minus CE-selected RPS =',diff,'saved 95% interval =',interval)
assert diff>0 and interval[0]>0
print('CONFIRMED contradiction: never significantly worse has a saved CE-win exception.')
print('LIMIT: raw human ratings, per-artifact losses, and bootstrap draws absent; interval quoted, not regenerated.')
