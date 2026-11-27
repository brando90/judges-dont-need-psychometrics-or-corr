import json, sys
sys.path.insert(0, '/tmp/qa55/code/exp36')
import write_reports as W
REPO = '/home/claude/judges-dont-need-psychometrics-or-corr/experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026'
payload = json.load(open(REPO + '/results.json'))
for fn, txt in (('objective_metric_matrix.md', W._matrix_md(payload)), ('crossover.md', W._curve_md(payload)), ('iso_budget.md', W._iso_md(payload))):
    committed = open(REPO + '/tables/' + fn).read()
    print(fn, 'IDENTICAL to regeneration from committed results.json' if committed.strip() == txt.strip() else 'DIFFERS')
summ = open(REPO + '/RESULTS_SUMMARY.md').read()
print('matrix block inside RESULTS_SUMMARY.md:', W._matrix_md(payload).strip() in summ)
print('curve block inside RESULTS_SUMMARY.md:', W._curve_md(payload).strip() in summ)
# POWER prose numbers vs results_power_mdd.json
pm = json.load(open(REPO + '/results_power_mdd.json'))
print(type(pm), list(pm.keys())[:10] if isinstance(pm, dict) else len(pm))
