"""Checks for the exp 18 reconciliation (11-21-2026): (a) our N15 / protocol C cells vs the Codex run's published paired_rps.csv; (b) direct, loop-based recomputation of the selectors on 12 splits.
Usage: python3 verify_reconcile.py --pool_json judge_forecasts.json --data dataset_pool.json --out out/full --codex_csv paired_rps.csv"""
import sys, csv
src = open('reconcile.py').read(); head = src[:src.index('# ---------------- in-sample diagnostics')]
i = sys.argv.index('--codex_csv'); codex = sys.argv[i+1]; sys.argv = sys.argv[:i] + sys.argv[i+2:] + ['--draws', '1']
exec(head)
res = json.load(open(OUT/'reconcile.json')); cells = {(c['pool'], c['proto'], c['n_train_units'], c['k']): c for c in res['cells']}
mx = 0
for row in csv.DictReader(open(codex)):
    if row['subtrahend_selector'] != 'ce': continue
    c = cells[('N15', 'C', int(row['train_families']), int(row['train_ratings_per_artifact']))]
    d = [abs(c['pearson_minus_ce']-float(row['pearson_selected_minus_other_test_rps'])), abs(c['ci_lo']-float(row['family_bayes_95_lo'])), abs(c['ci_hi']-float(row['family_bayes_95_hi']))]
    mx = max(mx, *d)
print('max abs difference to Codex paired_rps.csv (point, lo, hi, 6 budgets):', mx)
# (b) direct check: split 0..11, N15, n=3, k=1, per-judge loops
q = POOLS['N15']; J = q.shape[1]; bad = 0
for s, trs, te in splits_C():
    if s >= 12: break
    rng = np.random.default_rng(np.random.SeedSequence([20261121, 190, s])); perms = [rng.permutation(r) for r in ratings]
    tr = trs[3]; first = [p_[:1] for p_ in perms]
    ce_ = [np.mean([-np.log(q[i, j, first[i][0]]) for i in tr]) for j in range(J)]
    rps_te = lambda j: np.mean([np.mean([(q[i, j].cumsum()[c_] - (y <= c_))**2 for y in ratings[i] for c_ in range(5)]) for i in te])
    pe = [np.corrcoef(q[tr, j]@R, [first[i][0] for i in tr])[0, 1] if np.ptp(q[tr, j]@R) > 1e-12 and np.ptp([first[i][0] for i in tr]) > 0 else -9 for j in range(J)]
    j_ce, j_pe = int(np.argmin(ce_)), int(np.argmax(pe))
    obs = hists(first); tr_rps, tr_ce = losses(q, obs)
    c = select(np.ones((1, len(fam_names)))[:, fid[tr]], (q@R)[tr], (obs@R)[tr], tr_ce[tr], tr_rps[tr])[0]
    if (c[0], c[1]) != (j_ce, j_pe): bad += 1
    full_rps = losses(q, full)[0]
    assert abs(full_rps[te][:, j_pe].mean() - rps_te(j_pe)) < 1e-12 and abs(full_rps[te][:, j_ce].mean() - rps_te(j_ce)) < 1e-12
print('direct loop check, 12 splits: selector disagreements =', bad, '; test RPS from individual ratings matches histogram RPS to 1e-12')
