#!/usr/bin/env python3
"""Import a small public aggregate receipt from exact committed sister-repo reports."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,required=True)
args=parser.parse_args()
old_commit='5aef572d30048e6ce574b39d91877c14ab9b3ad7'
active_commit='ea5d84238f6f27bf24ddc0359409921a2e55c459'
base='experiments/06_information_loss_vs_correlation_judge_selection/'
paths=[(old_commit,base+'RESULTS.md'),(old_commit,base+'followup_11-07-2026/RESULTS.md'),
       (active_commit,base+'training_followup_11-08-2026/RESULTS_training.md')]
sources=[]
texts=[]
for commit,path in paths:
    data=subprocess.check_output(['git','-C',str(args.repo),'show',commit+':'+path])
    texts.append(data.decode())
    sources.append({'commit':commit,'path':path,'sha256':hashlib.sha256(data).hexdigest(),
                    'url':f'https://github.com/brando90/trustworthy-theorem-judges/blob/{commit}/{path}'})
assert '8.868 for CE selection versus 8.098 for Spearman selection' in texts[0]
assert '+0.770 [-0.895, 1.317]' in texts[0]
assert 'Five seeded five-fold partitions reuse these same artifacts; they are not 375 independent artifacts.' in texts[0]
assert '| Historical validation selection / old test | 30 test artifacts / 10 tasks / 78 expert ratings | 9.133 | 8.710 | +0.423 |' in texts[0]
assert 'These old ratings, test outcomes, and candidate prompts had prior exposure. New family folds do not undo upstream selection or demonstration exposure.' in texts[0]
assert '215/225 Pearson fits did not converge, including 24/25 outer refits' in texts[0]
assert '+0.023822 [-0.036327, +0.053333]' in texts[1]
assert '+0.107807 [-0.128950, +0.171668]' in texts[1]
assert 'Status: COMPLETE, independently verified.' in texts[2]
assert '-0.010409 [-0.061911, +0.005933]' in texts[2]
assert '998/1,000 valid draws' in texts[2]
receipt={'scope':'public aggregate results only; no individual ratings, private candidates or sealed data imported',
 'sources':sources,'historical_counts':{'artifacts':75,'tasks':25,'families':19,'expert_ratings':195},
 'cached_selection':{'ce_selected_ce':8.868,'spearman_selected_ce':8.098,'difference':.770,'selection_refit_ci95':[-.895,1.317],
                     'evaluation_denominator':'5 seeded 5-fold family cross-validation partitions (25 outer folds; 375 pooled out-of-fold rows reusing 75 artifacts)',
                     'original_historical_split':{'validation_artifacts':15,'test_artifacts':30,'test_tasks':10,'test_expert_ratings':78,
                                                  'ce_selected_ce':9.133,'spearman_selected_ce':8.710,'difference':.423},
                     'prior_exposure':'historical ratings, test outcomes and candidate prompts/demonstrations had prior exposure; new family folds do not undo upstream exposure'},
 'fitted_heads':{'ce_fitted_ce':1.527,'pearson_fitted_ce':1.897,'difference':-.370,'conditional_ci95':[-.690,-.080],
                 'pearson_nonconverged_fits':215,'pearson_total_fits':225,'pearson_nonconverged_final_refits':24,'total_final_refits':25},
 'distance_followup':{'wasserstein_difference':.023822,'wasserstein_ci95':[-.036327,.053333],
                      'mmd2_difference':.107807,'mmd2_ci95':[-.128950,.171668]},
 'active_followup_status_at_pinned_commit':'COMPLETE: independently verified historical comparison; advantage unresolved',
 'weighted_training_followup':{'weighted_loss':.191659,'pearson_loss':.202068,'difference':-.010409,'refit_ci95':[-.061911,.005933],'valid_refit_draws':998,'attempted_refit_draws':1000,'primary_fits':50,'secondary_fits':900},
 'human_return_cloud_status':'not independently verified by this worker; do not declare fresh returns absent'}
(root/'results/public_ttj_evidence.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,indent=2))
