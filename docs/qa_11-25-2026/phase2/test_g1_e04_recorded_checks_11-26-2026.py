"""Assert experiment 04's recorded Boolean checks without overwriting outputs (11-26-2026)."""
import importlib.util,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3];p=next((root/'experiments').glob('04_*/scripts/test_metrics.py'))
sys.path.insert(0,str(p.parent));spec=importlib.util.spec_from_file_location('qa_e04_tests',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for name in ('test_losses','test_moments_and_components','test_d_study_by_covariance','test_ems_and_anova','test_reml','test_icc_matches_g_coefficients','test_serialization'):getattr(m,name)()
failures=[r for r in m.CHECKS if not r['pass']]
print('Experiment 04 recorded formula checks (11-26-2026):',len(m.CHECKS),'checks;',len(failures),'failures')
assert not failures,failures
