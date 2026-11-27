"""Compare the Pearson-trained head's raw test nRPS (from the repo JSON) with the test nRPS of the UNIFORM forecast, seed 0..4."""
import sys, json
from pathlib import Path
COPY = Path("/tmp/opus_review_exp40/copy/experiments/40_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026")
sys.path.insert(0, str(COPY / "scripts"))
import numpy as np
from e40_common import d, fe, three_level
J = json.loads((COPY / "results/k_sweep_matched_calibration_maxiter35.json").read_text())
rows = {(r["view"], r["seed"]): r for r in J["rows"]}
pop_off, pop_pol = d.load_popquorn("offensiveness"), d.load_popquorn("politeness")
se_e, se_c = d.load_summeval("expert"), d.load_summeval("crowd")
d990 = d._load_dices("990"); ch3 = d.load_chaos_3way(); cha = d.load_chaos_alpha()
views = []
for b in (pop_off, pop_pol, se_e, se_c):
    views += [(b.name + "|K2", d.binarize(b, 4, "x")), (b.name + "|K3", three_level(b, "x")), (b.name + "|K5", b)]
views += [("dices990|K2", d.binarize(d990, 2, "x")), ("dices990|K3", d990), ("chaosnli_snli_mnli|K3", ch3), ("chaosnli_alphanli|K2", cha)]
for vname, data in views:
    k = data.n_classes; values = np.asarray(data.class_values, float)
    u, p = [], []
    for s in range(5):
        tr, va, te = d.split_by_group(data.groups, seed=s)
        q = np.full((len(te), k), 1.0 / k)
        rec = fe._item_records(q, q @ values, data.ratings, te, values)
        u.append(rec["rps"].mean() / (k - 1)); p.append(rows[(vname, s)]["mean"]["pearson"]["raw"])
    print(f"{vname:28s} uniform {np.mean(u):.4f}  pearson_raw {np.mean(p):.4f}  diff {np.mean(p)-np.mean(u):+.4f}")
