"""Matched calibration control for method 2 (Pearson-trained) vs method 4 (CE/RPS-trained) (M+K) heads.
For each head objective, using the committed per-fold/exp27 selected HPs (from final_results JSON):
  1) cross-fit: 4-fold inner OOF predictions of the head on the TRAINING subset only (task-grouped, their make_4fold_inner_tasks)
  2) fit ONE recalibration-map family (1-dist logit pool on the head's own centered log-probs + E[q] linear/quad,
     CE-fitted, alpha=1e-2, no Dirichlet mix) on those OOF predictions vs training ratings
  3) fit the head on the whole training subset, predict held-out, apply its map.
Same map family, same data, same procedure for every objective. Prints raw and recalibrated held-out metrics and
paired task-cluster bootstrap differences (10,000 draws, seed 20261126).
Usage: python3 calib.py <copy of experiments/28 dir (scratch copy, scripts importable)>"""
import json, sys
from pathlib import Path
import numpy as np
E = Path(sys.argv[1]); sys.path.insert(0, str(E / "scripts"))
sys.path.insert(0, str(E.parent / "27_build_a_judge_that_beats_the_constant_forecast_on_held_out_tasks_old_new_ratings_start_24Nov2026" / "scripts"))
import data_exp28 as de, distlib as dl
from search_inner_cv import build_lp_and_x, make_4fold_inner_tasks
from final_eval_exp28 import slice_bundle, eval_mk_model
J = json.loads((E / "results/final_results_11-26-2026.json").read_text())
D = de.load_bundle(include_test=True, require_complete=True)
MAPOBJ = sys.argv[2] if len(sys.argv) > 2 else "ce"

def head_fit_predict(Dfull, spec, tr, te):
    return eval_mk_model(Dfull, spec, tr, te)

def recal_fit(q_oof, P):
    lp = dl.centered_log_probs(q_oof)[:, None, :]
    x = np.column_stack([q_oof @ dl.GRID])
    return dl.MKCalibrator(objective=MAPOBJ, alpha=1e-2, alpha_a=0.05, dirichlet_eta=0.0, use_quad=True,
                           set_aware_prior=False).fit(lp, x, np.zeros(len(P)), P)

def recal_apply(cal, q):
    return cal.predict(dl.centered_log_probs(q)[:, None, :], np.column_stack([q @ dl.GRID]), np.zeros(len(q)))

def cross_fit_map(spec, tr_mask):
    Dsub = slice_bundle(D, tr_mask)
    inner = make_4fold_inner_tasks(Dsub)
    q_oof = np.zeros((Dsub["n"], 6))
    for f in range(4):
        q_oof[inner == f] = eval_mk_model(Dsub, spec, inner != f, inner == f)
    return recal_fit(q_oof, Dsub["P"])

def run(regime, folds):
    raw = {o: np.zeros((D["n"], 6)) for o in ("ce", "rps", "pearson")}
    cal = {o: np.zeros((D["n"], 6)) for o in ("ce", "rps", "pearson")}
    held = np.zeros(D["n"], bool)
    for tr, te, specs in folds:
        held |= te
        for o in ("ce", "rps", "pearson"):
            spec = specs[o]
            q = head_fit_predict(D, spec, tr, te)
            raw[o][te] = q
            cal[o][te] = recal_apply(cross_fit_map(spec, tr), q)
    T, P = D["task"][held], D["P"][held]
    print(f"\n=== {regime}: n_items={held.sum()} n_tasks={len(set(T))}  (map objective={MAPOBJ}) ===")
    for o in ("ce", "rps", "pearson"):
        a = dl.evaluate_panel(P, raw[o][held]); b = dl.evaluate_panel(P, cal[o][held])
        print(f"  {o:8s} raw: RPS={a['RPS']:.4f} CE={a['CE']:.4f} TVD={a['TVD']:.4f} Bias={a['Bias']:+.3f} r={a['Pearson']:.3f} | "
              f"recal: RPS={b['RPS']:.4f} CE={b['CE']:.4f} TVD={b['TVD']:.4f} Bias={b['Bias']:+.3f} r={b['Pearson']:.3f}")
    for lab, A, B in (("CE-trained minus Pearson-trained", "ce", "pearson"), ("RPS-trained minus Pearson-trained", "rps", "pearson")):
        for kind, Q in (("raw", raw), ("recal", cal)):
            r = dl.task_cluster_bootstrap(T, P, Q[A][held], q_ref=Q[B][held], draws=10000, seed=20261126)["paired_vs_ref"]
            print(f"  [{kind}] {lab}: dRPS={r['dRPS']['point']:+.4f} [{r['dRPS']['ci95'][0]:+.4f},{r['dRPS']['ci95'][1]:+.4f}] p={r['dRPS']['p_val_two_sided']:.4f} | "
                  f"dCE={r['dCE']['point']:+.4f} [{r['dCE']['ci95'][0]:+.4f},{r['dCE']['ci95'][1]:+.4f}] p={r['dCE']['p_val_two_sided']:.4f} | dTVD={r['dTVD']['point']:+.4f} p={r['dTVD']['p_val_two_sided']:.4f}")
    return raw

# exp27 split, C7 head, committed selected HPs
tr27 = D["exp27_split"] != "test"; te27 = ~tr27
hp27 = J["exp27_head_matched"]["selected_hps"]
raw27 = run("exp27_test C7 head (train+val 30 tasks -> test 20 tasks)", [(tr27, te27, hp27)])
# check raw reproduces committed
for o in ("ce", "rps", "pearson"):
    c = J["exp27_head_matched"][o]["point"]["RPS"]; print(f"  check exp27 raw {o}: committed RPS={c:.4f}")
for lab, key in (("cv5 C7 head", "head_matched_cv5_c7"), ("cv5 G9 head", "head_matched_cv5_g9")):
    hps = J[key]["selected_hps_by_fold"]
    folds = [(D["cv5_fold"] != f, D["cv5_fold"] == f, {o: hps[f][o] for o in ("ce", "rps", "pearson")}) for f in range(5)]
    run(f"{lab} (outer 5-fold, 50 tasks)", folds)
    for o in ("ce", "rps", "pearson"):
        print(f"  check {lab} raw {o}: committed RPS={J[key][o]['point']['RPS']:.4f}")
