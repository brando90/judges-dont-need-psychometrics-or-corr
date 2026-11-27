"""Symmetric recalibration diagnostic for Qwen3-4B LoRA heads (CE/RPS/Pearson). No LoRA validation predictions
were committed, so a validation-fitted map is impossible. Two DIAGNOSTIC controls, applied identically to all heads:
 (a) cv5: map for fold f fitted on the committed OOF predictions + ratings of the other 4 folds (cross-model map);
 (b) exp27 test: leave-one-task-out map fitted on the other 19 test tasks (uses test labels; diagnostic only).
Map family: same as matched_calibration_control.py (1-dist logit pool + E[q] lin/quad, CE objective, alpha=1e-2).
Usage: python3 lora_symmetric_recal.py <scratch E28 copy>"""
import json, sys
from pathlib import Path
import numpy as np
E = Path(sys.argv[1]); sys.path.insert(0, str(E / "scripts"))
import data_exp28 as de, distlib as dl
D = de.load_bundle(include_test=True, require_complete=True)
L = json.loads((E / "results/qwen3_4b_lora_ce_rps_pearson.json").read_text())
def fit(q, P):
    return dl.MKCalibrator(objective="ce", alpha=1e-2, alpha_a=0.05, dirichlet_eta=0.0, use_quad=True, set_aware_prior=False).fit(
        dl.centered_log_probs(q)[:, None, :], np.column_stack([q @ dl.GRID]), np.zeros(len(P)), P)
def app(c, q): return c.predict(dl.centered_log_probs(q)[:, None, :], np.column_stack([q @ dl.GRID]), np.zeros(len(q)))
def report(lab, T, P, raw, cal):
    print(f"\n=== {lab}: n={len(T)} tasks={len(set(T))} ===")
    for o in raw:
        a = dl.evaluate_panel(P, raw[o]); b = dl.evaluate_panel(P, cal[o])
        print(f"  {o:8s} raw RPS={a['RPS']:.4f} CE={a['CE']:.3f} Bias={a['Bias']:+.3f} r={a['Pearson']:.3f} | recal RPS={b['RPS']:.4f} CE={b['CE']:.3f} Bias={b['Bias']:+.3f}")
    for A in ("ce", "rps"):
        for kind, Q in (("raw", raw), ("recal", cal)):
            r = dl.task_cluster_bootstrap(T, P, Q[A], q_ref=Q["pearson"], draws=10000, seed=20261126)["paired_vs_ref"]["dRPS"]
            print(f"  [{kind}] {A.upper()}-trained minus Pearson-trained LoRA, test RPS: {r['point']:+.4f} [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] p={r['p_val_two_sided']:.4f}")
# (a) cv5
items = D["item"]; raw = {}; cal = {}
for o in ("ce", "rps", "pearson"):
    q = np.stack([np.asarray(L[o]["cv5_oof_preds"][it]["probs"], float) for it in items]); raw[o] = q; cal[o] = np.zeros_like(q)
    for f in range(5):
        te = D["cv5_fold"] == f
        cal[o][te] = app(fit(q[~te], D["P"][~te]), q[te])
report("cv5 OOF LoRA, cross-fold map (diagnostic)", D["task"], D["P"], raw, cal)
# (b) exp27 test LOTO
te = D["exp27_split"] == "test"; T = D["task"][te]; P = D["P"][te]; raw = {}; cal = {}
for o in ("ce", "rps", "pearson"):
    q = np.stack([np.asarray(L[o]["exp27_test_preds"][it]["probs"], float) for it in items[te]]); raw[o] = q; cal[o] = np.zeros_like(q)
    for t in sorted(set(T)):
        m = T == t; cal[o][m] = app(fit(q[~m], P[~m]), q[m])
report("exp27 test LoRA, leave-one-test-task-out map (uses test labels; diagnostic)", T, P, raw, cal)
# constant reference
print("\nset-aware constant test RPS (committed): exp27", )
