"""Verify g7-H2: verbal versus scalar CE comparator (11-26-2026)."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
path = next((ROOT / "experiments").glob("09_*")) / "results/analysis_summary.json"
data = json.loads(path.read_text())["representation_bridge"]
print("g7-H2 verification (11-26-2026)")
print("Source:", path.relative_to(ROOT))
print("Evidence level: aggregate field attribution and subtraction; protected ratings and family losses absent.")
for pool in ["exp09_verbal_6prob_pool", "exp14_3sample_scalar_39judges"]:
    rows = data[pool]["methods"]
    for method in ["ce_mix_train", "corr_mix_pearson"]:
        print(pool, method, {m: rows[method][m] for m in ["rps", "ce", "tvd"]})
    print("Pearson-trained minus CE-trained:",
          {m: rows["corr_mix_pearson"][m] - rows["ce_mix_train"][m] for m in ["rps", "ce", "tvd"]})
assert round(data["exp09_verbal_6prob_pool"]["methods"]["corr_mix_pearson"]["ce"], 3) == 1.804
assert round(data["exp14_3sample_scalar_39judges"]["methods"]["ce_mix_train"]["ce"], 3) == 3.480
print("Verdict: CONFIRMED wrong comparator; verbal CE 1.629 vs 1.804, not 3.480 nats.")
