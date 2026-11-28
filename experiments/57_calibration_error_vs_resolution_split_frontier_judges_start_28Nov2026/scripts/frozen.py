"""Frozen-bundle loader shared by Experiments 57-60 (owned by Experiment 57; 58, 59 and 60 import it by path).

Reads experiments/39_*/data/processed/study_a_bundle.npz through experiments/44_*/src/lib44.py (invalid judge forecasts -> uniform,
all forecasts floored at 1e-6) and exposes the five frozen frontier judges on the five public multi-annotator rating sets.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src" / "judge_scores"))
_spec = importlib.util.spec_from_file_location("lib44", next((REPO / "experiments").glob("44_*/src/lib44.py")))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

SEED = 20261128
DATASETS = list(L.DATASETS)  # chaosnli_snli, chaosnli_mnli, chaosnli_alphanli, dices350, dices990
JUDGES = list(L.JUDGES)  # gpt-6-astra, claude-opus-5-5, kimi-k3, glm-5.3, deepseek-v4.1-flash
SHORT = {"gpt-6-astra": "Astra", "claude-opus-5-5": "Opus", "kimi-k3": "Kimi", "glm-5.3": "GLM", "deepseek-v4.1-flash": "DeepSeek"}
PRIMARY = {"chaosnli_snli": "brier", "chaosnli_mnli": "brier", "chaosnli_alphanli": "brier", "dices350": "nrps", "dices990": "nrps"}
RULES_FOR = {"chaosnli_snli": ["brier", "ce"], "chaosnli_mnli": ["brier", "ce"], "chaosnli_alphanli": ["brier", "ce"],
             "dices350": ["nrps", "brier", "ce"], "dices990": ["nrps", "brier", "ce"]}
LABELS = {"chaosnli_snli": "ChaosNLI SNLI", "chaosnli_mnli": "ChaosNLI MNLI-m", "chaosnli_alphanli": "ChaosNLI alphaNLI",
          "dices350": "DICES-350", "dices990": "DICES-990"}


def load():
    """-> ({dataset: dict(index, K, ids, groups, counts [N,K], P [N,5,K], ratings [N,R])}, bundle sha256)."""
    bundle, sha = L.load_bundle()
    out = {}
    for si, name in enumerate(DATASETS):
        d = bundle[name]
        P, _ = L.forecast_array(d)
        ids = [str(x) for x in d["ids"]]
        groups = [i.split("#")[0] if name == "chaosnli_snli" else i for i in ids]  # SNLI items that share an image share a group
        out[name] = dict(index=si, K=int(d["K"]), ids=ids, groups=groups, counts=np.asarray(d["counts"], float), P=P,
                         ratings=np.asarray(d["ratings"]))
    return out, sha


def seed(*parts):
    return np.random.SeedSequence([SEED, *parts])


def write_csv(path, rows):
    """Write a list of dicts as CSV with floats at 10 significant digits."""
    import csv

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def valid_ratings(d, i):
    """Individual human ratings of item i (the DICES-990 rows are padded with -1)."""
    r = d["ratings"][i]
    return r[r >= 0]
